import matplotlib
matplotlib.use("TkAgg")

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks, savgol_filter


# ============================================================
# 0. 配置
# ============================================================

SUBJECT_ID = "sub_002"
SCENARIO = ("z4_fwd")

INPUT_DIR = f"./{SUBJECT_ID}"
OUTPUT_DIR = f"./{SUBJECT_ID}_label"

INPUT_FILE = (
    f"{INPUT_DIR}/{SUBJECT_ID}_{SCENARIO}_data.xlsx"
)

OUTPUT_LABEL_FILE = (
    f"{OUTPUT_DIR}/"
    f"{SUBJECT_ID}_{SCENARIO}_aligned_labeled.xlsx"
)

# 当前版本不再单独保存 alignment_events.xlsx
# 只保存最终对齐并带左右总标签的主 Excel。

MIN_PEAK_DIST = 80
MIN_PEAK_PROM = 5
AUTO_ALIGN_MAX_GAP_S = 0.80

PELVIS_SMOOTH_WINDOW = 11
PELVIS_MIN_PROM = 0.5
PELVIS_PROM_FRAC = 0.12
PELVIS_PEAK_DIST = 40

EVENT_COLORS = {
    1: "#1f77b4",
    2: "#9467bd",
    3: "#d4a017",
}

PELVIS_X_MAX_COLOR = "darkorange"
PELVIS_X_MIN_COLOR = "teal"
PELVIS_Y_MAX_COLOR = "purple"
PELVIS_Y_MIN_COLOR = "goldenrod"

LINE_PICK_TOL_PX = 7

FIG_STEP1_SIZE = (15, 12)
FIG_STEP2_SIZE = (18, 10)


# ============================================================
# 1. 共享 segment 状态
# ============================================================

class SegmentState:
    """
    final shift =
        side_base_shift
        + internal_auto_shift
        + internal_manual_shift

    shank:
        internal_auto_shift = 0
        internal_manual_shift = 0
        final shift 只来自 STEP1 的 side_base_shift

    thigh / foot:
        在 shank 的 side_base_shift 基础上，
        再叠加本侧内部自动/人工 shift
    """

    def __init__(
        self,
        seg_name,
        side,
        t,
        y,
        color,
    ):
        self.seg_name = seg_name
        self.side = side
        self.t = np.asarray(t, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.color = color

        self.internal_auto_shift = 0.0
        self.internal_manual_shift = 0.0

        self.lines = []

    def final_shift(self, side_base_shift):
        return (
            float(side_base_shift[self.side])
            + float(self.internal_auto_shift)
            + float(self.internal_manual_shift)
        )

    def display_t(self, side_base_shift):
        return (
            self.t
            + self.final_shift(side_base_shift)
        )

    def add_line(self, ax, lw=1.5):
        line, = ax.plot(
            self.t,
            self.y,
            color=self.color,
            lw=lw,
        )
        self.lines.append(line)
        return line

    def redraw(self, side_base_shift):
        x = self.display_t(side_base_shift)

        for line in self.lines:
            line.set_xdata(x)


# ============================================================
# 2. 共享事件线
# ============================================================

class SharedEvent:
    def __init__(
        self,
        owner_seg,
        side,
        event_id,
        orig_time,
        source="auto",
        visual_only_fixed=False,
    ):
        self.owner_seg = owner_seg
        self.side = side
        self.event_id = int(event_id)
        self.orig_time = float(orig_time)

        # auto   = 程序初始自动检测出来的事件
        # manual = 交互过程中人工新增的事件
        self.source = str(source)

        # visual_only_fixed=True：
        # 只做视觉参考，不参与标签/对齐；
        # 曲线前后移动时这条参考线不跟着移动。
        self.visual_only_fixed = bool(visual_only_fixed)

        # 只要人工拖动过这条事件线，就记为 True
        self.edited = False

        self.artists = []
        self.removed = False

    def add_artist(self, ax):
        line = ax.axvline(
            0,
            color=EVENT_COLORS[self.event_id],
            linestyle="--",
            linewidth=1.8,
            alpha=0.95,
        )

        self.artists.append(
            (ax, line)
        )

    def display_time(
        self,
        segment_states,
        side_base_shift,
    ):
        # 所有事件线都跟随所属 IMU 曲线的 shift 一起移动。
        # 是否参与最终标签由 build_total_label 单独决定。
        state = segment_states[
            self.owner_seg
        ]

        return (
            self.orig_time
            + state.final_shift(
                side_base_shift
            )
        )

    def redraw(
        self,
        segment_states,
        side_base_shift,
    ):
        x = self.display_time(
            segment_states,
            side_base_shift,
        )

        for _, line in self.artists:
            line.set_xdata([x, x])

    def hit(
        self,
        ax,
        mouse_x_pixel,
        segment_states,
        side_base_shift,
    ):
        if self.removed:
            return False

        if not any(
            artist_ax is ax
            for artist_ax, _ in self.artists
        ):
            return False

        x = self.display_time(
            segment_states,
            side_base_shift,
        )

        x_pixel, _ = ax.transData.transform(
            (x, 0)
        )

        return (
            abs(mouse_x_pixel - x_pixel)
            <= LINE_PICK_TOL_PX
        )

    def remove(self):
        for _, line in list(
            self.artists
        ):
            try:
                line.remove()
            except Exception:
                pass

        self.artists.clear()
        self.removed = True


# ============================================================
# 3. 峰值
# ============================================================

def find_event_peaks(
    signal,
    min_dist,
    min_prom,
    mode="max",
):
    x = np.asarray(
        signal,
        dtype=float,
    )

    work = (
        x
        if mode == "max"
        else -x
    )

    idx, _ = find_peaks(
        work,
        distance=min_dist,
        prominence=min_prom,
    )

    return idx


def detect_side_features(
    prefix,
    t,
    df,
):
    """
    图上保留原来的主要事件线：

        thigh Event2  = thigh 局部最小
        shank Event2  = shank 局部最大
        shank Event3  = shank 局部最小
        foot  Event1  = foot 局部最小
        foot  Event3  = foot 局部最大

    但最终左右统一标签只读取：
        foot Event1
        shank Event2
        shank Event3

    thigh Event2 和 foot Event3：
        只作为视觉参考线，
        不参与标签，
        不参与自动对齐，
        不决定任何 shift。
    """

    thigh = df[
        f"{prefix}thigh_angle_x_deg"
    ].to_numpy(dtype=float)

    shank = df[
        f"{prefix}shank_angle_x_deg"
    ].to_numpy(dtype=float)

    foot = df[
        f"{prefix}foot_angle_y_deg"
    ].to_numpy(dtype=float)

    # thigh min -> Event2（视觉参考）
    thigh_min = find_event_peaks(
        thigh,
        MIN_PEAK_DIST,
        MIN_PEAK_PROM,
        "min",
    )

    # shank max -> Event2（最终标签）
    shank_max = find_event_peaks(
        shank,
        MIN_PEAK_DIST,
        MIN_PEAK_PROM,
        "max",
    )

    # shank min -> Event3（最终标签）
    shank_min = find_event_peaks(
        shank,
        MIN_PEAK_DIST,
        MIN_PEAK_PROM,
        "min",
    )

    # foot max -> Event3（视觉参考）
    foot_max = find_event_peaks(
        foot,
        MIN_PEAK_DIST,
        MIN_PEAK_PROM,
        "max",
    )

    # foot min -> Event1（最终标签）
    foot_min = find_event_peaks(
        foot,
        MIN_PEAK_DIST,
        MIN_PEAK_PROM,
        "min",
    )

    return {
        f"{prefix}thigh": {
            "y": thigh,
            "color": "#e15759",
            "events": {
                2: list(t[thigh_min]),
            },
        },

        f"{prefix}shank": {
            "y": shank,
            "color": "#4e79a7",
            "events": {
                2: list(t[shank_max]),
                3: list(t[shank_min]),
            },
        },

        f"{prefix}foot": {
            "y": foot,
            "color": "#59a14f",
            "events": {
                1: list(t[foot_min]),
                3: list(t[foot_max]),
            },
        },
    }


# ============================================================
# 4. 同侧自动初始对齐
# ============================================================

def nearest_diffs(
    reference_times,
    target_times,
    max_gap_s,
):
    ref = np.sort(
        np.asarray(
            reference_times,
            dtype=float,
        )
    )

    tar = np.sort(
        np.asarray(
            target_times,
            dtype=float,
        )
    )

    if len(ref) == 0 or len(tar) == 0:
        return np.array(
            [],
            dtype=float,
        )

    diffs = []

    for tt in tar:
        pos = np.searchsorted(
            ref,
            tt,
        )

        candidates = []

        if pos > 0:
            candidates.append(
                ref[pos - 1]
            )

        if pos < len(ref):
            candidates.append(
                ref[pos]
            )

        if not candidates:
            continue

        rr = min(
            candidates,
            key=lambda v: abs(v - tt),
        )

        diff = rr - tt

        if (
            abs(diff)
            <= max_gap_s
        ):
            diffs.append(diff)

    return np.asarray(
        diffs,
        dtype=float,
    )


def median_shift(
    reference_times,
    target_times,
):
    diffs = nearest_diffs(
        reference_times,
        target_times,
        AUTO_ALIGN_MAX_GAP_S,
    )

    if len(diffs) == 0:
        return (
            0.0,
            0,
            np.nan,
        )

    shift = float(
        np.median(diffs)
    )

    mad = float(
        np.median(
            np.abs(
                diffs - shift
            )
        )
    )

    return (
        shift,
        len(diffs),
        mad,
    )


def initialize_internal_shifts(
    side,
    prefix,
    feature_cfg,
    segment_states,
):
    """
    不再使用其他 IMU 的事件位置进行自动内部对齐。

    原则：
        STEP1:
            pelvis 固定
            用户拖左/右 shank，确定左右侧相对 pelvis 的位置

        STEP2:
            shank 固定
            用户直接拖 thigh / foot 曲线去对齐 shank

    因此：
        thigh internal auto shift = 0
        shank internal auto shift = 0
        foot  internal auto shift = 0

    最终移动多少，只取决于用户实际拖动曲线得到的 shift。
    """

    thigh_seg = f"{prefix}thigh"
    shank_seg = f"{prefix}shank"
    foot_seg = f"{prefix}foot"

    for seg in (
        thigh_seg,
        shank_seg,
        foot_seg,
    ):
        segment_states[
            seg
        ].internal_auto_shift = 0.0

    return [
        {
            "Side": side,
            "Reference": shank_seg,
            "Target": thigh_seg,
            "Event": "NONE",
            "Matched": 0,
            "AutoInternalShift_s": 0.0,
            "MAD_s": np.nan,
            "Note": "manual curve alignment only; event positions not used",
        },
        {
            "Side": side,
            "Reference": shank_seg,
            "Target": foot_seg,
            "Event": "NONE",
            "Matched": 0,
            "AutoInternalShift_s": 0.0,
            "MAD_s": np.nan,
            "Note": "manual curve alignment only; event positions not used",
        },
    ]


# ============================================================
# 5. pelvis 辅助峰谷
# ============================================================

def smooth_signal(x):
    y = np.asarray(
        x,
        dtype=float,
    )

    if len(y) < 7:
        return y.copy()

    y = (
        pd.Series(y)
        .interpolate(
            limit_direction="both"
        )
        .to_numpy(dtype=float)
    )

    w = int(
        PELVIS_SMOOTH_WINDOW
    )

    if w % 2 == 0:
        w += 1

    if w >= len(y):
        w = len(y) - 1

    if w % 2 == 0:
        w -= 1

    if w < 5:
        return y

    return savgol_filter(
        y,
        window_length=w,
        polyorder=2,
        mode="interp",
    )


def pelvis_extrema(y):
    ys = smooth_signal(y)

    finite = ys[
        np.isfinite(ys)
    ]

    if len(finite) < 5:
        prom = PELVIS_MIN_PROM
    else:
        robust_range = (
            np.percentile(
                finite,
                95,
            )
            - np.percentile(
                finite,
                5,
            )
        )

        prom = max(
            PELVIS_MIN_PROM,
            PELVIS_PROM_FRAC
            * robust_range,
        )

    max_idx, _ = find_peaks(
        ys,
        distance=PELVIS_PEAK_DIST,
        prominence=prom,
    )

    min_idx, _ = find_peaks(
        -ys,
        distance=PELVIS_PEAK_DIST,
        prominence=prom,
    )

    return (
        max_idx,
        min_idx,
        prom,
    )


# ============================================================
# 6. 保存用 shift / 标签工具
# ============================================================

def sample_dt(t):
    d = np.diff(
        np.asarray(
            t,
            dtype=float,
        )
    )

    d = d[
        np.isfinite(d)
        & (d > 0)
    ]

    if len(d) == 0:
        raise ValueError(
            "Cannot determine the sampling interval from time_s"
        )

    return float(
        np.median(d)
    )


def shift_seconds_to_samples(
    shift_s,
    dt,
):
    return int(
        np.rint(
            float(shift_s)
            / float(dt)
        )
    )


def shift_sensor_columns(
    df,
    prefix,
    shift_samples,
):
    """
    prefix 示例：
        l_foot
        l_shank
        pelvis

    该 IMU 的所有数据列一起移动。
    """

    cols = [
        c
        for c in df.columns
        if c.startswith(
            prefix + "_"
        )
    ]

    if not cols:
        raise KeyError(
            f"No {prefix}_* data columns were found"
        )

    if shift_samples != 0:
        df.loc[:, cols] = (
            df.loc[:, cols]
            .shift(shift_samples)
        )

    return cols


def common_trim_range(
    n_rows,
    shifts,
):
    """
    只删除由于 shift 新产生的首尾空缺。
    """

    shifts = list(shifts)

    trim_start = max(
        [0]
        + [
            s
            for s in shifts
            if s > 0
        ]
    )

    most_negative = min(
        [0]
        + [
            s
            for s in shifts
            if s < 0
        ]
    )

    trim_end = (
        n_rows
        + most_negative
    )

    if trim_start >= trim_end:
        raise ValueError(
            "The shift leaves no common valid data interval"
        )

    return (
        int(trim_start),
        int(trim_end),
    )


def build_total_label(
    t,
    side,
    prefix,
    events,
    applied_shift_s,
):
    """
    ==========================================================
    左/右侧最终统一标签 —— 连续贴标版
    ==========================================================

    最终规则严格为：

        数据开始 -> 第一个 shank Event3      = 0

        Event3 -> Event1                    = 3
        Event1 -> Event2                    = 1
        Event2 -> 下一个 Event3              = 2

        然后继续：
        Event3 -> Event1                    = 3
        Event1 -> Event2                    = 1
        Event2 -> Event3                    = 2
        ...

        最后一个 Event3 -> 最后一个 Event1   = 3
        最后一个 Event1 -> 数据结束           = 0


    Event 来源固定：
        Event1 = foot
        Event2 = shank
        Event3 = shank


    【重要修正】
    旧版的问题：
        如果某个 E3-to-E3 周期没有严格找到 E1 或 E2，
        代码会 continue。
        由于 labels 初始值是 0，
        因此会在标签中间产生一整段 0。

    本版改成：
        标签按“实际存在的事件线”连续传播。

        遇到 Event3 -> 从该时刻开始贴 3
        遇到 Event1 -> 从该时刻开始贴 1
        遇到 Event2 -> 从该时刻开始贴 2
        遇到下一个 Event3 -> 再改成 3

    因此：
        第一个 Event3 到最后一个 Event1 之间绝不允许出现 0。

    同时：
        最后一个 Event3 到最后一个 Event1 强制贴 3，
        最后一个 Event1 及以后强制贴 0。
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    labels = np.zeros(
        len(t),
        dtype=int,
    )

    foot_seg = f"{prefix}foot"
    shank_seg = f"{prefix}shank"

    valid = [
        ev
        for ev in events
        if (
            not ev.removed
            and ev.side == side
        )
    ]

    def aligned_time(ev):
        return (
            float(ev.orig_time)
            + float(
                applied_shift_s[
                    ev.owner_seg
                ]
            )
        )

    # --------------------------------------------------------
    # 1. 只取最终总标签真正需要的事件
    # --------------------------------------------------------

    e1_items = sorted(
        [
            (
                aligned_time(ev),
                ev,
            )
            for ev in valid
            if (
                ev.owner_seg == foot_seg
                and ev.event_id == 1
            )
        ],
        key=lambda x: x[0],
    )

    e2_items = sorted(
        [
            (
                aligned_time(ev),
                ev,
            )
            for ev in valid
            if (
                ev.owner_seg == shank_seg
                and ev.event_id == 2
            )
        ],
        key=lambda x: x[0],
    )

    e3_items = sorted(
        [
            (
                aligned_time(ev),
                ev,
            )
            for ev in valid
            if (
                ev.owner_seg == shank_seg
                and ev.event_id == 3
            )
        ],
        key=lambda x: x[0],
    )

    qc_rows = []
    unified_rows = []
    used_events = set()

    # --------------------------------------------------------
    # 2. 必须至少有 E3 和 E1
    # --------------------------------------------------------

    if len(e3_items) == 0:
        qc_rows.append({
            "Side": side,
            "Status": "INVALID",
            "Reason": "missing shank Event3",
        })

        return (
            labels,
            unified_rows,
            qc_rows,
            used_events,
        )

    if len(e1_items) == 0:
        qc_rows.append({
            "Side": side,
            "Status": "INVALID",
            "Reason": "missing foot Event1",
        })

        return (
            labels,
            unified_rows,
            qc_rows,
            used_events,
        )

    # 第一个有效边界：第一个 shank E3
    first_e3_time, first_e3_ev = (
        e3_items[0]
    )

    # “最后一个 Event1”必须是在第一个 E3 后面的 Event1
    e1_after_first_e3 = [
        item
        for item in e1_items
        if item[0] > first_e3_time
    ]

    if not e1_after_first_e3:
        qc_rows.append({
            "Side": side,
            "Status": "INVALID",
            "Reason": "no foot Event1 after first shank Event3",
            "FirstEvent3_s": first_e3_time,
        })

        return (
            labels,
            unified_rows,
            qc_rows,
            used_events,
        )

    last_e1_time, last_e1_ev = (
        e1_after_first_e3[-1]
    )

    # --------------------------------------------------------
    # 3. 找最后一个 Event1 前面的最后一个 shank Event3
    #
    #    这段必须严格：
    #       terminal E3 -> last E1 = 3
    # --------------------------------------------------------

    terminal_e3_candidates = [
        item
        for item in e3_items
        if (
            first_e3_time
            <= item[0]
            < last_e1_time
        )
    ]

    if not terminal_e3_candidates:
        qc_rows.append({
            "Side": side,
            "Status": "INVALID",
            "Reason": "no shank Event3 before last foot Event1",
            "LastEvent1_s": last_e1_time,
        })

        return (
            labels,
            unified_rows,
            qc_rows,
            used_events,
        )

    terminal_e3_time, terminal_e3_ev = (
        terminal_e3_candidates[-1]
    )

    # --------------------------------------------------------
    # 4. 构造“连续状态事件”
    #
    #    first E3 到 terminal E3 之间：
    #        E3 -> state 3
    #        E1 -> state 1
    #        E2 -> state 2
    #
    #    不再因为缺某个事件而把一整个周期留成 0。
    # --------------------------------------------------------

    state_events = []

    # Event3
    for tt, ev in e3_items:
        if (
            first_e3_time
            <= tt
            < terminal_e3_time
        ):
            state_events.append(
                (
                    tt,
                    3,
                    ev,
                    shank_seg,
                )
            )

    # Event1
    for tt, ev in e1_items:
        if (
            first_e3_time
            < tt
            < terminal_e3_time
        ):
            state_events.append(
                (
                    tt,
                    1,
                    ev,
                    foot_seg,
                )
            )

    # Event2
    for tt, ev in e2_items:
        if (
            first_e3_time
            < tt
            < terminal_e3_time
        ):
            state_events.append(
                (
                    tt,
                    2,
                    ev,
                    shank_seg,
                )
            )

    # 同一时刻若出现不同事件，
    # 按正常相位顺序 E3 -> E1 -> E2 排序。
    # 因为后一个状态会覆盖该时刻之后的标签。
    event_priority = {
        3: 0,
        1: 1,
        2: 2,
    }

    state_events = sorted(
        state_events,
        key=lambda x: (
            x[0],
            event_priority[
                x[1]
            ],
        ),
    )

    # 理论上第一个一定应该是 first E3。
    # 为保险起见显式加入。
    if (
        not state_events
        or not np.isclose(
            state_events[0][0],
            first_e3_time,
            rtol=0.0,
            atol=1e-9,
        )
        or state_events[0][1] != 3
    ):
        state_events.insert(
            0,
            (
                first_e3_time,
                3,
                first_e3_ev,
                shank_seg,
            ),
        )

    # --------------------------------------------------------
    # 5. 从第一个 E3 开始连续贴标签
    #
    #    每一条事件线只负责把“当前状态”改成：
    #        E3 => 3
    #        E1 => 1
    #        E2 => 2
    #
    #    一直持续到下一条有效事件。
    # --------------------------------------------------------

    cycle_index = 0
    previous_event_id = None

    for i, (
        event_time,
        event_id,
        ev,
        source_seg,
    ) in enumerate(
        state_events
    ):

        if event_id == 3:
            cycle_index += 1

        if i + 1 < len(
            state_events
        ):
            next_time = (
                state_events[
                    i + 1
                ][0]
            )
        else:
            next_time = (
                terminal_e3_time
            )

        # 如果同一时刻多条线，
        # 前面的零长度区间自动跳过。
        if next_time > event_time:
            labels[
                (t >= event_time)
                & (t < next_time)
            ] = event_id

        used_events.add(ev)

        unified_rows.append({
            "Side": side,
            "Cycle": max(
                cycle_index,
                1,
            ),
            "Event": event_id,
            "SourceSegment": source_seg,
            "EventSource": ev.source,
            "Edited": bool(
                ev.edited
            ),
            "Time_s": event_time,
        })

        # ----------------------------------------------------
        # QC 只报警，不再把标签清零
        # ----------------------------------------------------
        if previous_event_id is not None:
            expected_next = {
                3: 1,
                1: 2,
                2: 3,
            }

            expected = (
                expected_next[
                    previous_event_id
                ]
            )

            if event_id == previous_event_id:
                qc_rows.append({
                    "Side": side,
                    "Time_s": event_time,
                    "Status": "WARNING",
                    "Reason": (
                        f"duplicate consecutive Event"
                        f"{event_id}; labels kept continuous"
                    ),
                })

            elif event_id != expected:
                qc_rows.append({
                    "Side": side,
                    "Time_s": event_time,
                    "PreviousEvent": (
                        previous_event_id
                    ),
                    "CurrentEvent": event_id,
                    "ExpectedEvent": expected,
                    "Status": "WARNING",
                    "Reason": (
                        "event sequence not 3->1->2->3; "
                        "labels were still propagated continuously"
                    ),
                })

        previous_event_id = (
            event_id
        )

    # --------------------------------------------------------
    # 6. 最终不完整周期：
    #
    #    最后一个 shank E3 -> 最后一个 foot E1 = 3
    # --------------------------------------------------------

    labels[
        (t >= terminal_e3_time)
        & (t < last_e1_time)
    ] = 3

    used_events.add(
        terminal_e3_ev
    )

    used_events.add(
        last_e1_ev
    )

    cycle_index += 1

    unified_rows.append({
        "Side": side,
        "Cycle": cycle_index,
        "Event": 3,
        "SourceSegment": shank_seg,
        "EventSource": terminal_e3_ev.source,
        "Edited": bool(
            terminal_e3_ev.edited
        ),
        "Time_s": terminal_e3_time,
    })

    unified_rows.append({
        "Side": side,
        "Cycle": cycle_index,
        "Event": 1,
        "SourceSegment": foot_seg,
        "EventSource": last_e1_ev.source,
        "Edited": bool(
            last_e1_ev.edited
        ),
        "Time_s": last_e1_time,
    })

    qc_rows.append({
        "Side": side,
        "Cycle": cycle_index,
        "StartE3_s": terminal_e3_time,
        "LastEvent1_s": last_e1_time,
        "Status": "TERMINAL",
        "Reason": (
            "terminal interval: "
            "last Event3 -> last Event1 = 3; "
            "last Event1 -> end = 0"
        ),
    })

    # --------------------------------------------------------
    # 7. 两端强制为 0
    # --------------------------------------------------------

    labels[
        t < first_e3_time
    ] = 0

    labels[
        t >= last_e1_time
    ] = 0

    # --------------------------------------------------------
    # 8. 关键安全检查：
    #    first E3 到 last E1 之间绝不允许再出现 0
    # --------------------------------------------------------

    active_mask = (
        (t >= first_e3_time)
        & (t < last_e1_time)
    )

    internal_zero_count = int(
        np.sum(
            labels[
                active_mask
            ] == 0
        )
    )

    if internal_zero_count > 0:
        zero_times = t[
            active_mask
        ][
            labels[
                active_mask
            ] == 0
        ]

        raise RuntimeError(
            f"[{side}] 标签逻辑错误："
            f"第一个Event3({first_e3_time:.3f}s)"
            f"到最后一个Event1({last_e1_time:.3f}s)"
            f"之间仍有 {internal_zero_count} 个0。"
            f"第一个异常时间={zero_times[0]:.3f}s"
        )

    qc_rows.append({
        "Side": side,
        "FirstEvent3_s": first_e3_time,
        "LastEvent1_s": last_e1_time,
        "InternalZeroCount": (
            internal_zero_count
        ),
        "Status": "LABEL_RANGE_OK",
        "Reason": (
            "no zero labels between "
            "first Event3 and last Event1"
        ),
    })

    return (
        labels,
        unified_rows,
        qc_rows,
        used_events,
    )


# ============================================================
# 7. 创建 UI
# ============================================================

def build_ui(
    t,
    df,
):
    required = [
        "l_thigh_angle_x_deg",
        "l_shank_angle_x_deg",
        "l_foot_angle_y_deg",
        "r_thigh_angle_x_deg",
        "r_shank_angle_x_deg",
        "r_foot_angle_y_deg",
        "pelvis_angle_x_deg",
        "pelvis_angle_y_deg",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            "Missing columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # 数据 + shared states
    # --------------------------------------------------------

    left_cfg = detect_side_features(
        "l_",
        t,
        df,
    )

    right_cfg = detect_side_features(
        "r_",
        t,
        df,
    )

    all_cfg = {
        **left_cfg,
        **right_cfg,
    }

    segment_states = {}

    for seg_name, cfg in all_cfg.items():
        side = (
            "left"
            if seg_name.startswith("l_")
            else "right"
        )

        segment_states[
            seg_name
        ] = SegmentState(
            seg_name=seg_name,
            side=side,
            t=t,
            y=cfg["y"],
            color=cfg["color"],
        )

    # STEP1 的 shank -> pelvis 左右整体基准 shift
    side_base_shift = {
        "left": 0.0,
        "right": 0.0,
    }

    # 单腿内部自动初始 shift
    auto_qc = []

    auto_qc.extend(
        initialize_internal_shifts(
            "left",
            "l_",
            left_cfg,
            segment_states,
        )
    )

    auto_qc.extend(
        initialize_internal_shifts(
            "right",
            "r_",
            right_cfg,
            segment_states,
        )
    )

    # --------------------------------------------------------
    # STEP1：pelvis固定，左右 shank 移动
    # --------------------------------------------------------

    fig1, ax1 = plt.subplots(
        4,
        1,
        figsize=FIG_STEP1_SIZE,
        sharex=True,
    )

    try:
        fig1.canvas.manager.set_window_title(
            "STEP1 - Align left and right shanks to the fixed pelvis"
        )
    except Exception:
        pass

    plt.figure(fig1.number)
    plt.subplots_adjust(
        hspace=0.30,
        top=0.89,
        bottom=0.07,
        left=0.08,
        right=0.98,
    )

    segment_states[
        "l_shank"
    ].add_line(
        ax1[0],
        lw=1.7,
    )

    segment_states[
        "r_shank"
    ].add_line(
        ax1[1],
        lw=1.7,
    )

    pelvis_x = df[
        "pelvis_angle_x_deg"
    ].to_numpy(dtype=float)

    pelvis_y = df[
        "pelvis_angle_y_deg"
    ].to_numpy(dtype=float)

    pelvis_x_line, = ax1[2].plot(
        t,
        pelvis_x,
        color="#f28e2b",
        lw=1.6,
    )

    pelvis_y_line, = ax1[3].plot(
        t,
        pelvis_y,
        color="#b07aa1",
        lw=1.6,
    )

    titles1 = [
        "LEFT SHANK — drag horizontally",
        "RIGHT SHANK — drag horizontally",
        "PELVIS X — fixed",
        "PELVIS Y — fixed",
    ]

    for a, title in zip(
        ax1,
        titles1,
    ):
        a.set_title(
            title,
            loc="left",
            fontsize=10,
            fontweight="bold",
        )

        a.set_ylabel(
            "Angle (deg)"
        )

        # STEP1（含 pelvis）每个子图都显示时间坐标和刻度
        a.set_xlabel(
            "Time (s)"
        )

        a.tick_params(
            axis="x",
            which="both",
            labelbottom=True,
        )

        a.grid(
            True,
            alpha=0.3,
        )

    # pelvis extrema
    px_max, px_min, px_prom = (
        pelvis_extrema(
            pelvis_x
        )
    )

    py_max, py_min, py_prom = (
        pelvis_extrema(
            pelvis_y
        )
    )

    for idx in px_max:
        ax1[2].axvline(
            t[idx],
            color=PELVIS_X_MAX_COLOR,
            ls="--",
            lw=1.5,
            alpha=0.8,
        )

    for idx in px_min:
        ax1[2].axvline(
            t[idx],
            color=PELVIS_X_MIN_COLOR,
            ls="--",
            lw=1.5,
            alpha=0.8,
        )

    for idx in py_max:
        ax1[3].axvline(
            t[idx],
            color=PELVIS_Y_MAX_COLOR,
            ls="--",
            lw=1.5,
            alpha=0.8,
        )

    for idx in py_min:
        ax1[3].axvline(
            t[idx],
            color=PELVIS_Y_MIN_COLOR,
            ls="--",
            lw=1.5,
            alpha=0.8,
        )

    fig1.suptitle(
        f"{SUBJECT_ID} | {SCENARIO}\n"
        "STEP1：Pelvis固定；拖 L/R shank 做左右整体对齐。"
        "点击shank事件线=上下4图同步阴影；[S]保存",
        fontsize=10,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # STEP1 事件选中阴影：L shank / R shank / Pelvis X / Pelvis Y
    # --------------------------------------------------------
    STEP1_SHADOW_HALF_WIDTH_S = 0.03

    step1_shadow_patches = []

    for a in ax1:
        patch = a.axvspan(
            -1.0,
            -1.0,
            color="black",
            alpha=0.10,
            visible=False,
            zorder=0,
        )
        step1_shadow_patches.append(patch)

    def _update_span_patch(patch, x0, x1):
        # 兼容不同 matplotlib 版本中 axvspan 返回的 Rectangle / Polygon
        if hasattr(patch, "set_x") and hasattr(patch, "set_width"):
            patch.set_x(x0)
            patch.set_width(x1 - x0)
        elif hasattr(patch, "set_xy"):
            patch.set_xy([
                [x0, 0],
                [x0, 1],
                [x1, 1],
                [x1, 0],
                [x0, 0],
            ])

    def show_step1_shadow(x):
        x0 = x - STEP1_SHADOW_HALF_WIDTH_S
        x1 = x + STEP1_SHADOW_HALF_WIDTH_S

        for patch in step1_shadow_patches:
            _update_span_patch(
                patch,
                x0,
                x1,
            )
            patch.set_visible(True)

    # --------------------------------------------------------
    # STEP2：单腿内部
    # --------------------------------------------------------

    fig2, ax2 = plt.subplots(
        3,
        2,
        figsize=FIG_STEP2_SIZE,
        sharex=True,
    )

    try:
        fig2.canvas.manager.set_window_title(
            "STEP2 - Align segments within each leg"
        )
    except Exception:
        pass

    plt.figure(fig2.number)
    plt.subplots_adjust(
        hspace=0.06,
        wspace=0.10,
        top=0.88,
        bottom=0.08,
        left=0.07,
        right=0.98,
    )

    seg_axis = {
        "l_thigh": ax2[0, 0],
        "r_thigh": ax2[0, 1],
        "l_shank": ax2[1, 0],
        "r_shank": ax2[1, 1],
        "l_foot": ax2[2, 0],
        "r_foot": ax2[2, 1],
    }

    for seg_name, a in seg_axis.items():
        segment_states[
            seg_name
        ].add_line(
            a,
            lw=1.5,
        )

        a.set_ylabel(
            f"{seg_name}\n(deg)"
        )

        a.grid(
            True,
            alpha=0.3,
        )

    ax2[0, 0].set_title(
        "LEFT LEG",
        fontweight="bold",
    )

    ax2[0, 1].set_title(
        "RIGHT LEG",
        fontweight="bold",
    )

    # STEP2 只保留最下面一排横坐标
    ax2[2, 0].set_xlabel(
        "Time (s)"
    )

    ax2[2, 1].set_xlabel(
        "Time (s)"
    )

    fig2.suptitle(
        f"{SUBJECT_ID} | {SCENARIO}\n"
        "STEP2：Shank固定；拖 thigh/foot 做单腿内部对齐。"
        "点击任意事件线=同侧上下3图同步阴影；[S]保存",
        fontsize=10,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # STEP2 事件选中阴影：
    # 左侧事件 -> 左 thigh/shank/foot 三图同时显示
    # 右侧事件 -> 右 thigh/shank/foot 三图同时显示
    # --------------------------------------------------------
    STEP2_SHADOW_HALF_WIDTH_S = 0.03

    step2_shadow_patches = {
        "left": [],
        "right": [],
    }

    for row in range(3):
        left_patch = ax2[row, 0].axvspan(
            -1.0,
            -1.0,
            color="black",
            alpha=0.10,
            visible=False,
            zorder=0,
        )

        right_patch = ax2[row, 1].axvspan(
            -1.0,
            -1.0,
            color="black",
            alpha=0.10,
            visible=False,
            zorder=0,
        )

        step2_shadow_patches[
            "left"
        ].append(left_patch)

        step2_shadow_patches[
            "right"
        ].append(right_patch)

    def show_step2_shadow(side, x):
        x0 = x - STEP2_SHADOW_HALF_WIDTH_S
        x1 = x + STEP2_SHADOW_HALF_WIDTH_S

        for patch in step2_shadow_patches[side]:
            _update_span_patch(
                patch,
                x0,
                x1,
            )
            patch.set_visible(True)

    # 记录当前最后点击/选中的事件。
    # 曲线继续移动时，阴影会跟着这条事件线更新。
    selected_event_state = {
        "event": None,
    }

    def show_event_shadow(ev):
        if ev is None or ev.removed:
            return

        x = ev.display_time(
            segment_states,
            side_base_shift,
        )

        show_step2_shadow(
            ev.side,
            x,
        )

        # shank 事件在线路1也有对应显示，因此同时更新 pelvis 窗口的4图阴影
        if ev.owner_seg.endswith("shank"):
            show_step1_shadow(x)

    # --------------------------------------------------------
    # Shared Events
    # --------------------------------------------------------

    events = []

    def add_initial_event(
        seg_name,
        event_id,
        event_time,
    ):
        side = (
            "left"
            if seg_name.startswith("l_")
            else "right"
        )

        # 所有原始事件线都随各自曲线一起移动。
        # thigh E2 / foot E3 仍只做参考，不参与最终总标签。
        ev = SharedEvent(
            owner_seg=seg_name,
            side=side,
            event_id=event_id,
            orig_time=event_time,
            source="auto",
            visual_only_fixed=False,
        )

        # STEP2 总是显示
        ev.add_artist(
            seg_axis[seg_name]
        )

        # shank 在 STEP1 也同步显示
        if seg_name == "l_shank":
            ev.add_artist(
                ax1[0]
            )

        elif seg_name == "r_shank":
            ev.add_artist(
                ax1[1]
            )

        events.append(ev)

    for seg_name, cfg in all_cfg.items():
        for event_id, times in (
            cfg["events"].items()
        ):
            for tt in times:
                add_initial_event(
                    seg_name,
                    event_id,
                    tt,
                )

    # --------------------------------------------------------
    # redraw helpers
    # --------------------------------------------------------

    def redraw_segment(
        seg_name,
    ):
        segment_states[
            seg_name
        ].redraw(
            side_base_shift
        )

        for ev in events:
            if (
                not ev.removed
                and ev.owner_seg
                == seg_name
            ):
                ev.redraw(
                    segment_states,
                    side_base_shift,
                )

    def redraw_side(side):
        prefix = (
            "l_"
            if side == "left"
            else "r_"
        )

        for seg in (
            f"{prefix}thigh",
            f"{prefix}shank",
            f"{prefix}foot",
        ):
            redraw_segment(seg)

    redraw_side("left")
    redraw_side("right")

    # --------------------------------------------------------
    # legends
    # --------------------------------------------------------

    event_legend = [
        Line2D(
            [0],
            [0],
            color=EVENT_COLORS[i],
            ls="--",
            label=f"Event {i}",
        )
        for i in (1, 2, 3)
    ]

    fig1.legend(
        handles=event_legend,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        fontsize=8,
    )

    fig2.legend(
        handles=event_legend,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=8,
    )

    # ========================================================
    # STEP1 鼠标：shank动，pelvis不动
    # ========================================================

    step1_state = {
        "drag_side": None,
        "start_x": None,
        "start_shift": None,
        "active_event": None,
    }

    def step1_axis_role(a):
        if a is ax1[0]:
            return "left"

        if a is ax1[1]:
            return "right"

        if (
            a is ax1[2]
            or a is ax1[3]
        ):
            return "pelvis"

        return None

    def on_step1_press(evt):
        if (
            evt.inaxes is None
            or evt.xdata is None
        ):
            return

        role = step1_axis_role(
            evt.inaxes
        )

        if role is None:
            return

        # pelvis 固定
        if role == "pelvis":
            if evt.button == 1:
                print(
                    "[STEP1] The pelvis is a fixed reference and cannot be moved."
                )
            return

        # shank event 删除
        if evt.button == 3:
            for ev in list(events):
                if (
                    not ev.removed
                    and ev.side == role
                    and ev.owner_seg.endswith(
                        "shank"
                    )
                    and ev.hit(
                        evt.inaxes,
                        evt.x,
                        segment_states,
                        side_base_shift,
                    )
                ):
                    ev.remove()
                    events.remove(ev)

                    fig1.canvas.draw_idle()
                    fig2.canvas.draw_idle()
                    return

            return

        if evt.button != 1:
            return

        # 优先拖事件线
        for ev in events:
            if (
                not ev.removed
                and ev.side == role
                and ev.owner_seg.endswith(
                    "shank"
                )
                and ev.hit(
                    evt.inaxes,
                    evt.x,
                    segment_states,
                    side_base_shift,
                )
            ):
                step1_state[
                    "active_event"
                ] = ev

                selected_event_state[
                    "event"
                ] = ev

                show_event_shadow(ev)

                fig1.canvas.draw_idle()
                fig2.canvas.draw_idle()
                return

        # 否则拖整侧 shank 基准
        step1_state[
            "drag_side"
        ] = role

        step1_state[
            "start_x"
        ] = evt.xdata

        step1_state[
            "start_shift"
        ] = side_base_shift[
            role
        ]

        print(
            f"[STEP1] 开始移动 {role} shank；Pelvis固定"
        )

    def on_step1_motion(evt):
        if evt.xdata is None:
            return

        active_event = (
            step1_state[
                "active_event"
            ]
        )

        if active_event is not None:
            seg_state = segment_states[
                active_event.owner_seg
            ]

            active_event.orig_time = (
                evt.xdata
                - seg_state.final_shift(
                    side_base_shift
                )
            )
            active_event.edited = True

            active_event.redraw(
                segment_states,
                side_base_shift,
            )

            show_event_shadow(
                active_event
            )

            fig1.canvas.draw_idle()
            fig2.canvas.draw_idle()
            return

        side = step1_state[
            "drag_side"
        ]

        if side is None:
            return

        delta = (
            evt.xdata
            - step1_state[
                "start_x"
            ]
        )

        side_base_shift[side] = (
            step1_state[
                "start_shift"
            ]
            + delta
        )

        # 整侧基础 shift 同步：
        # shank + thigh + foot 全部跟随
        redraw_side(side)

        selected_ev = selected_event_state[
            "event"
        ]

        if (
            selected_ev is not None
            and not selected_ev.removed
            and selected_ev.side == side
        ):
            show_event_shadow(
                selected_ev
            )

        fig1.canvas.draw_idle()
        fig2.canvas.draw_idle()

    def on_step1_release(evt):
        if (
            step1_state[
                "active_event"
            ]
            is not None
        ):
            step1_state[
                "active_event"
            ] = None

        side = step1_state[
            "drag_side"
        ]

        if side is not None:
            print(
                f"[STEP1] {side} side base shift = "
                f"{side_base_shift[side]:+.3f}s"
            )

        step1_state[
            "drag_side"
        ] = None

    # ========================================================
    # STEP2 鼠标：shank固定，thigh/foot动
    # ========================================================

    step2_state = {
        "mode": "shift",
        "active_seg": None,
        "start_x": None,
        "start_internal_manual": None,
        "active_event": None,
    }

    # ========================================================
    # STEP2 最终裁剪范围（两条灰线）
    # 左右灰线在所有子图同步显示/移动
    # 保存时删除灰线外数据
    # ========================================================
    crop_state = {
        "left": float(t[0]),
        "right": float(t[-1]),
        "drag": None,
    }

    crop_lines = {
        "left": [],
        "right": [],
    }

    for _, a in seg_axis.items():
        crop_lines["left"].append(
            a.axvline(
                crop_state["left"],
                color="gray",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
            )
        )
        crop_lines["right"].append(
            a.axvline(
                crop_state["right"],
                color="gray",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
            )
        )

    def update_crop_lines():
        for line in crop_lines["left"]:
            line.set_xdata(
                [crop_state["left"], crop_state["left"]]
            )
        for line in crop_lines["right"]:
            line.set_xdata(
                [crop_state["right"], crop_state["right"]]
            )
        fig2.canvas.draw_idle()

    def hit_crop_line(evt):
        if evt.inaxes not in ax_to_seg:
            return None

        for side in ("left", "right"):
            xpix, _ = evt.inaxes.transData.transform(
                (crop_state[side], 0)
            )
            if abs(evt.x - xpix) <= 10:
                return side
        return None

    ax_to_seg = {
        a: seg
        for seg, a in seg_axis.items()
    }

    # 只有最终总标签真正需要的事件才允许人工添加
    allowed_event_ids = {
        "thigh": set(),
        "shank": {2, 3},
        "foot": {1},
    }

    def seg_kind(seg_name):
        if seg_name.endswith(
            "thigh"
        ):
            return "thigh"

        if seg_name.endswith(
            "shank"
        ):
            return "shank"

        return "foot"

    def on_step2_key(evt):
        if evt.key in (
            "1",
            "2",
            "3",
        ):
            step2_state["mode"] = int(
                evt.key
            )

            print(
                f"[STEP2] Event {evt.key} 添加模式"
            )

        elif evt.key in (
            "0",
            "escape",
        ):
            step2_state[
                "mode"
            ] = "shift"

            print(
                "[STEP2] 曲线平移模式"
            )

        elif evt.key in (
            "s",
            "S",
        ):
            save_result(
                df=df,
                t=t,
                segment_states=segment_states,
                side_base_shift=side_base_shift,
                events=events,
                auto_qc=auto_qc,
                pelvis_qc={
                    "PelvisXProminence": px_prom,
                    "PelvisYProminence": py_prom,
                },
                crop_state=crop_state,
            )

    def on_step2_press(evt):
        crop_side = hit_crop_line(evt)
        if crop_side is not None and evt.button == 1:
            crop_state["drag"] = crop_side
            return

        if (
            evt.inaxes is None
            or evt.xdata is None
            or evt.inaxes
            not in ax_to_seg
        ):
            return

        seg_name = ax_to_seg[
            evt.inaxes
        ]

        state = segment_states[
            seg_name
        ]

        # right click delete event
        if evt.button == 3:
            for ev in list(events):
                if (
                    not ev.removed
                    and ev.owner_seg
                    == seg_name
                    and ev.hit(
                        evt.inaxes,
                        evt.x,
                        segment_states,
                        side_base_shift,
                    )
                ):
                    ev.remove()
                    events.remove(ev)

                    if selected_event_state["event"] is ev:
                        selected_event_state["event"] = None

                    fig1.canvas.draw_idle()
                    fig2.canvas.draw_idle()
                    return

            return

        if evt.button != 1:
            return

        # 优先检查已有事件：
        # 所有事件线都可点击，点击后同侧上下3图显示阴影。
        for ev in events:
            if (
                not ev.removed
                and ev.owner_seg
                == seg_name
                and ev.hit(
                    evt.inaxes,
                    evt.x,
                    segment_states,
                    side_base_shift,
                )
            ):
                step2_state[
                    "active_event"
                ] = ev

                selected_event_state[
                    "event"
                ] = ev

                show_event_shadow(ev)

                fig1.canvas.draw_idle()
                fig2.canvas.draw_idle()
                return

        # 添加事件
        if step2_state[
            "mode"
        ] in (
            1,
            2,
            3,
        ):
            event_id = step2_state[
                "mode"
            ]

            kind = seg_kind(
                seg_name
            )

            if event_id not in (
                allowed_event_ids[
                    kind
                ]
            ):
                print(
                    f"[STEP2] {seg_name} 不使用 Event {event_id}。"
                    "最终标签只允许：foot=Event1；shank=Event2/Event3；"
                    "thigh不参与事件标签。"
                )
                return

            orig_time = (
                evt.xdata
                - state.final_shift(
                    side_base_shift
                )
            )

            ev = SharedEvent(
                owner_seg=seg_name,
                side=state.side,
                event_id=event_id,
                orig_time=orig_time,
                source="manual",
            )

            ev.add_artist(
                evt.inaxes
            )

            # 新增 shank 事件时，STEP1同步
            if seg_name == "l_shank":
                ev.add_artist(
                    ax1[0]
                )

            elif seg_name == "r_shank":
                ev.add_artist(
                    ax1[1]
                )

            events.append(ev)

            ev.redraw(
                segment_states,
                side_base_shift,
            )

            selected_event_state[
                "event"
            ] = ev

            show_event_shadow(ev)

            fig1.canvas.draw_idle()
            fig2.canvas.draw_idle()
            return

        # shift mode
        if (
            step2_state[
                "mode"
            ] == "shift"
        ):
            if seg_name.endswith(
                "shank"
            ):
                print(
                    f"[STEP2] {seg_name} 已在 STEP1 与 pelvis 对齐，"
                    "此处固定；请移动 thigh/foot。"
                )
                return

            step2_state[
                "active_seg"
            ] = seg_name

            step2_state[
                "start_x"
            ] = evt.xdata

            step2_state[
                "start_internal_manual"
            ] = state.internal_manual_shift

            print(
                f"[STEP2] 开始移动 {seg_name}，shank固定"
            )

    def on_step2_motion(evt):
        if evt.xdata is None:
            return

        if crop_state["drag"] is not None:
            side = crop_state["drag"]

            if side == "left":
                crop_state["left"] = min(
                    evt.xdata,
                    crop_state["right"] - sample_dt(t)
                )
            else:
                crop_state["right"] = max(
                    evt.xdata,
                    crop_state["left"] + sample_dt(t)
                )

            update_crop_lines()
            return

        active_event = (
            step2_state[
                "active_event"
            ]
        )

        if active_event is not None:
            state = segment_states[
                active_event.owner_seg
            ]

            active_event.orig_time = (
                evt.xdata
                - state.final_shift(
                    side_base_shift
                )
            )
            active_event.edited = True

            active_event.redraw(
                segment_states,
                side_base_shift,
            )

            show_event_shadow(
                active_event
            )

            fig1.canvas.draw_idle()
            fig2.canvas.draw_idle()
            return

        seg_name = step2_state[
            "active_seg"
        ]

        if seg_name is None:
            return

        state = segment_states[
            seg_name
        ]

        delta = (
            evt.xdata
            - step2_state[
                "start_x"
            ]
        )

        state.internal_manual_shift = (
            step2_state[
                "start_internal_manual"
            ]
            + delta
        )

        redraw_segment(
            seg_name
        )

        selected_ev = selected_event_state[
            "event"
        ]

        if (
            selected_ev is not None
            and not selected_ev.removed
            and selected_ev.owner_seg == seg_name
        ):
            show_event_shadow(
                selected_ev
            )

        fig2.canvas.draw_idle()

    def on_step2_release(evt):
        if crop_state["drag"] is not None:
            print(
                f"[裁剪范围] {crop_state['left']:.3f}s ~ {crop_state['right']:.3f}s"
            )
            crop_state["drag"] = None
            return

        if (
            step2_state[
                "active_event"
            ]
            is not None
        ):
            step2_state[
                "active_event"
            ] = None

        seg_name = step2_state[
            "active_seg"
        ]

        if seg_name is not None:
            state = segment_states[
                seg_name
            ]

            print(
                f"[STEP2] {seg_name}: "
                f"side={side_base_shift[state.side]:+.3f}s, "
                f"auto={state.internal_auto_shift:+.3f}s, "
                f"manual={state.internal_manual_shift:+.3f}s, "
                f"final={state.final_shift(side_base_shift):+.3f}s"
            )

        step2_state[
            "active_seg"
        ] = None

    # STEP1 也可以 S 保存
    def on_step1_key(evt):
        if evt.key in (
            "s",
            "S",
        ):
            save_result(
                df=df,
                t=t,
                segment_states=segment_states,
                side_base_shift=side_base_shift,
                events=events,
                auto_qc=auto_qc,
                pelvis_qc={
                    "PelvisXProminence": px_prom,
                    "PelvisYProminence": py_prom,
                },
                crop_state=crop_state,
            )

    # --------------------------------------------------------
    # connect
    # --------------------------------------------------------

    fig1.canvas.mpl_connect(
        "button_press_event",
        on_step1_press,
    )

    fig1.canvas.mpl_connect(
        "motion_notify_event",
        on_step1_motion,
    )

    fig1.canvas.mpl_connect(
        "button_release_event",
        on_step1_release,
    )

    fig1.canvas.mpl_connect(
        "key_press_event",
        on_step1_key,
    )

    fig2.canvas.mpl_connect(
        "button_press_event",
        on_step2_press,
    )

    fig2.canvas.mpl_connect(
        "motion_notify_event",
        on_step2_motion,
    )

    fig2.canvas.mpl_connect(
        "button_release_event",
        on_step2_release,
    )

    fig2.canvas.mpl_connect(
        "key_press_event",
        on_step2_key,
    )

    return (
        fig1,
        fig2,
        segment_states,
        side_base_shift,
        events,
        auto_qc,
    )


# ============================================================
# 8. 保存
# ============================================================

def save_result(
    df,
    t,
    segment_states,
    side_base_shift,
    events,
    auto_qc,
    pelvis_qc,
    crop_state=None,
):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    dt = sample_dt(t)

    # --------------------------------------------------------
    # 1) 计算每个 segment 的最终整数样本位移
    # --------------------------------------------------------

    requested_shift_s = {}
    applied_shift_samples = {}
    applied_shift_s = {}

    for seg_name, state in (
        segment_states.items()
    ):
        requested = state.final_shift(
            side_base_shift
        )

        samples = (
            shift_seconds_to_samples(
                requested,
                dt,
            )
        )

        requested_shift_s[
            seg_name
        ] = requested

        applied_shift_samples[
            seg_name
        ] = samples

        applied_shift_s[
            seg_name
        ] = samples * dt

    # pelvis 固定
    applied_shift_samples[
        "pelvis"
    ] = 0

    applied_shift_s[
        "pelvis"
    ] = 0.0

    # --------------------------------------------------------
    # 2) 真正移动数据列
    # --------------------------------------------------------

    out = df.copy()

    move_rows = []

    for seg_name in (
        "l_thigh",
        "l_shank",
        "l_foot",
        "r_thigh",
        "r_shank",
        "r_foot",
    ):
        n_shift = (
            applied_shift_samples[
                seg_name
            ]
        )

        cols = shift_sensor_columns(
            out,
            seg_name,
            n_shift,
        )

        move_rows.append({
            "Segment": seg_name,
            "RequestedShift_s": requested_shift_s[
                seg_name
            ],
            "AppliedShiftSamples": n_shift,
            "AppliedShift_s": applied_shift_s[
                seg_name
            ],
            "NumColumnsMoved": len(cols),
        })

    # pelvis不动，只验证列存在
    pelvis_cols = shift_sensor_columns(
        out,
        "pelvis",
        0,
    )

    move_rows.append({
        "Segment": "pelvis",
        "RequestedShift_s": 0.0,
        "AppliedShiftSamples": 0,
        "AppliedShift_s": 0.0,
        "NumColumnsMoved": len(pelvis_cols),
    })

    # --------------------------------------------------------
    # 3) 删除移动后首尾空缺
    # --------------------------------------------------------

    all_shifts = [
        applied_shift_samples[
            seg
        ]
        for seg in (
            "l_thigh",
            "l_shank",
            "l_foot",
            "r_thigh",
            "r_shank",
            "r_foot",
        )
    ]

    # pelvis=0，也纳入共同参考
    all_shifts.append(0)

    trim_start, trim_end = (
        common_trim_range(
            len(out),
            all_shifts,
        )
    )

    # 人工灰线裁剪：在自动去空缺基础上进一步限制范围
    if crop_state is not None:
        manual_start = int(
            np.searchsorted(
                t,
                crop_state["left"]
            )
        )
        manual_end = int(
            np.searchsorted(
                t,
                crop_state["right"]
            )
        )

        trim_start = max(
            trim_start,
            manual_start,
        )
        trim_end = min(
            trim_end,
            manual_end,
        )

    rows_before = len(out)

    out = (
        out
        .iloc[
            trim_start:trim_end
        ]
        .copy()
        .reset_index(drop=True)
    )

    t_kept = np.asarray(
        t[
            trim_start:trim_end
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # 4) 只生成左右两个总标签
    # --------------------------------------------------------

    l_label, l_unified, l_qc, l_used_events = (
        build_total_label(
            t=t_kept,
            side="left",
            prefix="l_",
            events=events,
            applied_shift_s=applied_shift_s,
        )
    )

    r_label, r_unified, r_qc, r_used_events = (
        build_total_label(
            t=t_kept,
            side="right",
            prefix="r_",
            events=events,
            applied_shift_s=applied_shift_s,
        )
    )

    used_for_total_label = (
        set(l_used_events)
        | set(r_used_events)
    )

    # 如果原表已经有同名标签，先删掉，确保最后两列就是它们
    for col in (
        "l_label",
        "r_label",
    ):
        if col in out.columns:
            out = out.drop(
                columns=[col]
            )

    # 最后两列
    out["l_label"] = l_label
    out["r_label"] = r_label

    # --------------------------------------------------------
    # 标签来源安全检查：
    # l_label / r_label 绝不能被 thigh 或 foot Event3 等影响
    # --------------------------------------------------------
    for ev in used_for_total_label:
        legal = (
            (
                ev.owner_seg.endswith("foot")
                and ev.event_id == 1
            )
            or
            (
                ev.owner_seg.endswith("shank")
                and ev.event_id in (2, 3)
            )
        )

        if not legal:
            raise RuntimeError(
                "发现非法标签事件来源："
                f"{ev.owner_seg}, Event{ev.event_id}。"
                "最终标签只允许 foot Event1 "
                "和 shank Event2/Event3。"
            )

    # --------------------------------------------------------
    # 5) 主 Excel：
    #    原表列（数值已对齐） + l_label + r_label
    # --------------------------------------------------------

    out.to_excel(
        OUTPUT_LABEL_FILE,
        sheet_name="data",
        index=False,
    )

    # --------------------------------------------------------
    # 6) 不再单独输出事件 metadata 文件
    # --------------------------------------------------------

    shift_rows = []

    for seg_name, state in (
        segment_states.items()
    ):
        shift_rows.append({
            "Side": state.side,
            "Segment": seg_name,
            "SideBaseShift_s": side_base_shift[
                state.side
            ],
            "InternalAutoShift_s": state.internal_auto_shift,
            "InternalManualShift_s": state.internal_manual_shift,
            "RequestedFinalShift_s": requested_shift_s[
                seg_name
            ],
            "AppliedShiftSamples": applied_shift_samples[
                seg_name
            ],
            "AppliedShift_s": applied_shift_s[
                seg_name
            ],
        })

    # --------------------------------------------------------
    # 所有当前仍存在的事件线必须完整保存。
    # AllEvents 是“交互结果的完整审计表”，不再等同于“用于标签的事件”。
    # --------------------------------------------------------
    event_rows = []

    visible_events = [
        ev
        for ev in events
        if not ev.removed
    ]

    # 为了 Excel 更容易检查，按 side / aligned time 排序
    visible_events = sorted(
        visible_events,
        key=lambda ev: (
            ev.side,
            ev.orig_time
            + applied_shift_s[ev.owner_seg],
            ev.owner_seg,
            ev.event_id,
        ),
    )

    for event_index, ev in enumerate(
        visible_events,
        start=1,
    ):
        aligned_time = (
            ev.orig_time
            + applied_shift_s[
                ev.owner_seg
            ]
        )

        # 裁剪后是否仍落在最终主数据时间范围内
        if len(t_kept):
            inside_kept_range = bool(
                t_kept[0]
                <= aligned_time
                <= t_kept[-1]
            )
        else:
            inside_kept_range = False

        event_rows.append({
            "EventIndex": event_index,
            "Side": ev.side,
            "Segment": ev.owner_seg,
            "Event": ev.event_id,

            # auto / manual
            "Source": ev.source,

            # 是否被人工拖过
            "Edited": bool(ev.edited),

            # 当前版本所有事件线都随所属曲线一起移动
            "VisualOnlyFixed": False,

            # 用户在该传感器原时间轴上的事件位置
            "OrigTime_s": ev.orig_time,

            # 最终真正应用到数据列的整数样本位移
            "AppliedShift_s": applied_shift_s[
                ev.owner_seg
            ],

            # 与最终对齐数据对应的事件时间
            "AlignedTime_s": aligned_time,

            # 是否仍落在裁剪后的最终数据区间
            "InsideKeptRange": inside_kept_range,

            # 是否真正被最终 l_label/r_label 选用
            "UsedForTotalLabel": (
                ev in used_for_total_label
            ),

            # 明确说明该事件在最终左右总标签中的角色
            "LabelRole": (
                "Event1_from_foot"
                if (
                    ev.owner_seg.endswith("foot")
                    and ev.event_id == 1
                )
                else (
                    "Event2_from_shank"
                    if (
                        ev.owner_seg.endswith("shank")
                        and ev.event_id == 2
                    )
                    else (
                        "Event3_from_shank"
                        if (
                            ev.owner_seg.endswith("shank")
                            and ev.event_id == 3
                        )
                        else "IGNORED"
                    )
                )
            ),
        })

    # 保存前强制校验：图上存在多少事件，AllEvents 就必须有多少行。
    if len(event_rows) != len(visible_events):
        raise RuntimeError(
            "事件保存数量校验失败："
            f"visible={len(visible_events)}, "
            f"saved={len(event_rows)}"
        )

    trim_df = pd.DataFrame([
        {
            "SamplePeriod_s": dt,
            "RowsBefore": rows_before,
            "TrimStartRows": trim_start,
            "TrimEndRows": (
                rows_before
                - trim_end
            ),
            "RowsAfter": len(out),
            "FirstKeptTime_s": (
                float(t_kept[0])
                if len(t_kept)
                else np.nan
            ),
            "LastKeptTime_s": (
                float(t_kept[-1])
                if len(t_kept)
                else np.nan
            ),
        }
    ])

    # --------------------------------------------------------
    # 当前版本不再保存事件/对齐元数据 Excel。
    # 事件线、人工调整和 shift 仍然参与本次对齐与标签生成，
    # 但不会另存为 alignment_events.xlsx。
    # --------------------------------------------------------

    print("=" * 80)
    print(
        f"[保存完成] {OUTPUT_LABEL_FILE}"
    )
    print(
        "[说明] 只保存主 Excel，不再生成 alignment_events.xlsx"
    )
    print("")
    print(
        f"原始 {rows_before} 行 -> "
        f"最终 {len(out)} 行"
    )
    print(
        f"删除开头 {trim_start} 行；"
        f"删除结尾 {rows_before - trim_end} 行"
    )
    print("")
    print(
        f"[事件审计] 当前图上有效事件线 = {len(visible_events)} 条；"
        f"AllEvents 已保存 = {len(event_rows)} 条"
    )
    print(
        f"[人工事件] Source=manual 共 "
        f"{sum(1 for ev in visible_events if ev.source == 'manual')} 条"
    )
    print(
        f"[人工修改] Edited=True 共 "
        f"{sum(1 for ev in visible_events if ev.edited)} 条"
    )
    print("")
    print("主 Excel 最后只新增两列：")
    print("  l_label")
    print("  r_label")
    print("")
    print("标签规则：")
    print("  开始 -> first E3 = 0")
    print("  E3 -> E1 = 3")
    print("  E1 -> E2 = 1")
    print("  E2 -> next E3 = 2")
    print("  last E1 -> 结束 = 0")
    print("=" * 80)


# ============================================================
# 9. main
# ============================================================

def main():
    df = pd.read_excel(
        INPUT_FILE,
        sheet_name="data",
    )

    t = df[
        "time_s"
    ].to_numpy(dtype=float)

    build_ui(
        t=t,
        df=df,
    )

    print("=" * 80)
    print("使用顺序：")
    print("")
    print("STEP1 窗口：")
    print("  Pelvis X/Y 固定")
    print("  拖 L shank -> 左侧整体时间移动")
    print("  拖 R shank -> 右侧整体时间移动")
    print("")
    print("STEP2 窗口：")
    print("  shank 固定")
    print("  拖 thigh / foot -> 单腿内部微调")
    print("  事件只用于总标签：foot只加E1；shank只加E2/E3；thigh原Event2保留但固定，仅作参考")
    print("  拖虚线 -> 微调事件")
    print("  右键虚线 -> 删除事件")
    print("")
    print("任一窗口按 S -> 保存")
    print("")
    print("最终主Excel只新增 l_label / r_label")
    print("=" * 80)

    plt.show()


if __name__ == "__main__":
    main()
