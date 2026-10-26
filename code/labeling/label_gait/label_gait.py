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
# 0. Configuration
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

# The current version no longer saves alignment_events.xlsx separately
# Save only the final aligned Excel workbook with left and right aggregate labels.

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
# 1. Shared segment states
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
# 2. Shared event lines
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

        # auto = events detected automatically at initialization
        # manual = events added manually during interaction
        self.source = str(source)

        # visual_only_fixed=True：
        # Visual reference only; excluded from labeling and alignment.
        # This reference line remains fixed when its curve is shifted.
        self.visual_only_fixed = bool(visual_only_fixed)

        # Mark True if this event line has ever been dragged manually
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
        # All event lines move together with the shift of their associated IMU curves.
        # build_total_label determines which event lines contribute to the final labels.
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
# 3. Peaks and extrema
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

    # thigh minimum -> Event2 (visual reference only)
    thigh_min = find_event_peaks(
        thigh,
        MIN_PEAK_DIST,
        MIN_PEAK_PROM,
        "min",
    )

    # shank maximum -> Event2 (used for final labels)
    shank_max = find_event_peaks(
        shank,
        MIN_PEAK_DIST,
        MIN_PEAK_PROM,
        "max",
    )

    # shank minimum -> Event3 (used for final labels)
    shank_min = find_event_peaks(
        shank,
        MIN_PEAK_DIST,
        MIN_PEAK_PROM,
        "min",
    )

    # foot maximum -> Event3 (visual reference only)
    foot_max = find_event_peaks(
        foot,
        MIN_PEAK_DIST,
        MIN_PEAK_PROM,
        "max",
    )

    # foot minimum -> Event1 (used for final labels)
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
# 4. Initial automatic alignment within one side
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
# 5. Auxiliary pelvis peaks and valleys
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
# 6. Shift and labeling utilities for saving
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
    # 1. Retain only events required for the final aggregate labels
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
    # 2. Require at least E3 and E1
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

    # First valid boundary: the first shank E3
    first_e3_time, first_e3_ev = (
        e3_items[0]
    )

    # The final Event1 must occur after the first E3
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
    # 3. Find the final shank Event3 preceding the final Event1
    #
    #    This segment must satisfy the strict condition:
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
    # 4. Construct continuous-state events
    #
    #    Between the first E3 and terminal E3:
    #        E3 -> state 3
    #        E1 -> state 1
    #        E2 -> state 2
    #
    #    Do not leave an entire cycle labeled 0 just because an event is missing.
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

    # If different events occur at the same time,
    # sort them in normal phase order: E3 -> E1 -> E2.
    # A later state overrides the labels after that instant.
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

    # In principle, the first event should be the first E3.
    # Include it explicitly as a precaution.
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
    # 5. Assign continuous labels starting at the first E3
    #
    #    Each event line only updates the current state to:
    #        E3 => 3
    #        E1 => 1
    #        E2 => 2
    #
    #    This state persists until the next valid event.
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

        # If several event lines share a timestamp,
        # skip the preceding zero-length intervals automatically.
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
        # QC reports warnings only; it no longer resets labels to 0
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
    # 6. Final incomplete cycle:
    #
    #    Final shank E3 to final foot E1 = 3
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
    # 7. Force labels at both ends to 0
    # --------------------------------------------------------

    labels[
        t < first_e3_time
    ] = 0

    labels[
        t >= last_e1_time
    ] = 0

    # --------------------------------------------------------
    # 8. Critical consistency check:
    #    No 0 label is permitted between the first E3 and final E1
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
# 7. Build the UI
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
    # Data and shared states
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

    # STEP1: overall left/right shank-to-pelvis baseline shifts
    side_base_shift = {
        "left": 0.0,
        "right": 0.0,
    }

    # Automatic initial shifts within each leg
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
    # STEP1: pelvis fixed, left/right shanks movable
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

        # In STEP1, show time axes and ticks on every subplot, including the pelvis
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
    # STEP1 selection shading: L shank / R shank / pelvis X / pelvis Y
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
        # Support Rectangle / Polygon objects returned by axvspan across matplotlib versions
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
    # STEP2: alignment within each leg
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

    # In STEP2, show x-axis labels only on the bottom row
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
    # STEP2 event selection shading:
    # Left-side events -> show on all three left thigh/shank/foot plots
    # Right-side events -> show on all three right thigh/shank/foot plots
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

    # Track the most recently clicked or selected event.
    # Update the shading with the event line when its curve moves.
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

        # Shank events also appear in STEP1, so update shading on all four pelvis-window plots
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

        # All original event lines move with their respective curves.
        # Thigh E2 and foot E3 remain visual references and do not affect final aggregate labels.
        ev = SharedEvent(
            owner_seg=seg_name,
            side=side,
            event_id=event_id,
            orig_time=event_time,
            source="auto",
            visual_only_fixed=False,
        )

        # Always show STEP2
        ev.add_artist(
            seg_axis[seg_name]
        )

        # Also display the shank synchronously in STEP1
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
    # STEP1 mouse interaction: shank moves, pelvis stays fixed
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

        # Pelvis stays fixed
        if role == "pelvis":
            if evt.button == 1:
                print(
                    "[STEP1] The pelvis is a fixed reference and cannot be moved."
                )
            return

        # Delete a shank event
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

        # Prioritize dragging event lines
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

        # Otherwise drag the whole-side shank baseline
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

        # Synchronize the baseline shift for the entire side:
        # Shank, thigh, and foot all follow
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
    # STEP2 mouse interaction: shank fixed, thigh/foot movable
    # ========================================================

    step2_state = {
        "mode": "shift",
        "active_seg": None,
        "start_x": None,
        "start_internal_manual": None,
        "active_event": None,
    }

    # ========================================================
    # STEP2 final crop bounds (two gray lines)
    # Display and move both gray boundary lines across all subplots
    # Remove data outside the gray boundaries on save
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

    # Only events required for final aggregate labels can be added manually
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

        # First check existing events:
        # All event lines are clickable; selection shades three plots on the same side.
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

        # Add an event
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

            # Synchronize STEP1 when adding a shank event
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

    # S also saves in STEP1
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
# 8. Save
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
    # 1) Compute the final integer-sample shift for each segment
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

    # Pelvis stays fixed
    applied_shift_samples[
        "pelvis"
    ] = 0

    applied_shift_s[
        "pelvis"
    ] = 0.0

    # --------------------------------------------------------
    # 2) Apply shifts to data columns
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

    # Do not shift the pelvis; only verify required columns exist
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
    # 3) Remove leading and trailing gaps after shifting
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

    # Include the pelvis as a common reference with zero shift
    all_shifts.append(0)

    trim_start, trim_end = (
        common_trim_range(
            len(out),
            all_shifts,
        )
    )

    # Manual gray-line crop further restricts the range after automatic gap removal
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
    # 4) Generate only left and right aggregate labels
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

    # Remove existing same-named labels to ensure these labels occupy the final two columns
    for col in (
        "l_label",
        "r_label",
    ):
        if col in out.columns:
            out = out.drop(
                columns=[col]
            )

    # Final two columns
    out["l_label"] = l_label
    out["r_label"] = r_label

    # --------------------------------------------------------
    # Validate label event sources:
    # l_label / r_label must not be affected by thigh events or foot Event3
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
    # 5) Primary Excel workbook:
    #    Aligned original columns plus l_label and r_label
    # --------------------------------------------------------

    out.to_excel(
        OUTPUT_LABEL_FILE,
        sheet_name="data",
        index=False,
    )

    # --------------------------------------------------------
    # 6) Do not export a separate event metadata file
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
    # Preserve records for every event line that still exists.
    # AllEvents is a complete interaction audit, not merely events used for labeling.
    # --------------------------------------------------------
    event_rows = []

    visible_events = [
        ev
        for ev in events
        if not ev.removed
    ]

    # Sort by side and aligned time for easier Excel inspection
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

        # Whether the event remains in the final data time range after cropping
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

            # Whether the event was dragged manually
            "Edited": bool(ev.edited),

            # All event lines move with their associated curves in the current version
            "VisualOnlyFixed": False,

            # Event location on the sensor's original time axis
            "OrigTime_s": ev.orig_time,

            # Final integer-sample shift actually applied to the data columns
            "AppliedShift_s": applied_shift_s[
                ev.owner_seg
            ],

            # Event time corresponding to the final aligned data
            "AlignedTime_s": aligned_time,

            # Whether the event remains within the final cropped data interval
            "InsideKeptRange": inside_kept_range,

            # Whether the event is used by the final l_label / r_label
            "UsedForTotalLabel": (
                ev in used_for_total_label
            ),

            # Explicitly describe the event's role in the final left/right aggregate labels
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

    # Before saving, require AllEvents rows to match the number of displayed events.
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
    # The current version no longer saves a separate event/alignment metadata workbook.
    # Event lines, manual adjustments, and shifts still affect alignment and label generation,
    # but are not saved separately to alignment_events.xlsx.
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
