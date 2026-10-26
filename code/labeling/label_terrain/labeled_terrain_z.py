import os
import re
import zipfile
import numpy as np
import pandas as pd


# ============================================================
# 0. Configuration
# ============================================================

SUB_ID = "sub_002"

# Select "z2", "z3", or "z4"
TERRAIN_ID = "z2"

# Data root directory
BASE_DIR = r"E:\data"

# Current participant directory
SUB_DIR = os.path.join(BASE_DIR, f"{SUB_ID}_label")

# Validate the terrain name
SUPPORTED_TERRAINS = ["z2", "z3", "z4"]

TERRAIN_ID = TERRAIN_ID.lower().strip()

if TERRAIN_ID not in SUPPORTED_TERRAINS:
    raise ValueError(
        f"不支持的地形名称: {TERRAIN_ID}\n"
        f"目前支持: {', '.join(SUPPORTED_TERRAINS)}"
    )

# Generate fwd/bwd input paths for the selected terrain
INPUT_FILES = [
    os.path.join(
        SUB_DIR,
        f"{SUB_ID}_{TERRAIN_ID}_{direction}_aligned_labeled.xlsx"
    )
    for direction in ["fwd", "bwd"]
]

PRINT_EVENTS = True

# Force labels to 0 starting from the final l_label == 3 event
FORCE_ZERO_FROM_LAST_LEFT_LABEL3 = True



# ============================================================
# 1. Terrain labels for Z2
# ============================================================

Z2_FWD_SINGLE_TERRAIN_SEQUENCE = [
    15,
    30,
    34,
    34,
    26,
    30,
    38,
    30,
    22,
    26,
    15,

    # Set subsequent labels to 0 at the final l_label == 3 event
    0,
]

Z2_FWD_SAME_TERRAIN_SEQUENCE = [
    15,
    15,
    19,
    15,
    11,
    19,
    19,
    11,
    11,
    15,
    0,
    0,
]

Z2_BWD_SINGLE_TERRAIN_SEQUENCE = [
    -15,
    -26,
    -22,
    -30,
    -38,
    -30,
    -26,
    -34,
    -34,
    -30,
    -15,

    # Set subsequent labels to 0 at the final l_label == 3 event
    0,
]

Z2_BWD_SAME_TERRAIN_SEQUENCE = [
    -15,
    -11,
    -11,
    -19,
    -19,
    -11,
    -15,
    -19,
    -15,
    -15,
    0,
    0,
]


# ============================================================
# 2. Terrain labels for Z3
# ============================================================

Z3_FWD_SINGLE_TERRAIN_SEQUENCE = [
    15,
    30,
    38,
    38,
    22,
    30,
    46,
    30,
    14,
    22,
    15,

    # Set subsequent labels to 0 at the final l_label == 3 event
    0,
]

Z3_FWD_SAME_TERRAIN_SEQUENCE = [
    15,
    15,
    23,
    15,
    7,
    23,
    23,
    7,
    7,
    15,
    0,
    0,
]

Z3_BWD_SINGLE_TERRAIN_SEQUENCE = [
    -15,
    -22,
    -14,
    -30,
    -46,
    -30,
    -22,
    -38,
    -38,
    -30,
    -15,

    # Set subsequent labels to 0 at the final l_label == 3 event
    0,
]

Z3_BWD_SAME_TERRAIN_SEQUENCE = [
    -15,
    -7,
    -7,
    -23,
    -23,
    -7,
    -15,
    -23,
    -15,
    -15,
    0,
    0,
]


# ============================================================
# 3. Terrain labels for Z4
# ============================================================

Z4_FWD_SINGLE_TERRAIN_SEQUENCE = [
    19,
    38,
    26,
    26,
    46,
    34,
    14,
    34,
    54,
    46,
    19,

    # Set subsequent labels to 0 at the final l_label == 3 event
    0,
]

Z4_FWD_SAME_TERRAIN_SEQUENCE = [
    19,
    19,
    7,
    19,
    27,
    7,
    7,
    27,
    27,
    19,
    0,
    0,
]

Z4_BWD_SINGLE_TERRAIN_SEQUENCE = [
    -19,
    -46,
    -54,
    -34,
    -14,
    -34,
    -46,
    -26,
    -26,
    -38,
    -19,

    # Set subsequent labels to 0 at the final l_label == 3 event
    0,
]

Z4_BWD_SAME_TERRAIN_SEQUENCE = [
    -19,
    -27,
    -27,
    -7,
    -7,
    -27,
    -19,
    -7,
    -19,
    -19,
    0,
    0,
]


# ============================================================
# 4. Automatic Excel loading
# ============================================================

def read_excel_auto_sheet(input_file):
    """
    Automatically load the Excel workbook.

    Prefer the data/Data/DATA worksheet.
    Otherwise read the first worksheet.

    如果文件不是有效 xlsx，会给出明确提示。
    """
    try:
        xls = pd.ExcelFile(input_file)

    except zipfile.BadZipFile:
        raise RuntimeError(
            "当前文件不是有效的 .xlsx 文件，可能文件损坏，"
            "或者只是被改名成 .xlsx。\n"
            f"File: {input_file}\n"
            "Open the file in Excel/WPS and save it as a valid Excel workbook (*.xlsx)."
        )

    except Exception as e:
        raise RuntimeError(
            "Failed to read the Excel workbook.\n"
            f"File: {input_file}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {e}"
        )

    print("=" * 80)
    print("[Excel worksheet inspection]")
    print(f"File: {input_file}")
    print(f"工作表: {xls.sheet_names}")

    preferred_sheets = ["data", "Data", "DATA"]

    sheet_name = None

    for name in preferred_sheets:
        if name in xls.sheet_names:
            sheet_name = name
            break

    if sheet_name is None:
        sheet_name = xls.sheet_names[0]
        print(f"[提示] 未找到 data/Data/DATA，自动读取第一个工作表: {sheet_name}")
    else:
        print(f"[提示] 使用工作表: {sheet_name}")

    df = pd.read_excel(input_file, sheet_name=sheet_name)

    print(f"[Read OK] 数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")

    if "time_s" in df.columns:
        print(f"[Time range] {df['time_s'].iloc[0]} s -> {df['time_s'].iloc[-1]} s")

    print("=" * 80)

    return df, sheet_name


# ============================================================
# 5. Determine z2 / z3 / z4 direction
# ============================================================

def detect_z234_direction_from_filename(input_file):
    """
    从文件名判断：
        z2_fwd / z2_bwd
        z3_fwd / z3_bwd
        z4_fwd / z4_bwd
    """
    basename = os.path.basename(input_file).lower()

    patterns = [
        "z2_fwd",
        "z2_bwd",
        "z3_fwd",
        "z3_bwd",
        "z4_fwd",
        "z4_bwd",
    ]

    for p in patterns:
        if p in basename:
            return p

    raise ValueError(
        "无法从文件名判断方向，请确保文件名包含：\n"
        "z2_fwd / z2_bwd / z3_fwd / z3_bwd / z4_fwd / z4_bwd\n"
        f"当前File: {input_file}"
    )


# ============================================================
# 6. Generate the output path
# ============================================================

def extract_subject_id(input_file):
    """
    Extract sub_xxx from the path or filename.
    """
    text = input_file.replace("\\", "/")

    match = re.search(r"(sub_\d+)", text, flags=re.IGNORECASE)

    if match:
        return match.group(1)

    basename = os.path.basename(input_file)
    stem = os.path.splitext(basename)[0]

    return stem.split("_")[0]


def build_output_file(input_file):
    """
    生成输出文件夹和输出文件路径。

    Output directory:
        sub_xxx_labeled_terrain

    Output file:
        Original filename + _terrain.xlsx
    """
    subject_id = extract_subject_id(input_file)

    input_dir = os.path.dirname(input_file)
    parent_dir = os.path.dirname(input_dir)

    if not parent_dir:
        parent_dir = input_dir if input_dir else "."

    output_dir = os.path.join(
        parent_dir,
        f"{subject_id}_labeled_terrain",
    )

    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.basename(input_file)
    stem, ext = os.path.splitext(basename)

    output_file = os.path.join(
        output_dir,
        f"{stem}_terrain{ext}",
    )

    return output_file


# ============================================================
# 7. Save the Excel workbook safely
# ============================================================

def save_excel_safely(df, output_file, sheet_name="data"):
    """
    Save the Excel workbook safely.

    If the target cannot be overwritten because it is open in Excel/WPS,
    automatically save as _v2, _v3, etc.
    """
    out_dir = os.path.dirname(output_file)

    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        df.to_excel(
            output_file,
            index=False,
            sheet_name=sheet_name if sheet_name else "data",
        )

        print(f"[Saved] {output_file}")
        return output_file

    except PermissionError:
        print("=" * 80)
        print("[警告] 输出文件无法写入，可能正在被 Excel/WPS 打开：")
        print(output_file)
        print("[处理] 自动尝试另存为新文件名。")

        folder = os.path.dirname(output_file)
        base = os.path.splitext(os.path.basename(output_file))[0]
        ext = os.path.splitext(output_file)[1]

        for i in range(2, 100):
            new_file = os.path.join(folder, f"{base}_v{i}{ext}")

            try:
                df.to_excel(
                    new_file,
                    index=False,
                    sheet_name=sheet_name if sheet_name else "data",
                )

                print(f"[Saved] {new_file}")
                return new_file

            except PermissionError:
                continue

        raise PermissionError(
            "Could not save using alternative names; close Excel/WPS and retry."
        )


# ============================================================
# 8. General event-detection functions
# ============================================================

def detect_label_value_rising_edges(df, label_col, label_value, side_name):
    """
    Detect the first sample of a label_value run in label_col.

    rising edge 条件：
        当前行 label == label_value
        且上一行 label != label_value
    """
    if label_col not in df.columns:
        raise KeyError(f"缺少列: {label_col}")

    label = df[label_col].to_numpy()

    is_target = label == label_value

    prev_is_target = np.r_[
        False,
        is_target[:-1],
    ]

    rising = is_target & (~prev_is_target)

    idxs = np.where(rising)[0]

    events = []

    for idx in idxs:
        if "time_s" in df.columns:
            time_s = float(df.loc[idx, "time_s"])
        else:
            time_s = float(idx)

        events.append(
            {
                "idx": int(idx),
                "time_s": time_s,
                "side": side_name,
                "label_col": label_col,
                "label_value": label_value,
            }
        )

    return events


# ============================================================
# 9. Shared event logic for Z2/Z3/Z4
# ============================================================

def collect_z234_terrain_events(df):
    """
    Z2/Z3/Z4 的贴标事件Logic:

    Initialize to 0.

    Event 1:
        First l_label == 3
        从这里开始贴第 1 个标签，
        一直到Second l_label == 3 之前。

    Event 2:
        Second l_label == 3

    Event 3:
        Second l_label == 3 之后，
        第一个 r_label == 3

    Event 4 and later:
        After the third event,
        l_label == 3 或 r_label == 3

    Finally:
        最后一个 l_label == 3 作为归 0 起点。
    """
    if "l_label" not in df.columns:
        raise KeyError("缺少列: l_label")

    if "r_label" not in df.columns:
        raise KeyError("缺少列: r_label")

    events = []

    l3_events = detect_label_value_rising_edges(
        df=df,
        label_col="l_label",
        label_value=3,
        side_name="L",
    )

    r3_events = detect_label_value_rising_edges(
        df=df,
        label_col="r_label",
        label_value=3,
        side_name="R",
    )

    if len(l3_events) < 2:
        raise RuntimeError(
            "l_label == 3 的起始事件少于 2 个，无法执行 Z2/Z3/Z4 的贴标规则。"
        )

    if len(r3_events) < 1:
        raise RuntimeError(
            "r_label == 3 的起始事件少于 1 个，无法执行 Z2/Z3/Z4 的贴标规则。"
        )

    # Event 1: the first l_label == 3
    first_event = dict(l3_events[0])
    first_event["event_rule"] = "first_l_label_3_start_first_label"
    events.append(first_event)

    # Event 2: the second l_label == 3
    second_event = dict(l3_events[1])
    second_event["event_rule"] = "second_l_label_3"
    events.append(second_event)

    second_idx = second_event["idx"]

    # Event 3: first r_label == 3 after the second l_label == 3
    r3_after_second = [
        e for e in r3_events
        if e["idx"] > second_idx
    ]

    if len(r3_after_second) == 0:
        raise RuntimeError(
            "已经找到Second l_label == 3，"
            "但其后没有找到 r_label == 3，无法执行第 3 次贴标。"
        )

    third_event = dict(r3_after_second[0])
    third_event["event_rule"] = "third_r_label_3_after_second_l3"
    events.append(third_event)

    third_idx = third_event["idx"]

    # Event 4 onward: l_label == 3 or r_label == 3 after the third event
    left_l3_after_third = [
        e for e in l3_events
        if e["idx"] > third_idx
    ]

    right_r3_after_third = [
        e for e in r3_events
        if e["idx"] > third_idx
    ]

    later_events = left_l3_after_third + right_r3_after_third

    later_events = sorted(
        later_events,
        key=lambda e: e["idx"],
    )

    for e in later_events:
        e["event_rule"] = "later_l_or_r_label_3"

    events.extend(later_events)

    return events


# ============================================================
# 10. Get label sequences for the direction
# ============================================================

def get_z234_sequences(direction):
    """
    根据方向返回单侧和同侧地形标签序列。
    """
    if direction == "z2_fwd":
        return Z2_FWD_SINGLE_TERRAIN_SEQUENCE, Z2_FWD_SAME_TERRAIN_SEQUENCE

    if direction == "z2_bwd":
        return Z2_BWD_SINGLE_TERRAIN_SEQUENCE, Z2_BWD_SAME_TERRAIN_SEQUENCE

    if direction == "z3_fwd":
        return Z3_FWD_SINGLE_TERRAIN_SEQUENCE, Z3_FWD_SAME_TERRAIN_SEQUENCE

    if direction == "z3_bwd":
        return Z3_BWD_SINGLE_TERRAIN_SEQUENCE, Z3_BWD_SAME_TERRAIN_SEQUENCE

    if direction == "z4_fwd":
        return Z4_FWD_SINGLE_TERRAIN_SEQUENCE, Z4_FWD_SAME_TERRAIN_SEQUENCE

    if direction == "z4_bwd":
        return Z4_BWD_SINGLE_TERRAIN_SEQUENCE, Z4_BWD_SAME_TERRAIN_SEQUENCE

    raise ValueError(f"未知Direction: {direction}")


# ============================================================
# 11. Strictly validate that label count matches event count
# ============================================================

def check_event_label_count_match(events, sequence, sequence_name):
    """
    严格检查事件数量和标签数量是否一致。

    如果不一致，返回 False。
    """
    event_count = len(events)
    label_count = len(sequence)

    print("=" * 80)
    print(f"[数量检查] {sequence_name}")
    print(f"Event count: {event_count}")
    print(f"Label count: {label_count}")

    if event_count == label_count:
        print("[检查通过] 事件数量和标签数量一致。")
        print("=" * 80)
        return True

    print("[检查失败] 事件数量和标签数量不一致。")
    print("将跳过当前文件，不生成新文件。")

    if event_count > label_count:
        print(f"事件数量比标签数量多 {event_count - label_count} 个。")
    else:
        print(f"标签数量比事件数量多 {label_count - event_count} 个。")

    print("=" * 80)

    return False


# ============================================================
# 12. Assign terrain labels from the event sequence
# ============================================================

def apply_terrain_sequence(
    df,
    events,
    terrain_sequence,
    output_col,
    initial_label=0,
    keep_last=False,
):
    """
    Assign terrain labels from the event sequence.

    Labeling logic:
        Before the first event:
            initial_label

        From event k to just before event k+1:
            terrain_sequence[k]

        After the final event:
            terrain_sequence[-1]
    """
    n = len(df)

    terrain = np.full(
        n,
        initial_label,
        dtype=float,
    )

    if len(events) == 0:
        print(f"[警告] 没有检测到事件，{output_col} 全部填 {initial_label}")
        df[output_col] = terrain.astype(int)
        return df

    if len(events) != len(terrain_sequence):
        msg = (
            f"{output_col}: 事件数量 {len(events)} "
            f"和地形序列长度 {len(terrain_sequence)} 不一致。"
        )

        if not keep_last:
            raise ValueError(msg)
        else:
            print(f"[警告] {msg}")
            print("[处理] 超出部分将保持 terrain_sequence 最后一个标签。")

    first_idx = events[0]["idx"]

    if first_idx > 0:
        terrain[:first_idx] = initial_label

    for k, event in enumerate(events):
        start_idx = event["idx"]

        if k + 1 < len(events):
            end_idx = events[k + 1]["idx"]
        else:
            end_idx = n

        if k < len(terrain_sequence):
            value = terrain_sequence[k]
        else:
            value = terrain_sequence[-1]

        terrain[start_idx:end_idx] = value

    if np.all(np.isfinite(terrain)):
        terrain = terrain.astype(int)

    df[output_col] = terrain

    return df


# ============================================================
# 13. Force labels to 0 after the final l_label == 3 event
# ============================================================

def force_zero_from_last_left_label3(df, output_col):
    """
    从最后一个 l_label == 3 的起始事件开始，
    Force the specified terrain-label column to 0.

    Logic:
        最后一个 l_label == 3 之前，仍然正常贴前一个标签；
        从最后一个 l_label == 3 开始，terrain = 0。
    """
    if "l_label" not in df.columns:
        raise KeyError("缺少列: l_label")

    if output_col not in df.columns:
        raise KeyError(f"缺少列: {output_col}")

    l3_events = detect_label_value_rising_edges(
        df=df,
        label_col="l_label",
        label_value=3,
        side_name="L",
    )

    if len(l3_events) == 0:
        print(f"[警告] 未找到 l_label == 3 起始事件，无法对 {output_col} 执行最后归 0")
        return df

    last_left_idx = int(l3_events[-1]["idx"])

    if "time_s" in df.columns:
        t = df.loc[last_left_idx, "time_s"]
        print(
            f"[Final reset to zero] {output_col}: "
            f"从最后一个 l_label == 3 开始置 0，"
            f"idx={last_left_idx}, time={t:.4f}s"
        )
    else:
        print(
            f"[Final reset to zero] {output_col}: "
            f"从最后一个 l_label == 3 开始置 0，"
            f"idx={last_left_idx}"
        )

    df.loc[last_left_idx:, output_col] = 0

    return df


# ============================================================
# 14. Print event-to-label mappings
# ============================================================

def print_event_mapping(events, sequence, title):
    print("=" * 130)
    print(title)
    print("-" * 130)
    print(
        f"{'No.':>4} | "
        f"{'idx':>8} | "
        f"{'time_s':>10} | "
        f"{'side':>4} | "
        f"{'source':>8} | "
        f"{'value':>6} | "
        f"{'rule':>34} | "
        f"{'terrain':>10}"
    )
    print("-" * 130)

    for k, event in enumerate(events):
        if k < len(sequence):
            value = sequence[k]
        else:
            value = sequence[-1]

        print(
            f"{k + 1:>4} | "
            f"{event['idx']:>8} | "
            f"{event['time_s']:>10.4f} | "
            f"{event['side']:>4} | "
            f"{event['label_col']:>8} | "
            f"{event.get('label_value', ''):>6} | "
            f"{event.get('event_rule', ''):>34} | "
            f"{value:>10}"
        )

    print("-" * 130)
    print(f"Event count: {len(events)}")
    print(f"Label count: {len(sequence)}")

    if len(events) == len(sequence):
        print("[检查] 事件数量与标签数量一致。")
    elif len(events) < len(sequence):
        print(f"[提示] 标签数量比事件数量多 {len(sequence) - len(events)} 个。")
    else:
        print(f"[提示] 事件数量比标签数量多 {len(events) - len(sequence)} 个。")

    print("=" * 130)


# ============================================================
# 15. Label statistics
# ============================================================

def print_label_statistics(df):
    print("=" * 80)
    print("[Original gait label statistics]")

    for col in ["l_label", "r_label"]:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).sort_index())

    print("=" * 80)


def print_terrain_statistics(df):
    print("=" * 80)
    print("[New terrain label statistics]")

    for col in ["terrain_single_label", "terrain_same_label"]:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).sort_index())

    print("=" * 80)


# ============================================================
# 16. Check alternating left/right events
# ============================================================

def check_left_right_alternation(events):
    """
    检查特殊规则之后，L/R label==3 是否大致交替。
    """
    later_events = [
        e for e in events
        if e.get("event_rule") == "later_l_or_r_label_3"
    ]

    if len(later_events) < 2:
        return

    print("=" * 80)
    print("[左右事件交替检查：仅检查特殊规则之后的后续事件]")

    warnings = []

    for i in range(1, len(later_events)):
        prev_side = later_events[i - 1]["side"]
        curr_side = later_events[i]["side"]

        if curr_side == prev_side:
            warnings.append(
                (
                    i,
                    later_events[i - 1],
                    later_events[i],
                )
            )

    if not warnings:
        print("[检查通过] 后续 L/R label==3 事件呈交替出现。")
    else:
        print(f"[警告] 发现 {len(warnings)} 处连续同侧事件：")

        for i, prev_event, curr_event in warnings:
            print(
                f"  后续事件 No.{i} 和 No.{i + 1}: "
                f"{prev_event['side']} -> {curr_event['side']} | "
                f"time {prev_event['time_s']:.4f}s -> {curr_event['time_s']:.4f}s"
            )

    print("=" * 80)


# ============================================================
# 17. Process a single file
# ============================================================

def process_one_file(input_file):
    print("\n")
    print("#" * 100)
    print(f"[Processing] {input_file}")
    print("#" * 100)

    direction = detect_z234_direction_from_filename(input_file)

    print(f"[Direction detection] 当前文件Direction: {direction}")

    df, sheet_name = read_excel_auto_sheet(input_file)

    required_cols = ["l_label", "r_label"]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            "Required columns are missing:\n"
            + "\n".join(missing)
            + "\n\n请确认输入文件是已经贴好步态标签的表格。"
        )

    print_label_statistics(df)

    events = collect_z234_terrain_events(df)

    if len(events) == 0:
        raise RuntimeError("没有检测到任何可用于贴地形标签的事件。")

    single_sequence, same_sequence = get_z234_sequences(direction)

    print(f"[Event detection] {direction} 共检测到地形切换事件: {len(events)} 个")
    print(f"[标签数量] 单侧Label count: {len(single_sequence)}")
    print(f"[标签数量] 同侧Label count: {len(same_sequence)}")

    # ========================================================
    # Strict check: label count must match event count
    # Do not create an output file if counts differ
    # ========================================================

    single_ok = check_event_label_count_match(
        events=events,
        sequence=single_sequence,
        sequence_name=f"{direction} 单侧地形标签",
    )

    same_ok = check_event_label_count_match(
        events=events,
        sequence=same_sequence,
        sequence_name=f"{direction} 同侧地形标签",
    )

    if not single_ok or not same_ok:
        print("=" * 80)
        print(f"[跳过] {direction} 标签数量和事件数量对不上，不生成新文件。")
        print(f"输入File: {input_file}")
        print("=" * 80)
        return None

    check_left_right_alternation(events)

    if PRINT_EVENTS:
        print_event_mapping(
            events,
            single_sequence,
            title=f"[{direction} 单侧地形label mapping]",
        )

        print_event_mapping(
            events,
            same_sequence,
            title=f"[{direction} 同侧地形label mapping]",
        )

    # Assign single-side terrain labels
    df = apply_terrain_sequence(
        df=df,
        events=events,
        terrain_sequence=single_sequence,
        output_col="terrain_single_label",
        initial_label=0,
        keep_last=False,
    )

    # Assign same-side terrain labels
    df = apply_terrain_sequence(
        df=df,
        events=events,
        terrain_sequence=same_sequence,
        output_col="terrain_same_label",
        initial_label=0,
        keep_last=False,
    )

    # Force labels to 0 from the final l_label == 3 start event
    if FORCE_ZERO_FROM_LAST_LEFT_LABEL3:
        df = force_zero_from_last_left_label3(
            df=df,
            output_col="terrain_single_label",
        )

        df = force_zero_from_last_left_label3(
            df=df,
            output_col="terrain_same_label",
        )

    print_terrain_statistics(df)

    output_file = build_output_file(input_file)

    actual_output_file = save_excel_safely(
        df=df,
        output_file=output_file,
        sheet_name=sheet_name if sheet_name else "data",
    )

    print("=" * 80)
    print("[完成]")
    print(f"Direction: {direction}")
    print(f"输入File: {input_file}")
    print(f"输出File: {actual_output_file}")
    print("Added columns:")
    print("  terrain_single_label  单侧地形标签")
    print("  terrain_same_label    同侧地形标签")
    print("=" * 80)

    return actual_output_file


# ============================================================
# 18. Main entry point
# ============================================================

def main():
    print("=" * 100)
    print("[Z2/Z3/Z4 方向地形标签批处理开始]")
    print(f"Files to process: {len(INPUT_FILES)}")
    print("=" * 100)

    output_files = []

    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            print("=" * 80)
            print("[警告] 文件不存在，跳过：")
            print(input_file)
            print("=" * 80)
            continue

        try:
            actual_output_file = process_one_file(input_file)

            if actual_output_file is not None:
                output_files.append(actual_output_file)

        except Exception as e:
            print("=" * 80)
            print("[错误] 当前文件处理失败，已跳过，不影响后续文件。")
            print(f"File: {input_file}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("=" * 80)
            continue

    print("\n")
    print("#" * 100)
    print("[All processing complete]")
    print(f"Successfully generated files: {len(output_files)}")

    for f in output_files:
        print(f"  {f}")

    print("#" * 100)


if __name__ == "__main__":
    main()
