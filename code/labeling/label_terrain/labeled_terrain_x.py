import os
import re
import zipfile
import numpy as np
import pandas as pd


# ============================================================
# 0. Configuration
# ============================================================

SUB_ID = "sub_048"

# Root directory containing the data
BASE_DIR = r"E:\data"

# Current participant directory
SUB_DIR = os.path.join(BASE_DIR, f"{SUB_ID}_label")

# Both forward (fwd) and backward (bwd) directions can be processed
INPUT_FILES = [
    os.path.join(SUB_DIR, f"{SUB_ID}_x_fwd_aligned_labeled.xlsx"),
    os.path.join(SUB_DIR, f"{SUB_ID}_x_bwd_aligned_labeled.xlsx"),
]

# Whether to print detailed event mappings
# False = print only event counts and final results, not the detection process
PRINT_EVENTS = False

# Whether to force labels to 0 after the final event
FORCE_ZERO_FROM_LAST_SPECIAL_SIDE_LABEL3 = True


# ============================================================
# 1. Single-side terrain label sequence for x_fwd
# ============================================================

FWD_SINGLE_TERRAIN_SEQUENCE = [
    3,
    -3,
    5,
    0,

    5,
    0,
    5,
    0,
    5,
    0,
    5,
    -5,

    -20,
    -5,
    0,
    -5,
    0,
    -5,
    0,
    -5,

    5,
    15,
    5,
    0,
    5,
    0,
    5,
    -5,

    -10,
    -5,
    0,
    -5,
    5,
    5,
    5,
    -5,

    -25,
    28,
    -3,

    0,
]


# ============================================================
# 2. Bilateral terrain label sequence for x_fwd
# ============================================================

FWD_DOUBLE_TERRAIN_SEQUENCE = [
    3,
    6,
    11,
    11,
    16,
    16,
    21,
    21,
    26,
    26,
    31,
    36,

    16,
    21,
    21,
    26,
    26,
    31,
    31,
    36,
    41,
    26,
    31,
    31,

    46,
    36,
    41,
    46,
    36,
    41,
    41,
    46,
    51,
    46,
    51,
    56,

    31,
    3,
    0,

    0,
]


# ============================================================
# 3. Single-side and bilateral terrain label sequences for x_bwd
# ============================================================

# Single-side bwd: reverse the single-side fwd sequence, exclude its final 0, then append 0
BWD_SINGLE_TERRAIN_SEQUENCE = list(reversed(FWD_SINGLE_TERRAIN_SEQUENCE[:-1])) + [0]

# Bilateral bwd: use the specified sequence
BWD_DOUBLE_TERRAIN_SEQUENCE = [
    3,
    31,
    56,
    51,
    46,
    51,
    46,
    41,
    41,
    36,

    46,
    41,
    46,
    46,
    31,
    31,
    26,
    41,
    36,
    31,
    31,
    26,

    26,
    21,
    21,
    16,
    36,
    31,
    26,
    26,
    21,
    21,

    16,
    16,
    11,
    11,
    6,
    3,
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

    安静版：
        不打印worksheet inspection过程。
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
            f"Error: {e}"
        )

    preferred_sheets = ["data", "Data", "DATA"]

    sheet_name = None

    for name in preferred_sheets:
        if name in xls.sheet_names:
            sheet_name = name
            break

    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    df = pd.read_excel(input_file, sheet_name=sheet_name)

    return df, sheet_name


# ============================================================
# 5. Determine the fwd/bwd direction
# ============================================================

def detect_direction_from_filename(input_file):
    """
    从文件名判断方向。

    文件名中包含 x_fwd -> fwd
    文件名中包含 x_bwd -> bwd
    """
    basename = os.path.basename(input_file).lower()

    if "x_fwd" in basename:
        return "fwd"

    if "x_bwd" in basename:
        return "bwd"

    raise ValueError(
        "无法从文件名判断方向，请确保文件名包含 x_fwd 或 x_bwd：\n"
        f"{input_file}"
    )


# ============================================================
# 6. Generate the output directory and file path
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
    Create the output directory and output file path.

    输入：
        D:/PythonProject1/sub_030_label/sub_030_x_fwd_aligned_labeled.xlsx

    Output directory:
        D:/PythonProject1/sub_030_labeled_terrain

    Output file:
        D:/PythonProject1/sub_030_labeled_terrain/sub_030_x_fwd_aligned_labeled_terrain.xlsx
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

    If the output cannot be overwritten because it is open in Excel/WPS,
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

        return output_file

    except PermissionError:
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

                return new_file

            except PermissionError:
                continue

        raise PermissionError(
            "Could not save using alternative filenames. Close Excel/WPS and try again."
        )


# ============================================================
# 8. General event-detection functions
# ============================================================

def detect_label_value_rising_edges(df, label_col, label_value, side_name):
    """
    Detect the start of label_value in label_col.

    rising edge 条件：
        Current row label == label_value
        and previous row label != label_value

    A consecutive run of label == 3 counts as one event.
    """
    if label_col not in df.columns:
        raise KeyError(f"Missing column: {label_col}")

    label = pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy()

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
# 9. fwd rule: left foot leads
# ============================================================

def collect_fwd_terrain_events(df):
    """
    x_fwd 地形事件收集规则：

    Event 1:
        First l_label == 3

    Event 2:
        Second l_label == 3

    Event 3:
        First r_label == 3 after the second l_label == 3

    Event 4 and later:
        After the third event: l_label == 3 or r_label == 3
    """
    if "l_label" not in df.columns:
        raise KeyError("Missing column: l_label")

    if "r_label" not in df.columns:
        raise KeyError("Missing column: r_label")

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
            "x_fwd: l_label == 3 的起始事件少于 2 个，无法执行 fwd 规则。"
        )

    first_event = dict(l3_events[0])
    first_event["event_rule"] = "fwd_first_l_label_3"
    events.append(first_event)

    second_event = dict(l3_events[1])
    second_event["event_rule"] = "fwd_second_l_label_3"
    events.append(second_event)

    second_idx = second_event["idx"]

    r3_after_second = [
        e for e in r3_events
        if e["idx"] > second_idx
    ]

    if len(r3_after_second) == 0:
        raise RuntimeError(
            "x_fwd: 已经找到Second l_label == 3，"
            "但其后没有找到 r_label == 3，无法执行第 3 次贴标。"
        )

    third_event = dict(r3_after_second[0])
    third_event["event_rule"] = "fwd_third_r_label_3"
    events.append(third_event)

    third_idx = third_event["idx"]

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
        e["event_rule"] = "fwd_later_l_or_r_label_3"

    events.extend(later_events)

    return events, l3_events, r3_events


# ============================================================
# 10. bwd rule: right foot leads
# ============================================================

def collect_bwd_terrain_events(df):
    """
    x_bwd 地形事件收集规则：

    Event 1:
        第一个 r_label == 3

    Event 2:
        第二个 r_label == 3

    Event 3:
        第二个 r_label == 3 之后，First l_label == 3

    Event 4 and later:
        After the third event: l_label == 3 or r_label == 3
    """
    if "l_label" not in df.columns:
        raise KeyError("Missing column: l_label")

    if "r_label" not in df.columns:
        raise KeyError("Missing column: r_label")

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

    if len(r3_events) < 2:
        raise RuntimeError(
            "x_bwd: r_label == 3 的起始事件少于 2 个，无法执行 bwd 规则。"
        )

    first_event = dict(r3_events[0])
    first_event["event_rule"] = "bwd_first_r_label_3"
    events.append(first_event)

    second_event = dict(r3_events[1])
    second_event["event_rule"] = "bwd_second_r_label_3"
    events.append(second_event)

    second_idx = second_event["idx"]

    l3_after_second = [
        e for e in l3_events
        if e["idx"] > second_idx
    ]

    if len(l3_after_second) == 0:
        raise RuntimeError(
            "x_bwd: 已经找到第二个 r_label == 3，"
            "但其后没有找到 l_label == 3，无法执行第 3 次贴标。"
        )

    third_event = dict(l3_after_second[0])
    third_event["event_rule"] = "bwd_third_l_label_3"
    events.append(third_event)

    third_idx = third_event["idx"]

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
        e["event_rule"] = "bwd_later_l_or_r_label_3"

    events.extend(later_events)

    return events, l3_events, r3_events


# ============================================================
# 11. Get events and label sequences for the selected direction
# ============================================================

def get_events_and_sequences_by_direction(df, direction):
    """
    根据方向 fwd/bwd 返回：
        events
        l3_events
        r3_events
        single_sequence
        double_sequence
        force_zero_side
    """
    if direction == "fwd":
        events, l3_events, r3_events = collect_fwd_terrain_events(df)

        single_sequence = FWD_SINGLE_TERRAIN_SEQUENCE
        double_sequence = FWD_DOUBLE_TERRAIN_SEQUENCE

        # For fwd, set labels to 0 from the final l_label == 3 event onward
        force_zero_side = "L"

        return events, l3_events, r3_events, single_sequence, double_sequence, force_zero_side

    if direction == "bwd":
        events, l3_events, r3_events = collect_bwd_terrain_events(df)

        single_sequence = BWD_SINGLE_TERRAIN_SEQUENCE
        double_sequence = BWD_DOUBLE_TERRAIN_SEQUENCE

        # For bwd, set labels to 0 from the final r_label == 3 event onward
        force_zero_side = "R"

        return events, l3_events, r3_events, single_sequence, double_sequence, force_zero_side

    raise ValueError(f"Unknown direction: {direction}")


# ============================================================
# 12. Validate counts and print count results
# ============================================================

def check_event_label_count_match(events, sequence):
    """
    检查事件数量和标签数量是否一致。
    """
    return len(events) == len(sequence)


def print_count_summary(
    direction,
    events,
    l3_events,
    r3_events,
    single_sequence,
    double_sequence,
):
    """
    Print a concise count summary:

    原始标签数量：
        你给的标签序列数量。

    事件标签数量：
        根据 l_label/r_label == 3 实际检测到的贴标事件数量。
    """
    single_count = len(single_sequence)
    double_count = len(double_sequence)
    event_count = len(events)

    single_ok = single_count == event_count
    double_ok = double_count == event_count

    print("=" * 100)
    print(f"[数量结果] x_{direction}")

    print(
        f"Original / provided label count: "
        f"单侧={single_count}, "
        f"双侧={double_count}"
    )

    print(
        f"Detected labeling-event count: "
        f"{event_count}"
    )

    print(
        f"Original label == 3 start-event count: "
        f"l_label==3 events={len(l3_events)}, "
        f"r_label==3 events={len(r3_events)}"
    )

    if single_ok and double_ok:
        print("Count validation: passed")
    else:
        print("Count validation: failed")

        if not single_ok:
            print(
                f"  单侧不一致: "
                f"Original label count={single_count}, "
                f"Labeled event count={event_count}"
            )

        if not double_ok:
            print(
                f"  双侧不一致: "
                f"Original label count={double_count}, "
                f"Labeled event count={event_count}"
            )

    print("=" * 100)


# ============================================================
# 13. Assign terrain labels from the event sequence
# ============================================================

def apply_terrain_sequence(
    df,
    events,
    terrain_sequence,
    output_col,
    initial_label=0,
):
    """
    Assign terrain labels from the event sequence.

    Labeling logic:
        Before the first event:
            initial_label

        From event k to immediately before event k+1:
            terrain_sequence[k]

        After the final event:
            terrain_sequence[-1]

    严格模式：
        事件数量必须等于标签数量。
    """
    n = len(df)

    terrain = np.full(
        n,
        initial_label,
        dtype=float,
    )

    if len(events) == 0:
        df[output_col] = terrain.astype(int)
        return df

    if len(events) != len(terrain_sequence):
        raise ValueError(
            f"{output_col}: 事件数量 {len(events)} "
            f"和地形序列长度 {len(terrain_sequence)} 不一致。"
        )

    first_idx = events[0]["idx"]

    if first_idx > 0:
        terrain[:first_idx] = initial_label

    for k, event in enumerate(events):
        start_idx = event["idx"]

        if k + 1 < len(events):
            end_idx = events[k + 1]["idx"]
        else:
            end_idx = n

        value = terrain_sequence[k]

        terrain[start_idx:end_idx] = value

    if np.all(np.isfinite(terrain)):
        terrain = terrain.astype(int)

    df[output_col] = terrain

    return df


# ============================================================
# 14. Force labels to 0 after the final label == 3 event on the specified side
# ============================================================

def force_zero_from_last_side_label3(df, output_cols, side):
    """
    从最后一个指定侧 label == 3 的起始事件开始，
    Force the selected terrain label column to 0.

    side:
        "L" -> l_label
        "R" -> r_label
    """
    if side == "L":
        label_col = "l_label"
        side_name = "L"
    elif side == "R":
        label_col = "r_label"
        side_name = "R"
    else:
        raise ValueError("side 只能是 'L' 或 'R'")

    if label_col not in df.columns:
        raise KeyError(f"Missing column: {label_col}")

    label3_events = detect_label_value_rising_edges(
        df=df,
        label_col=label_col,
        label_value=3,
        side_name=side_name,
    )

    if len(label3_events) == 0:
        return df

    last_idx = int(label3_events[-1]["idx"])

    for output_col in output_cols:
        if output_col in df.columns:
            df.loc[last_idx:, output_col] = 0

    return df


# ============================================================
# 15. Optional: print event-to-label mappings
# ============================================================

def print_event_mapping(events, sequence, title):
    """
    Optionally print event mappings.
    Not called by default because PRINT_EVENTS=False.
    """
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
        f"{'rule':>28} | "
        f"{'terrain':>10}"
    )
    print("-" * 130)

    for k, event in enumerate(events):
        value = sequence[k]

        print(
            f"{k + 1:>4} | "
            f"{event['idx']:>8} | "
            f"{event['time_s']:>10.4f} | "
            f"{event['side']:>4} | "
            f"{event['label_col']:>8} | "
            f"{event.get('label_value', ''):>6} | "
            f"{event.get('event_rule', ''):>28} | "
            f"{value:>10}"
        )

    print("-" * 130)
    print(f"Event count: {len(events)}")
    print(f"Label count: {len(sequence)}")
    print("=" * 130)


# ============================================================
# 16. Process a single file
# ============================================================

def process_one_file(input_file):
    direction = detect_direction_from_filename(input_file)

    df, sheet_name = read_excel_auto_sheet(input_file)

    required_cols = ["l_label", "r_label"]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            "Required columns are missing from the workbook:\n"
            + "\n".join(missing)
            + "\n\n请确认输入文件是已经贴好步态标签的表格。"
        )

    (
        events,
        l3_events,
        r3_events,
        single_sequence,
        double_sequence,
        force_zero_side,
    ) = get_events_and_sequences_by_direction(
        df=df,
        direction=direction,
    )

    if len(events) == 0:
        raise RuntimeError("没有检测到任何可用于贴地形标签的事件。")

    # Print counts for the original labels and event labels
    print_count_summary(
        direction=direction,
        events=events,
        l3_events=l3_events,
        r3_events=r3_events,
        single_sequence=single_sequence,
        double_sequence=double_sequence,
    )

    # Strictly verify that the number of labels matches the number of events
    single_ok = check_event_label_count_match(
        events=events,
        sequence=single_sequence,
    )

    double_ok = check_event_label_count_match(
        events=events,
        sequence=double_sequence,
    )

    if not single_ok or not double_ok:
        raise RuntimeError(
            f"x_{direction} 标签数量和事件数量不一致，不生成新文件。\n"
            f"Event count: {len(events)}\n"
            f"单侧原始Label count: {len(single_sequence)}\n"
            f"双侧原始Label count: {len(double_sequence)}"
        )

    if PRINT_EVENTS:
        print_event_mapping(
            events,
            single_sequence,
            title=f"[x_{direction} 单侧地形label mapping]",
        )

        print_event_mapping(
            events,
            double_sequence,
            title=f"[x_{direction} 双侧地形label mapping]",
        )

    # Assign single-side terrain labels
    df = apply_terrain_sequence(
        df=df,
        events=events,
        terrain_sequence=single_sequence,
        output_col="terrain_single_label",
        initial_label=0,
    )

    # Assign bilateral terrain labels
    df = apply_terrain_sequence(
        df=df,
        events=events,
        terrain_sequence=double_sequence,
        output_col="terrain_double_label",
        initial_label=0,
    )

    # Force labels to 0 after the final label == 3 event on the specified side
    if FORCE_ZERO_FROM_LAST_SPECIAL_SIDE_LABEL3:
        df = force_zero_from_last_side_label3(
            df=df,
            output_cols=[
                "terrain_single_label",
                "terrain_double_label",
            ],
            side=force_zero_side,
        )

    output_file = build_output_file(input_file)

    actual_output_file = save_excel_safely(
        df=df,
        output_file=output_file,
        sheet_name=sheet_name if sheet_name else "data",
    )

    return {
        "direction": f"x_{direction}",
        "input_file": input_file,
        "output_file": actual_output_file,
        "row_count": len(df),
        "event_count": len(events),
        "l3_event_count": len(l3_events),
        "r3_event_count": len(r3_events),
        "single_label_count": len(single_sequence),
        "double_label_count": len(double_sequence),
    }


# ============================================================
# 17. Main entry point
# ============================================================

def main():
    print("=" * 100)
    print("[x_fwd / x_bwd 地形标签批处理开始]")
    print(f"Subject: {SUB_ID}")
    print(f"Input directory: {SUB_DIR}")
    print(f"Files to process: {len(INPUT_FILES)}")
    print("=" * 100)

    results = []
    failed_files = []

    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            failed_files.append(
                {
                    "file": input_file,
                    "reason": "文件不存在",
                }
            )

            print("=" * 100)
            print("[Skipped] 文件不存在")
            print(input_file)
            print("=" * 100)

            continue

        try:
            result = process_one_file(input_file)
            results.append(result)

            print(
                f"[Completed] {result['direction']} | "
                f"Rows: {result['row_count']} | "
                f"原始Label count: 单侧={result['single_label_count']}, "
                f"双侧={result['double_label_count']} | "
                f"Labeled event count={result['event_count']} | "
                f"Output: {result['output_file']}"
            )

        except Exception as e:
            failed_files.append(
                {
                    "file": input_file,
                    "reason": f"{type(e).__name__}: {e}",
                }
            )

            print("=" * 100)
            print("[Failed] 当前文件处理失败，已跳过")
            print(f"File: {input_file}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error: {e}")
            print("=" * 100)

            continue

    print("\n")
    print("#" * 100)
    print("[All processing complete]")
    print(f"Success count: {len(results)}")
    print(f"Failure count: {len(failed_files)}")
    print("#" * 100)

    if results:
        print("\n[成功生成文件]")
        for r in results:
            print(
                f"{r['direction']} | "
                f"原始Label count: 单侧={r['single_label_count']}, "
                f"双侧={r['double_label_count']} | "
                f"Labeled event count={r['event_count']} | "
                f"l_label==3 events={r['l3_event_count']} | "
                f"r_label==3 events={r['r3_event_count']} | "
                f"输出={r['output_file']}"
            )

    if failed_files:
        print("\n[失败文件]")
        for item in failed_files:
            print(f"File: {item['file']}")
            print(f"Reason: {item['reason']}")
            print("-" * 100)


if __name__ == "__main__":
    main()
