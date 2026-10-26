import os
import re
import numpy as np
import pandas as pd


# ============================================================
# 0. Configuration
# ============================================================

# Only this section needs to be edited
# Example:
# SUB_ID = "sub_006"
# SUB_ID = "sub_031"
# SUB_ID = "sub_101"
SUB_ID = "sub_041"

# Data root directory
BASE_DIR = r"E:\data"

# Current participant directory
SUB_DIR = os.path.join(BASE_DIR, f"{SUB_ID}_label")

INPUT_FILES = [
    os.path.join(SUB_DIR, f"{SUB_ID}_y_fwd_aligned_labeled.xlsx"),
    os.path.join(SUB_DIR, f"{SUB_ID}_y_bwd_aligned_labeled.xlsx"),
]

# Whether to print detailed event mappings
# False: print only counts and final results, not the event detection trace
PRINT_EVENTS = False

# Force labels to 0 from the final r_label == 3 event onward
FORCE_ZERO_FROM_LAST_RIGHT_LABEL3 = True


# ============================================================
# 1. Terrain labels for Y_fwd
# ============================================================

Y_FWD_SINGLE_TERRAIN_SEQUENCE = [
    45,
    90,
    100,
    100,
    110,
    110,
    120,
    120,
    130,
    130,
    140,
    140,
    110,
    120,
    120,
    110,
    130,
    140,
    140,
    150,
    160,
    130,
    140,
    140,
    150,
    150,
    160,
    170,
    150,
    160,
    160,
    170,
    180,
    170,
    180,
    190,
    140,
    45,
    0,
]

Y_FWD_SAME_TERRAIN_SEQUENCE = [
    45,
    45,
    55,
    45,
    65,
    45,
    75,
    45,
    85,
    45,
    95,
    55,
    55,
    65,
    55,
    75,
    55,
    85,
    55,
    95,
    65,
    65,
    75,
    65,
    85,
    65,
    95,
    75,
    75,
    85,
    75,
    95,
    85,
    85,
    95,
    95,
    45,
    0,
    0,
]


# ============================================================
# 2. Terrain labels for Y_bwd
# ============================================================

Y_BWD_SINGLE_TERRAIN_SEQUENCE = [
    45,
    140,
    190,
    180,
    170,
    180,
    170,
    160,
    160,
    150,
    170,
    160,
    150,
    150,
    140,
    140,
    130,
    160,
    150,
    140,
    140,
    130,
    110,
    120,
    120,
    110,
    140,
    140,
    130,
    130,
    120,
    120,
    110,
    110,
    100,
    100,
    90,
    45,
    0,
]

Y_BWD_SAME_TERRAIN_SEQUENCE = [
    45,
    95,
    95,
    85,
    85,
    95,
    75,
    85,
    75,
    75,
    95,
    65,
    85,
    65,
    75,
    65,
    65,
    95,
    55,
    85,
    55,
    75,
    55,
    65,
    55,
    55,
    95,
    45,
    85,
    45,
    75,
    45,
    65,
    45,
    55,
    45,
    45,
    0,
    0,
]


# ============================================================
# 3. Automatic Excel loading
# ============================================================

def read_excel_auto_sheet(input_file):
    """
    Automatically load the Excel workbook.

    Prefer the data/Data/DATA worksheet.
    Otherwise read the first worksheet.

    安静版：
        不打印 Excel worksheet inspection过程。
    """
    try:
        xls = pd.ExcelFile(input_file)

    except Exception as e:
        raise RuntimeError(
            "Failed to read the Excel workbook.\n"
            f"File: {input_file}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {e}"
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
# 4. Determine y_fwd / y_bwd
# ============================================================

def detect_y_direction_from_filename(input_file):
    """
    从文件名判断 y_fwd / y_bwd。
    """
    basename = os.path.basename(input_file).lower()

    if "y_fwd" in basename:
        return "y_fwd"

    if "y_bwd" in basename:
        return "y_bwd"

    raise ValueError(
        "无法从文件名判断方向，请确保文件名包含 y_fwd 或 y_bwd：\n"
        f"{input_file}"
    )


# ============================================================
# 5. Generate the output path
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
# 6. Save the Excel workbook safely
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
            "Could not save using alternative names; close Excel/WPS and retry."
        )


# ============================================================
# 7. General event-detection functions
# ============================================================

def detect_label_value_rising_edges(df, label_col, label_value, side_name):
    """
    Detect the first sample of a label_value run in label_col.

    rising edge 条件：
        当前行 label == label_value
        且上一行 label != label_value

    Each contiguous run of label == 3 counts as one event.
    """
    if label_col not in df.columns:
        raise KeyError(f"缺少列: {label_col}")

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
# 8. Shared event logic for y_fwd / y_bwd
# ============================================================

def collect_y_terrain_events_like_x_fwd(df):
    """
    Y_fwd 和 Y_bwd 都使用 x_fwd 的Labeling logic:

    Event 1:
        First l_label == 3

    Event 2:
        Second l_label == 3

    Event 3:
        First r_label == 3 after the second l_label == 3

    Event 4 and later:
        After the third event: l_label == 3 or r_label == 3

    返回：
        events
        l3_events
        r3_events
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
            "l_label == 3 的起始事件少于 2 个，无法执行 y 的贴标规则。"
        )

    if len(r3_events) < 1:
        raise RuntimeError(
            "r_label == 3 的起始事件少于 1 个，无法执行 y 的贴标规则。"
        )

    first_event = dict(l3_events[0])
    first_event["event_rule"] = "first_l_label_3"
    events.append(first_event)

    second_event = dict(l3_events[1])
    second_event["event_rule"] = "second_l_label_3"
    events.append(second_event)

    second_idx = second_event["idx"]

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
    third_event["event_rule"] = "third_r_label_3"
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
        e["event_rule"] = "later_l_or_r_label_3"

    events.extend(later_events)

    return events, l3_events, r3_events


# ============================================================
# 9. Get label sequences for y_fwd / y_bwd
# ============================================================

def get_y_sequences(direction):
    """
    根据方向返回单侧和同侧地形标签序列。
    """
    if direction == "y_fwd":
        return Y_FWD_SINGLE_TERRAIN_SEQUENCE, Y_FWD_SAME_TERRAIN_SEQUENCE

    if direction == "y_bwd":
        return Y_BWD_SINGLE_TERRAIN_SEQUENCE, Y_BWD_SAME_TERRAIN_SEQUENCE

    raise ValueError(f"未知Direction: {direction}")


# ============================================================
# 10. Check counts and print results
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
    same_sequence,
):
    """
    Print a concise count summary:

    原始标签数量：
        你给的标签序列数量。

    事件标签数量：
        根据 l_label/r_label == 3 实际检测到的贴标事件数量。
    """
    single_count = len(single_sequence)
    same_count = len(same_sequence)
    event_count = len(events)

    single_ok = single_count == event_count
    same_ok = same_count == event_count

    print("=" * 100)
    print(f"[数量结果] {direction}")

    print(
        f"原始标签数量/你给的Label count: "
        f"单侧={single_count}, "
        f"同侧={same_count}"
    )

    print(
        f"事件标签数量/实际检测到的贴标Event count: "
        f"{event_count}"
    )

    print(
        f"原始 label==3 起始Event count: "
        f"l_label==3 events={len(l3_events)}, "
        f"r_label==3 events={len(r3_events)}"
    )

    if single_ok and same_ok:
        print("数量检查结果: 通过")
    else:
        print("数量检查结果: 不通过")

        if not single_ok:
            print(
                f"  单侧不一致: "
                f"Original label count={single_count}, "
                f"Labeled event count={event_count}"
            )

        if not same_ok:
            print(
                f"  同侧不一致: "
                f"Original label count={same_count}, "
                f"Labeled event count={event_count}"
            )

    print("=" * 100)


# ============================================================
# 11. Assign terrain labels from event sequences
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

        From event k to just before event k+1:
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
# 12. Force labels to 0 after the final r_label == 3 event
# ============================================================

def force_zero_from_last_right_label3(df, output_cols):
    """
    从最后一个 r_label == 3 的起始事件开始，
    Force the specified terrain-label column to 0.
    """
    if "r_label" not in df.columns:
        raise KeyError("缺少列: r_label")

    r3_events = detect_label_value_rising_edges(
        df=df,
        label_col="r_label",
        label_value=3,
        side_name="R",
    )

    if len(r3_events) == 0:
        return df

    last_right_idx = int(r3_events[-1]["idx"])

    for output_col in output_cols:
        if output_col in df.columns:
            df.loc[last_right_idx:, output_col] = 0

    return df


# ============================================================
# 13. Optional event-to-label mapping output
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
        f"{'rule':>24} | "
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
            f"{event.get('event_rule', ''):>24} | "
            f"{value:>10}"
        )

    print("-" * 130)
    print(f"Event count: {len(events)}")
    print(f"Label count: {len(sequence)}")
    print("=" * 130)


# ============================================================
# 14. Process a single file
# ============================================================

def process_one_file(input_file):
    direction = detect_y_direction_from_filename(input_file)

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

    events, l3_events, r3_events = collect_y_terrain_events_like_x_fwd(df)

    if len(events) == 0:
        raise RuntimeError("没有检测到任何可用于贴地形标签的事件。")

    single_sequence, same_sequence = get_y_sequences(direction)

    # Print counts of original labels and event labels
    print_count_summary(
        direction=direction,
        events=events,
        l3_events=l3_events,
        r3_events=r3_events,
        single_sequence=single_sequence,
        same_sequence=same_sequence,
    )

    # Strict check: label count must match event count
    single_ok = check_event_label_count_match(
        events=events,
        sequence=single_sequence,
    )

    same_ok = check_event_label_count_match(
        events=events,
        sequence=same_sequence,
    )

    if not single_ok or not same_ok:
        raise RuntimeError(
            f"{direction} 标签数量和事件数量不一致，不生成新文件。\n"
            f"事件Label count: {len(events)}\n"
            f"单侧原始Label count: {len(single_sequence)}\n"
            f"同侧原始Label count: {len(same_sequence)}"
        )

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
    )

    # Assign same-side terrain labels
    df = apply_terrain_sequence(
        df=df,
        events=events,
        terrain_sequence=same_sequence,
        output_col="terrain_same_label",
        initial_label=0,
    )

    # Force labels to 0 from the final r_label == 3 start event
    if FORCE_ZERO_FROM_LAST_RIGHT_LABEL3:
        df = force_zero_from_last_right_label3(
            df=df,
            output_cols=[
                "terrain_single_label",
                "terrain_same_label",
            ],
        )

    output_file = build_output_file(input_file)

    actual_output_file = save_excel_safely(
        df=df,
        output_file=output_file,
        sheet_name=sheet_name if sheet_name else "data",
    )

    return {
        "direction": direction,
        "input_file": input_file,
        "output_file": actual_output_file,
        "row_count": len(df),
        "event_count": len(events),
        "l3_event_count": len(l3_events),
        "r3_event_count": len(r3_events),
        "single_label_count": len(single_sequence),
        "same_label_count": len(same_sequence),
    }


# ============================================================
# 15. Main entry point
# ============================================================

def main():
    print("=" * 100)
    print("[Y方向地形标签批处理开始]")
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
            print("[跳过] 文件不存在")
            print(input_file)
            print("=" * 100)

            continue

        try:
            result = process_one_file(input_file)
            results.append(result)

            print(
                f"[完成] {result['direction']} | "
                f"Rows: {result['row_count']} | "
                f"原始Label count: 单侧={result['single_label_count']}, "
                f"同侧={result['same_label_count']} | "
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
            print("[失败] 当前文件处理失败，已跳过")
            print(f"File: {input_file}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
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
                f"同侧={r['same_label_count']} | "
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
