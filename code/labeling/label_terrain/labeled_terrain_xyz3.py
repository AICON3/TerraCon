import os
import re
import zipfile
import numpy as np
import pandas as pd


# ============================================================
# 0. Configuration
# ============================================================

# Only this section needs to be changed
SUBJECT_ID = "sub_002"

# Project root directory
PROJECT_ROOT = r"E:\data"

# Name of the terrain to process
TERRAIN_NAME = "xyz3"

# Automatically generate the input directory
INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    f"{SUBJECT_ID}_label",
)

# Automatically generate fwd/bwd input file paths
INPUT_FILES = [
    os.path.join(
        INPUT_DIR,
        f"{SUBJECT_ID}_{TERRAIN_NAME}_fwd_aligned_labeled.xlsx",
    ),
    os.path.join(
        INPUT_DIR,
        f"{SUBJECT_ID}_{TERRAIN_NAME}_bwd_aligned_labeled.xlsx",
    ),
]

PRINT_EVENTS = True

# Force labels to 0 from the final r_label == 3 event onward
FORCE_ZERO_FROM_LAST_RIGHT_LABEL3 = True



# ============================================================
# 1. XYZ3 label sequence
# ============================================================

XYZ3_LABELS = {
    "xyz3_fwd": {
        "single": {
            "x": [19, 3, 20, 0, -20, -40, -40, 0, 40, 20, -20, -40, -12, 44,0],
            "y": [61.5, 126.5, 140, 130, 140, 130, 140, 125.5, 61, 61, 61, 61, 61, 30.5,0],
            "z": [0, 4, 0, 4, 12, -4, 3, 30, 30, 34, 30, 34, 30, 3,0],
        },
        "same": {
            "x": [19, 16, 36, 36, 16, 56, 16, 16, 56, 36, 16, 56, 44, 0,0],
            "y": [61.5, 65, 75, 55, 85, 45, 95, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 0,0],
            "z": [0, 4, -4, 8, 4, -8, 11, 19, 11, 23, 7, 27, 3, 0,0],
        },
    },
    "xyz3_bwd": {
        "single": {
            "x": [44, -12, -40, -20, 20, 40, 0, -40, -40, -20, 0, 20, 3, 19,0],
            "y": [30.5, 61, 61, 61, 61, 61, 125.5, 140, 130, 140, 130, 140, 126.5, 61.5,0],
            "z": [-3, -30, -34, -30, -34, -30, -30, -3, 4, -12, -4, 0, -4, 0,0],
        },
        "same": {
            "x": [44, 56, 16, 36, 56, 16, 16, 56, 16, 36, 36, 16, 19, 0,0],
            "y": [30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 95, 45, 85, 55, 75, 65, 61.5, 0,0],
            "z": [-3, -27, -7, -23, -11, -19, -11, 8, -4, -8, 4, -4, 0, 0,0],
        },
    },
}


# ============================================================
# 2. Automatic Excel loading
# ============================================================

def read_excel_auto_sheet(input_file):
    """
    自动读取 Excel 文件。

    优先读取 data/Data/DATA 工作表。
    如果没有，则读取第一个工作表。
    """
    try:
        xls = pd.ExcelFile(input_file)

    except zipfile.BadZipFile:
        raise RuntimeError(
            "当前文件不是有效的 .xlsx 文件，可能文件损坏，或者只是被改名成 .xlsx。\n"
            f"文件: {input_file}\n"
            "建议用 Excel/WPS 打开后重新另存为真正的 Excel 工作簿 (*.xlsx)。"
        )

    except Exception as e:
        raise RuntimeError(
            "读取 Excel 文件失败。\n"
            f"文件: {input_file}\n"
            f"错误类型: {type(e).__name__}\n"
            f"错误信息: {e}"
        )

    print("=" * 80)
    print("[Excel 工作表检测]")
    print(f"文件: {input_file}")
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

    print(f"[读取成功] 数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")

    if "time_s" in df.columns:
        print(f"[时间范围] {df['time_s'].iloc[0]} s -> {df['time_s'].iloc[-1]} s")

    print("=" * 80)

    return df, sheet_name


# ============================================================
# 3. Determine direction from the filename
# ============================================================

def detect_xyz3_direction_from_filename(input_file):
    """
    从文件名判断 xyz3_fwd / xyz3_bwd。
    """
    basename = os.path.basename(input_file).lower()

    if "xyz3_fwd" in basename:
        return "xyz3_fwd"

    if "xyz3_bwd" in basename:
        return "xyz3_bwd"

    raise ValueError(
        "无法从文件名判断方向，请确保文件名包含 xyz3_fwd 或 xyz3_bwd。\n"
        f"当前文件: {input_file}"
    )


# ============================================================
# 4. Output paths
# ============================================================

def extract_subject_id(input_file):
    """
    从路径或文件名中提取 sub_xxx。
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
    输出目录：
        sub_xxx_labeled_terrain

    输出文件：
        原文件名 + _terrain.xlsx
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


def save_excel_safely(df, output_file, sheet_name="data"):
    """
    安全保存 Excel。
    如果目标文件被打开，自动另存为 _v2, _v3...
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

        print(f"[保存成功] {output_file}")
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

                print(f"[保存成功] {new_file}")
                return new_file

            except PermissionError:
                continue

        raise PermissionError(
            "尝试多个备用文件名后仍无法保存，请关闭 Excel/WPS 后重试。"
        )


# ============================================================
# 5. Event detection
# ============================================================

def detect_label_value_rising_edges(df, label_col, label_value, side_name):
    """
    检测指定 label_col 中 label_value 的起始点。

    rising edge：
        当前行 == label_value
        且上一行 != label_value
    """
    if label_col not in df.columns:
        raise KeyError(f"缺少列: {label_col}")

    label = df[label_col].to_numpy()

    is_target = label == label_value
    prev_is_target = np.r_[False, is_target[:-1]]

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


def collect_xyz3_terrain_events(df):
    """
    xyz3 事件规则：

    初始为 0。

    第 1 个事件：
        第一个 l_label == 3

    第 2 个事件：
        第二个 l_label == 3

    第 3 个事件：
        第二个 l_label == 3 后第一个 r_label == 3

    第 4 个事件及以后：
        第三个事件之后，
        l_label == 3 或 r_label == 3

    最后：
        最后一个 r_label == 3 起始事件之后强制归 0。
    """
    if "l_label" not in df.columns:
        raise KeyError("缺少列: l_label")

    if "r_label" not in df.columns:
        raise KeyError("缺少列: r_label")

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
            "l_label == 3 的起始事件少于 2 个，无法执行 xyz3 贴标规则。"
        )

    if len(r3_events) < 1:
        raise RuntimeError(
            "r_label == 3 的起始事件少于 1 个，无法执行 xyz3 贴标规则。"
        )

    events = []

    first_event = dict(l3_events[0])
    first_event["event_rule"] = "first_l_label_3_start_first_label"
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
            "第二个 l_label == 3 后没有找到 r_label == 3。"
        )

    third_event = dict(r3_after_second[0])
    third_event["event_rule"] = "third_r_label_3_after_second_l3"
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

    return events


# ============================================================
# 6. Retrieve label sequences
# ============================================================

def get_xyz3_sequences(direction):
    """
    返回需要贴的 6 组标签。
    """
    if direction not in XYZ3_LABELS:
        raise ValueError(f"没有找到方向 {direction} 的标签。")

    item = XYZ3_LABELS[direction]

    return {
        "terrain_x_single_label": item["single"]["x"],
        "terrain_y_single_label": item["single"]["y"],
        "terrain_z_single_label": item["single"]["z"],
        "terrain_x_same_label": item["same"]["x"],
        "terrain_y_same_label": item["same"]["y"],
        "terrain_z_same_label": item["same"]["z"],
    }


# ============================================================
# 7. Validate counts
# ============================================================

def check_event_label_count_match(events, sequences, direction):
    """
    检查事件数量和每个标签序列长度是否一致。
    """
    event_count = len(events)

    print("=" * 80)
    print(f"[数量检查] {direction}")
    print(f"事件数量: {event_count}")

    all_ok = True

    for col, seq in sequences.items():
        label_count = len(seq)
        print(f"{col}: 标签数量 = {label_count}")

        if label_count != event_count:
            all_ok = False
            print(
                f"[不一致] {col}: "
                f"事件数量 {event_count}, 标签数量 {label_count}"
            )

    if all_ok:
        print("[检查通过] 所有标签数量均与事件数量一致。")
    else:
        print("[检查失败] 标签数量和事件数量不一致，将跳过当前文件，不生成新文件。")

    print("=" * 80)

    return all_ok


# ============================================================
# 8. Assign labels
# ============================================================

def apply_terrain_sequence(df, events, terrain_sequence, output_col, initial_label=0):
    """
    根据事件序列贴地形标签。

    逻辑：
        第一个事件之前：
            initial_label

        第 k 个事件 到 第 k+1 个事件之前：
            terrain_sequence[k]

        最后一个事件之后：
            terrain_sequence[-1]
    """
    if len(events) != len(terrain_sequence):
        raise ValueError(
            f"{output_col}: 事件数量 {len(events)} "
            f"和标签数量 {len(terrain_sequence)} 不一致。"
        )

    n = len(df)

    terrain = np.full(
        n,
        initial_label,
        dtype=float,
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

    df[output_col] = terrain

    return df


def force_zero_from_last_right_label3(df, output_cols):
    """
    从最后一个 r_label == 3 起始事件开始，
    将所有输出列强制置为 0。
    """
    r3_events = detect_label_value_rising_edges(
        df=df,
        label_col="r_label",
        label_value=3,
        side_name="R",
    )

    if len(r3_events) == 0:
        print("[警告] 未找到 r_label == 3 起始事件，无法执行最后归 0。")
        return df

    last_right_idx = int(r3_events[-1]["idx"])

    if "time_s" in df.columns:
        t = df.loc[last_right_idx, "time_s"]
        print(
            f"[最后归0] 从最后一个 r_label == 3 开始置 0，"
            f"idx={last_right_idx}, time={t:.4f}s"
        )
    else:
        print(
            f"[最后归0] 从最后一个 r_label == 3 开始置 0，"
            f"idx={last_right_idx}"
        )

    for col in output_cols:
        if col in df.columns:
            df.loc[last_right_idx:, col] = 0

    return df


# ============================================================
# 9. Print mappings and statistics
# ============================================================

def print_event_mapping(events, sequence, title):
    print("=" * 140)
    print(title)
    print("-" * 140)
    print(
        f"{'No.':>4} | "
        f"{'idx':>8} | "
        f"{'time_s':>10} | "
        f"{'side':>4} | "
        f"{'source':>8} | "
        f"{'value':>6} | "
        f"{'rule':>36} | "
        f"{'terrain':>10}"
    )
    print("-" * 140)

    for k, event in enumerate(events):
        value = sequence[k]

        print(
            f"{k + 1:>4} | "
            f"{event['idx']:>8} | "
            f"{event['time_s']:>10.4f} | "
            f"{event['side']:>4} | "
            f"{event['label_col']:>8} | "
            f"{event.get('label_value', ''):>6} | "
            f"{event.get('event_rule', ''):>36} | "
            f"{value:>10}"
        )

    print("-" * 140)
    print(f"事件数量: {len(events)}")
    print(f"标签数量: {len(sequence)}")
    print("=" * 140)


def print_label_statistics(df):
    print("=" * 80)
    print("[原始步态标签统计]")

    for col in ["l_label", "r_label"]:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).sort_index())

    print("=" * 80)


def print_terrain_statistics(df):
    print("=" * 80)
    print("[新增地形标签统计]")

    cols = [
        "terrain_x_single_label",
        "terrain_y_single_label",
        "terrain_z_single_label",
        "terrain_x_same_label",
        "terrain_y_same_label",
        "terrain_z_same_label",
    ]

    for col in cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).sort_index())

    print("=" * 80)


# ============================================================
# 10. Process a single file
# ============================================================

def process_one_file(input_file):
    print("\n")
    print("#" * 100)
    print(f"[开始处理] {input_file}")
    print("#" * 100)

    direction = detect_xyz3_direction_from_filename(input_file)

    print(f"[方向检测] 当前文件方向: {direction}")

    df, sheet_name = read_excel_auto_sheet(input_file)

    required_cols = ["l_label", "r_label"]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            "当前表格缺少必要列：\n"
            + "\n".join(missing)
        )

    print_label_statistics(df)

    events = collect_xyz3_terrain_events(df)

    sequences = get_xyz3_sequences(direction)

    print(f"[事件检测] {direction} 共检测到地形切换事件: {len(events)} 个")

    ok = check_event_label_count_match(
        events=events,
        sequences=sequences,
        direction=direction,
    )

    if not ok:
        print("=" * 80)
        print(f"[跳过] {direction} 标签数量和事件数量对不上，不生成新文件。")
        print(f"输入文件: {input_file}")
        print("=" * 80)
        return None

    if PRINT_EVENTS:
        for col, seq in sequences.items():
            print_event_mapping(
                events=events,
                sequence=seq,
                title=f"[{direction} {col} 标签映射]",
            )

    for col, seq in sequences.items():
        df = apply_terrain_sequence(
            df=df,
            events=events,
            terrain_sequence=seq,
            output_col=col,
            initial_label=0,
        )

    if FORCE_ZERO_FROM_LAST_RIGHT_LABEL3:
        df = force_zero_from_last_right_label3(
            df=df,
            output_cols=list(sequences.keys()),
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
    print(f"方向: {direction}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {actual_output_file}")
    print("新增列:")
    print("  terrain_x_single_label")
    print("  terrain_y_single_label")
    print("  terrain_z_single_label")
    print("  terrain_x_same_label")
    print("  terrain_y_same_label")
    print("  terrain_z_same_label")
    print("=" * 80)

    return actual_output_file


# ============================================================
# 11. Main entry point
# ============================================================

def main():
    print("=" * 100)
    print("[XYZ3 地形标签批处理开始]")
    print(f"待处理文件数量: {len(INPUT_FILES)}")
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
            print(f"文件: {input_file}")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print("=" * 80)
            continue

    print("\n")
    print("#" * 100)
    print("[全部处理完成]")
    print(f"成功生成文件数量: {len(output_files)}")

    for f in output_files:
        print(f"  {f}")

    print("#" * 100)


if __name__ == "__main__":
    main()
