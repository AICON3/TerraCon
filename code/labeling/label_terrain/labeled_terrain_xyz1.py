import os
import re
import zipfile
import numpy as np
import pandas as pd


# ============================================================
# 0. Configuration
# ============================================================

# Only this section needs to be changed
SUBJECT_ID = "sub_040"

# Project root directory
PROJECT_ROOT = r"E:\data"

# Terrain direction to process
TERRAIN_NAME = "xyz1"

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
# 1. XYZ1 label sequence
# ============================================================

XYZ1_LABELS = {
    "xyz1_fwd": {
        "single": {
            "x": [16, -20, 0, 20, 30, 40, 50, 30, -10, -30, -40, -50, -18, 38,0],
            "y": [65, 140, 130, 140, 130, 140, 160, 140, 130, 140, 130, 140, 158, 63,0],
            "z": [0, 3, 0, 4, 0, 4, 3, 5, 4, 4, 4, -20, -11, 0,0],
        },
        "same": {
            "x": [16, 36, 36, 16, 46, 6, 56, 26, 16, 46, 6, 56, 38, 0,0],
            "y": [65, 75, 55, 85, 45, 95, 65, 75, 55, 85, 45, 95, 63, 0,0],
            "z": [0, 3, -3, 7, -7, 11, -8, 13, -9, 13, -9, -11, 0, 0,0],
        },
    },
    "xyz1_bwd": {
        "single": {
            "x": [38, -18, -50, -40, -30, -10, 30, 50, 40, 30, 20, 0, -20, 16,0],
            "y": [63, 158, 140, 130, 140, 130, 140, 160, 140, 130, 140, 130, 140, 65,0],
            "z": [0, 11, 20, -4, -4, -4, -5, -3, -4, 0, -4, 0, -3, 0,0],
        },
        "same": {
            "x": [38, 56, 6, 46, 16, 26, 56, 6, 46, 16, 36, 36, 16, 0,0],
            "y": [63, 95, 45, 85, 55, 75, 65, 95, 45, 85, 55, 75, 65, 0,0],
            "z": [0, 11, 9, -13, 9, -13, 8, -11, 7, -7, 3, -3, 0, 0,0],
        },
    },
}


# ============================================================
# 2. Automatic Excel loading
# ============================================================

def read_excel_auto_sheet(input_file):
    """
    Automatically load the Excel workbook.

    Prefer the data/Data/DATA worksheet.
    Otherwise read the first worksheet.
    """
    try:
        xls = pd.ExcelFile(input_file)

    except zipfile.BadZipFile:
        raise RuntimeError(
            "The file is not a valid .xlsx workbook; it may be corrupt or incorrectly renamed.\n"
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

    print("=" * 80)
    print("[Excel worksheet inspection]")
    print(f"File: {input_file}")
    print(f"Worksheet: {xls.sheet_names}")

    preferred_sheets = ["data", "Data", "DATA"]

    sheet_name = None

    for name in preferred_sheets:
        if name in xls.sheet_names:
            sheet_name = name
            break

    if sheet_name is None:
        sheet_name = xls.sheet_names[0]
        print(f"[Info] data/Data/DATA not found; reading the first worksheet: {sheet_name}")
    else:
        print(f"[Info] Using worksheet: {sheet_name}")

    df = pd.read_excel(input_file, sheet_name=sheet_name)

    print(f"[Read OK] Data shape: {df.shape[0]} rows × {df.shape[1]} columns")

    if "time_s" in df.columns:
        print(f"[Time range] {df['time_s'].iloc[0]} s -> {df['time_s'].iloc[-1]} s")

    print("=" * 80)

    return df, sheet_name


# ============================================================
# 3. Determine direction from the filename
# ============================================================

def detect_xyz1_direction_from_filename(input_file):
    """
    Determine xyz1_fwd / xyz1_bwd from the filename.
    """
    basename = os.path.basename(input_file).lower()

    if "xyz1_fwd" in basename:
        return "xyz1_fwd"

    if "xyz1_bwd" in basename:
        return "xyz1_bwd"

    raise ValueError(
        "Cannot determine direction from the filename; ensure it contains xyz1_fwd or xyz1_bwd。\n"
        f"Current file: {input_file}"
    )


# ============================================================
# 4. Output paths
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


def save_excel_safely(df, output_file, sheet_name="data"):
    """
    Save the Excel workbook safely.
    If the target is open, automatically save to _v2, _v3, etc.
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
        print("[Warning] Cannot write output; it may be open in Excel/WPS:")
        print(output_file)
        print("[Processing] Trying an alternative output filename.")

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
            "Could not save using alternative filenames. Close Excel/WPS and try again."
        )


# ============================================================
# 5. Event detection
# ============================================================

def detect_label_value_rising_edges(df, label_col, label_value, side_name):
    """
    Detect the start of label_value in label_col.

    rising edge：
        Current row == label_value
        and previous row != label_value
    """
    if label_col not in df.columns:
        raise KeyError(f"Missing column: {label_col}")

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


def collect_xyz1_terrain_events(df):
    """
    xyz1 event rules:

    Initialize with 0.

    Event 1:
        First l_label == 3

    Event 2:
        Second l_label == 3

    Event 3:
        First r_label == 3 following the second l_label == 3

    Event 4 and later:
        After the third event,
        l_label == 3 or r_label == 3

    Finally:
        Force labels to zero after the final r_label == 3 start event.
    """
    if "l_label" not in df.columns:
        raise KeyError("Missing column: l_label")

    if "r_label" not in df.columns:
        raise KeyError("Missing column: r_label")

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
            "l_label == 3 has fewer than two start events, so cannot apply xyz1 labeling rules."
        )

    if len(r3_events) < 1:
        raise RuntimeError(
            "r_label == 3 has fewer than one start event, so cannot apply xyz1 labeling rules."
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
            "No r_label == 3 found after the second l_label == 3."
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

def get_xyz1_sequences(direction):
    """
    Return the six label sequences to apply.
    """
    if direction not in XYZ1_LABELS:
        raise ValueError(f"No labels found for direction {direction}.")

    item = XYZ1_LABELS[direction]

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
    Check that event counts match the length of each label sequence.
    """
    event_count = len(events)

    print("=" * 80)
    print(f"[Count validation] {direction}")
    print(f"Event count: {event_count}")

    all_ok = True

    for col, seq in sequences.items():
        label_count = len(seq)
        print(f"{col}: Label count = {label_count}")

        if label_count != event_count:
            all_ok = False
            print(
                f"[Mismatch] {col}: "
                f"Event count {event_count}, label count {label_count}"
            )

    if all_ok:
        print("[Validation passed] All label counts match event counts.")
    else:
        print("[Validation failed] Counts differ; skip this file without generating output.")

    print("=" * 80)

    return all_ok


# ============================================================
# 8. Assign labels
# ============================================================

def apply_terrain_sequence(df, events, terrain_sequence, output_col, initial_label=0):
    """
    Assign terrain labels from the event sequence.

    Logic:
        Before the first event:
            initial_label

        From event k to immediately before event k+1:
            terrain_sequence[k]

        After the final event:
            terrain_sequence[-1]
    """
    if len(events) != len(terrain_sequence):
        raise ValueError(
            f"{output_col}: Event count {len(events)} "
            f"and label count {len(terrain_sequence)} do not match."
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
    Starting at the final r_label == 3 start event,
    force all output columns to 0.
    """
    r3_events = detect_label_value_rising_edges(
        df=df,
        label_col="r_label",
        label_value=3,
        side_name="R",
    )

    if len(r3_events) == 0:
        print("[Warning] No r_label == 3 start event was found; cannot apply the final reset.")
        return df

    last_right_idx = int(r3_events[-1]["idx"])

    if "time_s" in df.columns:
        t = df.loc[last_right_idx, "time_s"]
        print(
            f"[Final reset to zero] Set labels to 0 from the final r_label == 3 event; "
            f"idx={last_right_idx}, time={t:.4f}s"
        )
    else:
        print(
            f"[Final reset to zero] Set labels to 0 from the final r_label == 3 event; "
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
    print(f"Event count: {len(events)}")
    print(f"Label count: {len(sequence)}")
    print("=" * 140)


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
    print(f"[Processing] {input_file}")
    print("#" * 100)

    direction = detect_xyz1_direction_from_filename(input_file)

    print(f"[Direction detection] Current direction: {direction}")

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
        )

    print_label_statistics(df)

    events = collect_xyz1_terrain_events(df)

    sequences = get_xyz1_sequences(direction)

    print(f"[Event detection] {direction} detected terrain transition events: {len(events)}")

    ok = check_event_label_count_match(
        events=events,
        sequences=sequences,
        direction=direction,
    )

    if not ok:
        print("=" * 80)
        print(f"[Skipped] {direction} label count differs from event count; no output generated.")
        print(f"Input file: {input_file}")
        print("=" * 80)
        return None

    if PRINT_EVENTS:
        for col, seq in sequences.items():
            print_event_mapping(
                events=events,
                sequence=seq,
                title=f"[{direction} {col} label mapping]",
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
    print("[Completed]")
    print(f"Direction: {direction}")
    print(f"Input file: {input_file}")
    print(f"Output file: {actual_output_file}")
    print("New columns:")
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
    print("[XYZ1 terrain batch labeling started]")
    print(f"Files to process: {len(INPUT_FILES)}")
    print("=" * 100)

    output_files = []

    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            print("=" * 80)
            print("[Warning] File not found; skipping:")
            print(input_file)
            print("=" * 80)
            continue

        try:
            actual_output_file = process_one_file(input_file)

            if actual_output_file is not None:
                output_files.append(actual_output_file)

        except Exception as e:
            print("=" * 80)
            print("[Error] Processing failed for this file; skipping and continuing.")
            print(f"File: {input_file}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error: {e}")
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
