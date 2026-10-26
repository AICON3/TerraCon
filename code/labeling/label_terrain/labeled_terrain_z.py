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
        f"Unsupported terrain name: {TERRAIN_ID}\n"
        f"Supported values: {', '.join(SUPPORTED_TERRAINS)}"
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

    Provide a clear error if the file is not a valid XLSX workbook.
    """
    try:
        xls = pd.ExcelFile(input_file)

    except zipfile.BadZipFile:
        raise RuntimeError(
            "The file is not a valid .xlsx workbook; it may be corrupted"
            "or merely renamed with an .xlsx extension.\n"
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
    print(f"Worksheet: {xls.sheet_names}")

    preferred_sheets = ["data", "Data", "DATA"]

    sheet_name = None

    for name in preferred_sheets:
        if name in xls.sheet_names:
            sheet_name = name
            break

    if sheet_name is None:
        sheet_name = xls.sheet_names[0]
        print(f"[Info] data/Data/DATA not found; reading first worksheet: {sheet_name}")
    else:
        print(f"[Info] Using worksheet: {sheet_name}")

    df = pd.read_excel(input_file, sheet_name=sheet_name)

    print(f"[Read OK] Data shape: {df.shape[0]} rows × {df.shape[1]} columns")

    if "time_s" in df.columns:
        print(f"[Time range] {df['time_s'].iloc[0]} s -> {df['time_s'].iloc[-1]} s")

    print("=" * 80)

    return df, sheet_name


# ============================================================
# 5. Determine z2 / z3 / z4 direction
# ============================================================

def detect_z234_direction_from_filename(input_file):
    """
    Determine from the filename:
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
        "Unable to determine direction from filename; ensure it contains:\n"
        "z2_fwd / z2_bwd / z3_fwd / z3_bwd / z4_fwd / z4_bwd\n"
        f"Current file: {input_file}"
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
    Generate output directory and output filename.

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
        print("[Warning] Cannot write the output file; it may be open in Excel/WPS:")
        print(output_file)
        print("[Processing] Trying another output filename.")

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

    Rising-edge condition:
        Current row label == label_value
        and previous row label != label_value
    """
    if label_col not in df.columns:
        raise KeyError(f"Missing column: {label_col}")

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
    Labeling-event logic for Z2/Z3/Z4:

    Initialize to 0.

    Event 1:
        First l_label == 3
        Assign the first label starting here,
        up to just before the second l_label == 3.

    Event 2:
        Second l_label == 3

    Event 3:
        After the second l_label == 3,
        First r_label == 3

    Event 4 and later:
        After the third event,
        l_label == 3 or r_label == 3

    Finally:
        Use the final l_label == 3 event as the point where labels return to zero.
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
            "l_label == 3 has fewer than two start events; cannot apply Z2/Z3/Z4 labeling rules."
        )

    if len(r3_events) < 1:
        raise RuntimeError(
            "r_label == 3 has fewer than one start event; cannot apply Z2/Z3/Z4 labeling rules."
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
            "The second l_label == 3 was found, "
            "but no subsequent r_label == 3 was found; cannot assign the third label."
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
    Return single-side and same-side terrain sequences for the direction.
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

    raise ValueError(f"Unknown direction: {direction}")


# ============================================================
# 11. Strictly validate that label count matches event count
# ============================================================

def check_event_label_count_match(events, sequence, sequence_name):
    """
    Strictly check that event and label counts agree.

    Return False when counts differ.
    """
    event_count = len(events)
    label_count = len(sequence)

    print("=" * 80)
    print(f"[Count validation] {sequence_name}")
    print(f"Event count: {event_count}")
    print(f"Label count: {label_count}")

    if event_count == label_count:
        print("[Validation passed] event count and label count match.")
        print("=" * 80)
        return True

    print("[Validation failed] Event count and label count differ。")
    print("Skip this file without generating an output.")

    if event_count > label_count:
        print(f"Event count exceeds label count by {event_count - label_count}.")
    else:
        print(f"Label count exceeds event count by {label_count - event_count}.")

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
        print(f"[Warning] No events detected; {output_col} fill all with {initial_label}")
        df[output_col] = terrain.astype(int)
        return df

    if len(events) != len(terrain_sequence):
        msg = (
            f"{output_col}: Event count {len(events)} "
            f"and terrain-sequence length {len(terrain_sequence)} do not match."
        )

        if not keep_last:
            raise ValueError(msg)
        else:
            print(f"[Warning] {msg}")
            print("[Processing] The excess portion retains the final terrain_sequence label.")

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
    Starting from the final l_label == 3 start event,
    Force the specified terrain-label column to 0.

    Logic:
        Before the final l_label == 3, keep assigning the preceding label;
        From the final l_label == 3 onward, set terrain = 0.
    """
    if "l_label" not in df.columns:
        raise KeyError("Missing column: l_label")

    if output_col not in df.columns:
        raise KeyError(f"Missing column: {output_col}")

    l3_events = detect_label_value_rising_edges(
        df=df,
        label_col="l_label",
        label_value=3,
        side_name="L",
    )

    if len(l3_events) == 0:
        print(f"[Warning] No l_label == 3 start event found; cannot apply final reset to {output_col}")
        return df

    last_left_idx = int(l3_events[-1]["idx"])

    if "time_s" in df.columns:
        t = df.loc[last_left_idx, "time_s"]
        print(
            f"[Final reset to zero] {output_col}: "
            f"Set labels to zero from the final l_label == 3 event; "
            f"idx={last_left_idx}, time={t:.4f}s"
        )
    else:
        print(
            f"[Final reset to zero] {output_col}: "
            f"Set labels to zero from the final l_label == 3 event; "
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
        print("[Check] event count and label count match.")
    elif len(events) < len(sequence):
        print(f"[Info] Label count exceeds event count by {len(sequence) - len(events)}.")
    else:
        print(f"[Info] Event count exceeds label count by {len(events) - len(sequence)}.")

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
    Check whether L/R label==3 events alternate after the special-rule events.
    """
    later_events = [
        e for e in events
        if e.get("event_rule") == "later_l_or_r_label_3"
    ]

    if len(later_events) < 2:
        return

    print("=" * 80)
    print("[Left/right event alternation: only subsequent events after special rules]")

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
        print("[Validation passed] Subsequent L/R label==3 events alternate.")
    else:
        print(f"[Warning] Found {len(warnings)} consecutive same-side events:")

        for i, prev_event, curr_event in warnings:
            print(
                f"  Later event No.{i} and No.{i + 1}: "
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

    print(f"[Direction detection] Current direction: {direction}")

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
            + "\n\nPlease verify that the input workbook already contains gait labels."
        )

    print_label_statistics(df)

    events = collect_z234_terrain_events(df)

    if len(events) == 0:
        raise RuntimeError("No events available for assigning terrain labels were detected.")

    single_sequence, same_sequence = get_z234_sequences(direction)

    print(f"[Event detection] {direction} terrain transition events detected: {len(events)} events")
    print(f"[Label counts] single-side labels: {len(single_sequence)}")
    print(f"[Label counts] same-side labels: {len(same_sequence)}")

    # ========================================================
    # Strict check: label count must match event count
    # Do not create an output file if counts differ
    # ========================================================

    single_ok = check_event_label_count_match(
        events=events,
        sequence=single_sequence,
        sequence_name=f"{direction} single-side terrain label",
    )

    same_ok = check_event_label_count_match(
        events=events,
        sequence=same_sequence,
        sequence_name=f"{direction} same-side terrain label",
    )

    if not single_ok or not same_ok:
        print("=" * 80)
        print(f"[Skipped] {direction} label count and event count differ; no output generated.")
        print(f"Input file: {input_file}")
        print("=" * 80)
        return None

    check_left_right_alternation(events)

    if PRINT_EVENTS:
        print_event_mapping(
            events,
            single_sequence,
            title=f"[{direction} single-side terrain label mapping]",
        )

        print_event_mapping(
            events,
            same_sequence,
            title=f"[{direction} same-side terrain label mapping]",
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
    print("[Completed]")
    print(f"Direction: {direction}")
    print(f"Input file: {input_file}")
    print(f"Output file: {actual_output_file}")
    print("Added columns:")
    print("  terrain_single_label  single-side terrain label")
    print("  terrain_same_label    same-side terrain label")
    print("=" * 80)

    return actual_output_file


# ============================================================
# 18. Main entry point
# ============================================================

def main():
    print("=" * 100)
    print("[Z2/Z3/Z4 terrain batch labeling started]")
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
            print("[Error] Processing failed for this file; skipping; continue processing subsequent files.")
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
