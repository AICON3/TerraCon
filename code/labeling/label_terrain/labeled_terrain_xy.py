import os
import re
import zipfile
import numpy as np
import pandas as pd


# ============================================================
# 0. Configuration
# ============================================================

# Only this section needs to be changed
SUBJECT_ID = "sub_041"

# Project root directory
PROJECT_ROOT = r"E:\data"



TERRAIN_NAMES = ["xy5"]

# ============================================================
# Select whether to process the fwd or bwd direction here
# ============================================================

# Process fwd only
# DIRECTIONS = ["fwd"]

# Process bwd only
# DIRECTIONS = ["bwd"]

# Process both fwd and bwd
DIRECTIONS = ["fwd", "bwd"]

# Automatically generate the input directory
INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    f"{SUBJECT_ID}_label",
)

# Automatically generate input file paths
INPUT_FILES = [
    os.path.join(
        INPUT_DIR,
        f"{SUBJECT_ID}_{terrain_name}_{direction}_aligned_labeled.xlsx",
    )
    for terrain_name in TERRAIN_NAMES
    for direction in DIRECTIONS
]

# Whether to print detailed event mappings
PRINT_EVENTS = False

# Whether to force labels to 0 after the final event
FORCE_ZERO_FROM_LAST_RIGHT_LABEL3 = True



# ============================================================
# 1. XY1–XY6 label sequences
# ============================================================

LABEL_TEXT = r"""
xy1_fwd
单侧
x:3, -18, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10,  5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5,5, -5, -10, 10, 5, -5, 5,-5, -13,28,0
y:26.5,91.5,150,140,130,170,160,150,140,130,170,140,130,140,130,170、140,110,120,130,170,140,110,150,160,170,140,110,150,140,150,140,110,150,140,130,125,50,0
同侧
x:3,21,26,31,36,41,31,21,26,31,36,41,31,21,26,31,36,41,31,21,26,31,36,41,31,21,26,31,36,41,31,21,26,31,36,41,28,0,0
y:26.5,65,85,55,75,95,65,85,55,75,95,45,85,55,75,95,45,65,55,75,95,45,65,85,75,95,45,65,85,55,95,45,65,85,55,75,50,0,0

xy1_bwd
单侧
x:28, -13, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, -18, 3,0
y:50, 125, 130, 140, 150, 110, 140, 150, 140, 150, 110, 140, 170, 160, 150, 110, 140, 170, 130, 120, 110, 140, 170, 130, 140, 130, 140, 170, 130, 140, 150, 160, 170, 130, 140, 150, 91.5, 26.5,0
同侧
x:28, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21,3,0,0
y:50, 75, 55, 85, 65, 45, 95, 55, 85, 65, 45, 95, 75, 85, 65, 45, 95, 75, 55, 65, 45, 95, 75, 55, 85, 45, 95, 75, 55, 85, 65, 95, 75, 55, 85, 65, 26.5,0,0

xy2_fwd
单侧
x:3, -18, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -13,28,0
y:26.5, 91.5, 150, 140, 130, 170, 160, 150, 140, 130, 170, 140, 100, 130, 170, 140, 110, 150, 160, 170, 140, 110, 150, 140, 140, 140, 130, 170, 140, 110, 160, 140, 110, 150, 140, 130, 125,50,0
同侧
x:3, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 28,0,0
y:26.5, 65, 85, 55, 75, 95, 65, 85, 55, 75, 95, 45, 55, 75, 95, 45, 65, 85, 75, 95, 45, 65, 85, 55, 85, 55, 75, 95, 45, 65, 95, 45, 65, 85, 55, 75, 50,0,0

xy2_bwd
单侧
x:28,-13, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, -18, 3,0
y:50,125, 130, 140, 150, 110, 140, 160, 110, 140, 170, 130, 140, 140, 140, 150, 110, 140, 170, 160, 150, 110, 140, 170, 130, 100, 140, 170, 130, 140, 150, 160, 170, 130, 140, 150, 91.5, 26.5,0
同侧
x:28, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 3,0,0
y:50, 75, 55, 85, 65, 45, 95, 65, 45, 95, 75, 55, 85, 55, 85, 65, 45, 95, 75, 85, 65, 45, 95, 75, 55, 45, 95, 75, 55, 85, 65, 95, 75, 55, 85, 65, 26.5,0,0

xy3_fwd
单侧
x:3, -18, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -13,28,0
y:26.5, 91.5, 150, 140, 130, 170, 180, 140, 130, 170, 140, 110, 120, 130, 170, 140, 110, 150, 150, 150, 140, 130, 170, 140, 120, 170, 140, 110, 150, 140, 150, 140, 110, 150, 140, 130, 125,50,0
同侧
x:3, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 28,0,0
y:26.5, 65, 85, 55, 75, 95, 85, 55, 75, 95, 45, 65, 55, 75, 95, 45, 65, 85, 65, 85, 55, 75, 95, 45, 75, 95, 45, 65, 85, 55, 95, 45, 65, 85, 55, 75, 50,0,0

xy3_bwd
单侧
x:28,-13, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, -18, 3,0
y:50,125, 130, 140, 150, 110, 140, 150, 140, 150, 110, 140, 170, 120, 140, 170, 130, 140, 150, 150, 150, 110, 140, 170, 130, 120, 110, 140, 170, 130, 140, 180, 170, 130, 140, 150, 91.5, 26.5,0
同侧
x:28, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 3,0,0
y:50, 75, 55, 85, 65, 45, 95, 55, 85, 65, 45, 95, 75, 45, 95, 75, 55, 85, 65, 85, 65, 45, 95, 75, 55, 65, 45, 95, 75, 55, 85, 95, 75, 55, 85, 65, 26.5,0,0

xy4_fwd
单侧
x:3, -18, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -13,28,0
y:26.5, 91.5, 150, 140, 130, 170, 170, 170, 140, 110, 150, 140, 110, 130, 170, 140, 110, 150, 150, 150, 140, 130, 170, 140, 130, 140, 130, 170, 140, 110, 160, 140, 110, 150, 140, 130, 125,50,0
同侧
x:3, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 28,0,0
y:26.5, 65, 85, 55, 75, 95, 75, 95, 45, 65, 85, 55, 55, 75, 95, 45, 65, 85, 65, 85, 55, 75, 95, 45, 85, 55, 75, 95, 45, 65, 95, 45, 65, 85, 55, 75, 50,0,0

xy4_bwd
单侧
x:28,-13, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, -18, 3,0
y:50,125, 130, 140, 150, 110, 140, 160, 110, 140, 170, 130, 140, 130, 140, 170, 130, 140, 150, 150, 150, 110, 140, 170, 130, 110, 140, 150, 110, 140, 170, 170, 170, 130, 140, 150, 91.5, 26.5,0
同侧
x:28, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 3,0,0
y:50, 75, 55, 85, 65, 45, 95, 65, 45, 95, 75, 55, 85, 45, 95, 75, 55, 85, 65, 85, 65, 45, 95, 75, 55, 55, 85, 65, 45, 95, 75, 95, 75, 55, 85, 65, 26.5,0,0

xy5_fwd
单侧
x:3, -18, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -13,28,0
y:26.5, 91.5, 150, 140, 130, 170, 150, 130, 170, 140, 110, 150, 170, 140, 130, 170, 140, 110, 140, 170, 140, 110, 150, 140, 120, 150, 140, 130, 170, 140, 140, 140, 110, 150, 140, 130, 125,50,0
同侧
x:3, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 28,0,0
y:26.5, 65, 85, 55, 75, 95, 55, 75, 95, 45, 65, 85, 85, 55, 75, 95, 45, 65, 75, 95, 45, 65, 85, 55, 65, 85, 55, 75, 95, 45, 95, 45, 65, 85, 55, 75, 50,0,0

xy5_bwd
单侧
x:28,-13, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, -18, 3,0
y:50,125, 130, 140, 150, 110, 140, 140, 140, 170, 130, 140, 150, 120, 140, 150, 110, 140, 170, 140, 110, 140, 170, 130, 140, 170, 150, 110, 140, 170, 130, 150, 170, 130, 140, 150, 91.5, 26.5,0
同侧
x:28, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 3,0,0
y:50, 75, 55, 85, 65, 45, 95, 45, 95, 75, 55, 85, 65, 55, 85, 65, 45, 95, 75, 65, 45, 95, 75, 55, 85, 85, 65, 45, 95, 75, 55, 95, 75, 55, 85, 65, 26.5,0,0

xy6_fwd
单侧
x:3, -18, 5, -5, 5, -5, 3, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, -18, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -10, 10, 5, -5, 5, -5, -13,28,0
y:26.5, 91.5, 150, 140, 130, 170, 130, 150, 140, 130, 170, 140, 170, 140, 130, 170, 140, 110, 150, 130, 170, 140, 110, 150, 120, 170, 140, 110, 150, 140, 150, 140, 110, 150, 140, 130, 125,50,0
同侧
x:3, 21, 26, 31, 36, 41, 3, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 31, 21, 26, 31, 36, 41, 28，0,0
y:26.5, 65, 85, 55, 75, 95, 65, 85, 55, 75, 95, 45, 85, 55, 75, 95, 45, 65, 55, 75, 95, 45, 65, 85, 75, 95, 45, 65, 85, 55, 95, 45, 65, 85, 55, 75, 50,0,0

xy6_bwd
单侧
x:28，-13, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, -18, -10, -5, 5, -5, 5, 10, -10, -5, 5, -5, 5, 10, 3, -5, 5, -5, 5, -18, 3,0
y:50，125, 130, 140, 150, 110, 140, 150, 140, 150, 110, 140, 170, 120, 150, 110, 140, 170, 130, 150, 110, 140, 170, 130, 140, 170, 140, 170, 130, 140, 150, 130, 140, 170, 130, 150, 91.5, 26.5,0
同侧
x:,28, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 31, 41, 36, 31, 26, 21, 3, 41, 36, 31, 26, 21, 3，0,0
y:50, 75, 55, 85, 65, 45, 95, 55, 85, 65, 45, 95, 75, 85, 65, 45, 95, 75, 55, 65, 45, 95, 75, 55, 85, 45, 95, 75, 55, 85, 65, 45, 95, 75, 85, 65, 26.5,0,0
"""



# ============================================================
# 2. Parse label sequences
# ============================================================

def parse_number_list(s):
    """
    Extract numeric values from text.
    Supported:
        ASCII commas
        Chinese commas
        enumeration commas
        decimal values
        negative values
    """
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)

    values = []

    for n in nums:
        v = float(n)

        if v.is_integer():
            v = int(v)

        values.append(v)

    return values


def build_xy_label_dict():
    """
    Parse LABEL_TEXT to generate LABELS.
    """
    labels = {}

    current_direction = None
    current_type = None

    valid_dirs = {
        "xy1_fwd", "xy1_bwd",
        "xy2_fwd", "xy2_bwd",
        "xy3_fwd", "xy3_bwd",
        "xy4_fwd", "xy4_bwd",
        "xy5_fwd", "xy5_bwd",
        "xy6_fwd", "xy6_bwd",
    }

    for raw_line in LABEL_TEXT.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        low = line.lower()

        if low in valid_dirs:
            current_direction = low
            labels[current_direction] = {
                "single": {},
                "same": {},
            }
            current_type = None
            continue

        if line == "单侧":
            current_type = "single"
            continue

        if line == "同侧":
            current_type = "same"
            continue

        if current_direction is None or current_type is None:
            continue

        if low.startswith("x"):
            labels[current_direction][current_type]["x"] = parse_number_list(line)
            continue

        if low.startswith("y"):
            labels[current_direction][current_type]["y"] = parse_number_list(line)
            continue

    return labels


LABELS = build_xy_label_dict()


# ============================================================
# 3. Automatic Excel loading
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
# 4. Determine direction from the filename
# ============================================================

def detect_xy_direction_from_filename(input_file):
    """
    Determine from filename: xy1_fwd ~ xy6_bwd。
    """
    basename = os.path.basename(input_file).lower()

    patterns = [
        "xy1_fwd", "xy1_bwd",
        "xy2_fwd", "xy2_bwd",
        "xy3_fwd", "xy3_bwd",
        "xy4_fwd", "xy4_bwd",
        "xy5_fwd", "xy5_bwd",
        "xy6_fwd", "xy6_bwd",
    ]

    for p in patterns:
        if p in basename:
            return p

    raise ValueError(
        "Cannot determine direction from filename; ensure it contains xy1_fwd ~ xy6_bwd。\n"
        f"Current file: {input_file}"
    )


# ============================================================
# 5. Output paths
# ============================================================

def extract_subject_id(input_file):
    text = input_file.replace("\\", "/")
    match = re.search(r"(sub_\d+)", text, flags=re.IGNORECASE)

    if match:
        return match.group(1)

    basename = os.path.basename(input_file)
    stem = os.path.splitext(basename)[0]

    return stem.split("_")[0]


def build_output_file(input_file):
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
    If the target file is open, save using _v2, _v3, etc.
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

        raise PermissionError("Could not save using alternative filenames. Close Excel/WPS and try again.")


# ============================================================
# 6. Event detection
# ============================================================

def detect_label_value_rising_edges(df, label_col, label_value, side_name):
    """
    Detect the start of a label == label_value run.
    A consecutive run of label == 3 counts as one event.
    """
    if label_col not in df.columns:
        raise KeyError(f"Missing column: {label_col}")

    label = pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy()

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


def collect_xy_terrain_events(df):
    """
    Event-labeling rules for XY1–XY6:

    Event 1:
        First l_label == 3

    Event 2:
        Second l_label == 3

    Event 3:
        First r_label == 3 after the second l_label == 3

    Event 4 and later:
        After the third event: l_label == 3 or r_label == 3

    Finally:
        Force labels to zero from the final r_label == 3 event.
    """
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
        raise RuntimeError("l_label == 3 has fewer than two start events; cannot apply XY labeling rules.")

    if len(r3_events) < 1:
        raise RuntimeError("r_label == 3 has fewer than one start event; cannot apply XY labeling rules.")

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
        raise RuntimeError("No r_label == 3 was found after the second l_label == 3.")

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
    later_events = sorted(later_events, key=lambda e: e["idx"])

    for e in later_events:
        e["event_rule"] = "later_l_or_r_label_3"

    events.extend(later_events)

    return events, l3_events, r3_events


# ============================================================
# 7. Retrieve labels and validate counts
# ============================================================

def get_xy_sequences(direction):
    """
    Get provided label sequences for the selected direction.
    """
    if direction not in LABELS:
        raise ValueError(f"No label sequence found for direction {direction}. Check that LABEL_TEXT includes this direction.")

    item = LABELS[direction]

    required = [
        ("single", "x"),
        ("single", "y"),
        ("same", "x"),
        ("same", "y"),
    ]

    for terrain_type, axis in required:
        if terrain_type not in item or axis not in item[terrain_type]:
            raise ValueError(
                f"{direction} missing labels: {terrain_type} {axis}。"
            )

    return {
        "terrain_x_single_label": item["single"]["x"],
        "terrain_y_single_label": item["single"]["y"],
        "terrain_x_same_label": item["same"]["x"],
        "terrain_y_same_label": item["same"]["y"],
    }


def get_sequence_lengths(sequences):
    """
    Get the provided label count.
    """
    return {
        col: len(seq)
        for col, seq in sequences.items()
    }


def check_event_label_count_match(events, sequences):
    """
    Check that event count equals the provided label count.
    """
    event_count = len(events)

    for col, seq in sequences.items():
        if len(seq) != event_count:
            return False

    return True


def print_count_summary(direction, sequences, events, l3_events, r3_events):
    """
    Print a concise count summary:
        Original label count = number of values supplied in LABEL_TEXT
        Event label count = number of labeling events detected from l_label/r_label
    """
    seq_lens = get_sequence_lengths(sequences)

    print("=" * 100)
    print(f"[Count results] {direction}")

    print(
        f"Original / provided label count: "
        f"x_single={seq_lens['terrain_x_single_label']}, "
        f"y_single={seq_lens['terrain_y_single_label']}, "
        f"x_same={seq_lens['terrain_x_same_label']}, "
        f"y_same={seq_lens['terrain_y_same_label']}"
    )

    print(
        f"Detected labeling-event count: "
        f"{len(events)}"
    )

    print(
        f"Original label == 3 start-event count: "
        f"l_label==3 events={len(l3_events)}, "
        f"r_label==3 events={len(r3_events)}"
    )

    if check_event_label_count_match(events, sequences):
        print("Count validation: passed")
    else:
        print("Count validation: failed")

    print("=" * 100)


# ============================================================
# 8. Assign labels
# ============================================================

def apply_terrain_sequence(df, events, terrain_sequence, output_col, initial_label=0):
    """
    Set labels to 0 before the first event.
    Assign terrain_sequence[k] from event k until event k+1.
    """
    if len(events) != len(terrain_sequence):
        raise ValueError(
            f"{output_col}: Event count {len(events)} and label count {len(terrain_sequence)} do not match."
        )

    n = len(df)

    terrain = np.full(
        n,
        initial_label,
        dtype=float,
    )

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
    Force labels to 0 from the final r_label == 3 start event.
    """
    r3_events = detect_label_value_rising_edges(
        df=df,
        label_col="r_label",
        label_value=3,
        side_name="R",
    )

    if len(r3_events) == 0:
        return df

    last_right_idx = int(r3_events[-1]["idx"])

    for col in output_cols:
        if col in df.columns:
            df.loc[last_right_idx:, col] = 0

    return df


# ============================================================
# 9. Optional event-mapping output
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
        f"{'rule':>34} | "
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
            f"{event.get('event_rule', ''):>34} | "
            f"{value:>10}"
        )

    print("-" * 130)
    print(f"Event count: {len(events)}")
    print(f"Label count: {len(sequence)}")
    print("=" * 130)


# ============================================================
# 10. Process a single file
# ============================================================

def process_one_file(input_file):
    direction = detect_xy_direction_from_filename(input_file)

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

    # Detect events internally without printing the search process
    events, l3_events, r3_events = collect_xy_terrain_events(df)

    # Retrieve the specified label sequences
    sequences = get_xy_sequences(direction)

    # Print count results
    print_count_summary(
        direction=direction,
        sequences=sequences,
        events=events,
        l3_events=l3_events,
        r3_events=r3_events,
    )

    # Validate counts
    ok = check_event_label_count_match(
        events=events,
        sequences=sequences,
    )

    if not ok:
        seq_lens = get_sequence_lengths(sequences)

        raise RuntimeError(
            f"{direction} Event count and the provided label count do not match.\n"
            f"Event count: {len(events)}\n"
            f"Original / provided label count: {seq_lens}"
        )

    # Optionally print event mappings
    if PRINT_EVENTS:
        for col, seq in sequences.items():
            print_event_mapping(
                events=events,
                sequence=seq,
                title=f"[{direction} {col} label mapping]",
            )

    # Assign labels
    for col, seq in sequences.items():
        df = apply_terrain_sequence(
            df=df,
            events=events,
            terrain_sequence=seq,
            output_col=col,
            initial_label=0,
        )

    # Set labels to 0 from the final r_label == 3 event onward
    if FORCE_ZERO_FROM_LAST_RIGHT_LABEL3:
        df = force_zero_from_last_right_label3(
            df=df,
            output_cols=list(sequences.keys()),
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
        "event_count": len(events),
        "l3_event_count": len(l3_events),
        "r3_event_count": len(r3_events),
        "sequence_lengths": get_sequence_lengths(sequences),
        "row_count": len(df),
    }


# ============================================================
# 11. Main entry point
# ============================================================

def main():
    print("=" * 100)
    print("[XY1~XY6 terrain label batch processing started]")
    print(f"Subject: {SUBJECT_ID}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Files to process: {len(INPUT_FILES)}")
    print("=" * 100)

    results = []
    failed_files = []

    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            failed_files.append(
                {
                    "file": input_file,
                    "reason": "File not found",
                }
            )

            print("=" * 100)
            print("[Skipped] File not found")
            print(input_file)
            print("=" * 100)

            continue

        try:
            result = process_one_file(input_file)
            results.append(result)

            print(
                f"[Completed] {result['direction']} | "
                f"Rows: {result['row_count']} | "
                f"Event count: {result['event_count']} | "
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
            print("[Failed] Processing failed for this file; skipping")
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
        print("\n[Successfully generated files]")
        for r in results:
            seq_lens = r["sequence_lengths"]

            print(
                f"{r['direction']} | "
                f"Original label count: "
                f"x_single={seq_lens['terrain_x_single_label']}, "
                f"y_single={seq_lens['terrain_y_single_label']}, "
                f"x_same={seq_lens['terrain_x_same_label']}, "
                f"y_same={seq_lens['terrain_y_same_label']} | "
                f"Labeled event count={r['event_count']} | "
                f"Output={r['output_file']}"
            )

    if failed_files:
        print("\n[Failed files]")
        for item in failed_files:
            print(f"File: {item['file']}")
            print(f"Reason: {item['reason']}")
            print("-" * 100)


if __name__ == "__main__":
    main()
