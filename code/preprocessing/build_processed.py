#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reorganize source Excel workbooks in label_3 into a unified processed dataset.

Target structure:
processed/
  sub_001/
    data/
      sub_001_<task>_data.xlsx
    labels/
      label_gait/
        sub_001_<task>_label_gait.xlsx
      label_terrain_ipsilateral/
        sub_001_<task>_label_terrain_ipsilateral.xlsx
      label_terrain_contralateral/
        sub_001_<task>_label_terrain_contralateral.xlsx
    figures/

Core rules:
1. Do not drop or truncate rows.
2. time_s = timestamp minus the first timestamp of the current task.
3. Copy the data time_s into all three label types to preserve row alignment.
4. Preserve fixed data column order.
5. Recursively scan all XLSX files under source-root.
6. Do not generate terrain labels for walk/stair; other tasks use task axes and existing source columns.
7. Terrain semantics: single=ipsilateral, same=contralateral; double=contralateral for x-only tasks.
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


DEFAULT_SOURCE_ROOT = Path(
    "/mnt/newdisk/Trigait/Trigait数据库_7.20目录示范_20260721/label_3"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/newdisk/Trigait/Trigait数据库_7.20目录示范_20260721/processed"
)


TASKS = [
    "stair_bwd", "stair_fwd",
    "walk_bwd", "walk_fwd",
    "x_bwd", "x_fwd",
    "xy1_bwd", "xy1_fwd",
    "xy2_bwd", "xy2_fwd",
    "xy3_bwd", "xy3_fwd",
    "xy4_bwd", "xy4_fwd",
    "xy5_bwd", "xy5_fwd",
    "xy6_bwd", "xy6_fwd",
    "xyz1_bwd", "xyz1_fwd",
    "xyz2_bwd", "xyz2_fwd",
    "xyz3_bwd", "xyz3_fwd",
    "y_bwd", "y_fwd",
    "z1_bwd", "z1_fwd",
    "z2_bwd", "z2_fwd",
    "z3_bwd", "z3_fwd",
    "z4_bwd", "z4_fwd",
]


DATA_COLUMNS = [
    "timestamp",
    "time_s",

    "pelvis_accel_x_g",
    "pelvis_accel_y_g",
    "pelvis_gyro_x_deg_s",
    "pelvis_gyro_y_deg_s",
    "pelvis_angle_x_deg",
    "pelvis_angle_y_deg",

    "l_thigh_accel_x_g",
    "l_thigh_accel_y_g",
    "l_thigh_gyro_x_deg_s",
    "l_thigh_gyro_y_deg_s",
    "l_thigh_angle_x_deg",
    "l_thigh_angle_y_deg",

    "l_shank_accel_x_g",
    "l_shank_accel_y_g",
    "l_shank_gyro_x_deg_s",
    "l_shank_gyro_y_deg_s",
    "l_shank_angle_x_deg",
    "l_shank_angle_y_deg",

    "l_foot_accel_x_g",
    "l_foot_accel_y_g",
    "l_foot_gyro_x_deg_s",
    "l_foot_gyro_y_deg_s",
    "l_foot_angle_x_deg",
    "l_foot_angle_y_deg",

    "r_thigh_accel_x_g",
    "r_thigh_accel_y_g",
    "r_thigh_gyro_x_deg_s",
    "r_thigh_gyro_y_deg_s",
    "r_thigh_angle_x_deg",
    "r_thigh_angle_y_deg",

    "r_shank_accel_x_g",
    "r_shank_accel_y_g",
    "r_shank_gyro_x_deg_s",
    "r_shank_gyro_y_deg_s",
    "r_shank_angle_x_deg",
    "r_shank_angle_y_deg",

    "r_foot_accel_x_g",
    "r_foot_accel_y_g",
    "r_foot_gyro_x_deg_s",
    "r_foot_gyro_y_deg_s",
    "r_foot_angle_x_deg",
    "r_foot_angle_y_deg",
]


GAIT_COLUMNS = [
    "time_s",
    "l_label_gait",
    "r_label_gait",
]

IPSI_COLUMNS = [
    "time_s",
    "terrain_ipsilateral_x_label",
    "terrain_ipsilateral_y_label",
    "terrain_ipsilateral_z_label",
]

CONTRA_COLUMNS = [
    "time_s",
    "terrain_contralateral_x_label",
    "terrain_contralateral_y_label",
    "terrain_contralateral_z_label",
]


# Common legacy column names -> new column names
ALIASES = {
    "timestamp": [
        "timestamp", "Timestamp", "time_stamp", "TimeStamp",
        "unix_timestamp", "time"
    ],

    "l_label_gait": [
        "l_label_gait", "l_label", "left_label", "label_l", "left_gait_label"
    ],
    "r_label_gait": [
        "r_label_gait", "r_label", "right_label", "label_r", "right_gait_label"
    ],

    "terrain_ipsilateral_x_label": [
        "terrain_ipsilateral_x_label",
        "terrain_x_single_label",
        "ipsilateral_x_label",
        "terrain_ipsi_x_label",
        "ipsi_x_label",
    ],
    "terrain_ipsilateral_y_label": [
        "terrain_ipsilateral_y_label",
        "terrain_y_single_label",
        "ipsilateral_y_label",
        "terrain_ipsi_y_label",
        "ipsi_y_label",
    ],
    "terrain_ipsilateral_z_label": [
        "terrain_ipsilateral_z_label",
        "terrain_z_single_label",
        "ipsilateral_z_label",
        "terrain_ipsi_z_label",
        "ipsi_z_label",
    ],

    "terrain_contralateral_x_label": [
        "terrain_contralateral_x_label",
        "terrain_x_same_label",
        "terrain_x_double_label",
        "contralateral_x_label",
        "terrain_contra_x_label",
        "contra_x_label",
    ],
    "terrain_contralateral_y_label": [
        "terrain_contralateral_y_label",
        "terrain_y_same_label",
        "contralateral_y_label",
        "terrain_contra_y_label",
        "contra_y_label",
    ],
    "terrain_contralateral_z_label": [
        "terrain_contralateral_z_label",
        "terrain_z_same_label",
        "contralateral_z_label",
        "terrain_contra_z_label",
        "contra_z_label",
    ],
}


def clean_column_name(col):
    """Normalize leading/trailing spaces, newlines, and repeated whitespace."""
    s = str(col).strip()
    s = re.sub(r"\s+", "_", s)
    return s



def detect_terrain_column_mapping(columns, task_name=None):
    """
    Detect terrain label columns and normalize their final database names.

    [Final semantics]
    single -> ipsilateral
    same   -> contralateral

    Common source columns for multiaxis tasks (xy/xyz):
      terrain_x_single_label -> terrain_ipsilateral_x_label
      terrain_y_single_label -> terrain_ipsilateral_y_label
      terrain_z_single_label -> terrain_ipsilateral_z_label

      terrain_x_same_label   -> terrain_contralateral_x_label
      terrain_y_same_label   -> terrain_contralateral_y_label
      terrain_z_same_label   -> terrain_contralateral_z_label

    Single-axis source columns may omit x/y/z; infer the axis from task_name:

      x_fwd / x_bwd:
        terrain_single_label -> terrain_ipsilateral_x_label
        terrain_double_label -> terrain_contralateral_x_label

      y_fwd / y_bwd:
        terrain_single_label -> terrain_ipsilateral_y_label
        terrain_same_label   -> terrain_contralateral_y_label

      z1~z4_fwd / z1~z4_bwd:
        terrain_single_label -> terrain_ipsilateral_z_label
        terrain_same_label   -> terrain_contralateral_z_label

    Allow minor variations in separators and capitalization of source columns.
    """
    mapping = {}

    task_name = (task_name or "").lower()

    # Determine the axis of a single-axis task from its task name
    task_axis = None
    if task_name.startswith("x_"):
        task_axis = "x"
    elif task_name.startswith("y_"):
        task_axis = "y"
    elif re.match(r"^z\d+_", task_name):
        task_axis = "z"

    for original in columns:
        clean = clean_column_name(original).lower()
        clean = re.sub(r"[^a-z0-9]+", "_", clean).strip("_")
        tokens = [t for t in clean.split("_") if t]
        token_set = set(tokens)

        if "terrain" not in token_set and "terrain" not in clean:
            continue

        # First check whether source columns explicitly specify x/y/z
        axis = None
        for a in ("x", "y", "z"):
            if a in token_set:
                axis = a
                break

        # Generic terrain_*_label columns in single-axis tasks omit the axis name
        if axis is None:
            axis = task_axis

        if axis is None:
            continue

        # corrected semantic mapping:
        # single = ipsilateral
        # same   = contralateral
        # double = contralateral (for the x-only task)
        if "single" in token_set:
            side = "ipsilateral"
        elif "same" in token_set:
            side = "contralateral"
        elif "double" in token_set and axis == "x":
            side = "contralateral"
        else:
            continue

        canonical = f"terrain_{side}_{axis}_label"
        mapping[original] = canonical

    return mapping


def normalize_columns(df, task_name=None):
    """Normalize column names and rename them using ALIASES."""
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    # First identify terrain labels using flexible matching.
    # This recognizes same/single and x/y/z even if source column names vary.
    terrain_auto_map = detect_terrain_column_mapping(df.columns, task_name=task_name)
    if terrain_auto_map:
        df = df.rename(columns=terrain_auto_map)

    lower_to_real = {str(c).lower(): c for c in df.columns}
    rename_map = {}

    # Handle special aliases first
    for canonical, candidates in ALIASES.items():
        if canonical in df.columns:
            continue

        for cand in candidates:
            real = lower_to_real.get(cand.lower())
            if real is not None:
                rename_map[real] = canonical
                break

    df = df.rename(columns=rename_map)

    # Match sensor data columns to canonical names case-insensitively
    lower_to_real = {str(c).lower(): c for c in df.columns}
    for canonical in DATA_COLUMNS:
        if canonical == "time_s":
            continue
        if canonical not in df.columns:
            real = lower_to_real.get(canonical.lower())
            if real is not None and real != canonical:
                df = df.rename(columns={real: canonical})

    return df


def parse_subject(path):
    """
    Identify sub_001–sub_036 from filenames or parent directories.
    """
    text = str(path).replace("\\", "/")
    m = re.search(r"sub[_-]?(\d{1,3})", text, flags=re.IGNORECASE)
    if not m:
        return None

    num = int(m.group(1))
    if not (1 <= num <= 999):
        return None

    return f"sub_{num:03d}"


def parse_task(path):
    """
    Identify the task from the path.
    Match against predefined TASKS in descending length to avoid confusing x_fwd with xyz1_fwd.
    """
    text = str(path).lower().replace("-", "_")
    for task in sorted(TASKS, key=len, reverse=True):
        pattern = rf"(?<![a-z0-9]){re.escape(task.lower())}(?![a-z0-9])"
        if re.search(pattern, text):
            return task
    return None


def read_excel_safe(path, task_name=None):
    try:
        df = pd.read_excel(path, engine="openpyxl")
        return normalize_columns(df, task_name=task_name), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def calculate_time_s(timestamp_series):
    """
    time_s = timestamp minus the first timestamp

    Supports:
    - datetime
    - Unix seconds
    - Unix milliseconds
    - Unix microseconds
    - Generic numeric timestamps

    Returns:
        time_s, mode
    """
    s = timestamp_series.copy()

    # 1) First check whether the source column is already datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s)
        time_s = (dt - dt.iloc[0]).dt.total_seconds()
        return time_s.astype(float), "datetime"

    # 2) Try numeric conversion
    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = numeric.notna().mean()

    if numeric_ratio >= 0.95:
        if numeric.isna().iloc[0]:
            raise ValueError("First timestamp is invalid; cannot compute time_s")

        base = float(numeric.iloc[0])
        diff = numeric.astype(float) - base

        valid_vals = numeric.dropna().astype(float)
        valid_diff = valid_vals.diff().abs()
        valid_diff = valid_diff[valid_diff > 0]

        median_dt = float(valid_diff.median()) if len(valid_diff) else np.nan
        abs_base = abs(base)

        # Infer units from epoch magnitude and adjacent sampling intervals
        # Epoch seconds are usually around 1e9
        # Epoch milliseconds are usually around 1e12
        # Epoch microseconds are usually around 1e15
        if abs_base >= 1e14:
            return diff / 1_000_000.0, "numeric_microseconds"
        elif abs_base >= 1e11:
            return diff / 1_000.0, "numeric_milliseconds"
        elif abs_base >= 1e8:
            return diff, "numeric_seconds"
        else:
            # For non-epoch numeric values, infer units conservatively from intervals
            if np.isfinite(median_dt):
                if median_dt >= 1000:
                    return diff / 1_000_000.0, "relative_microseconds"
                elif median_dt >= 1:
                    # At 100 Hz, adjacent timestamps in milliseconds differ by about 10
                    return diff / 1_000.0, "relative_milliseconds"

            return diff, "relative_seconds"

    # 3) Finally try parsing datetime strings
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().mean() >= 0.95:
        if pd.isna(dt.iloc[0]):
            raise ValueError("First timestamp cannot be parsed as datetime")
        time_s = (dt - dt.iloc[0]).dt.total_seconds()
        return time_s.astype(float), "datetime_string"

    raise ValueError(
        "Cannot reliably parse timestamp; check whether the column stores numeric timestamps or datetimes."
    )


def count_useful_columns(df):
    wanted = set(DATA_COLUMNS + GAIT_COLUMNS + IPSI_COLUMNS + CONTRA_COLUMNS)
    return len(wanted.intersection(df.columns))


def combine_group_files(file_infos, task_name=None):
    """
    One subject/task pair may contain multiple Excel workbooks.
    Select as base the file containing the most valid columns,
    then fill missing base columns from other files of the same length.

    Do not truncate or force merging files with different row counts.
    """
    loaded = []

    for info in file_infos:
        df, err = read_excel_safe(info["path"], task_name=task_name)
        if err:
            info["read_error"] = err
            continue

        loaded.append({
            "path": info["path"],
            "df": df,
            "score": count_useful_columns(df),
            "rows": len(df),
        })

    if not loaded:
        return None, [], "No readable Excel workbook for this subject/task"

    loaded.sort(key=lambda x: (x["score"], x["rows"]), reverse=True)
    base_info = loaded[0]
    base = base_info["df"].copy()
    used_files = [str(base_info["path"])]
    warnings = []

    for item in loaded[1:]:
        other = item["df"]

        if len(other) != len(base):
            warnings.append(
                f"Skipping supplementary columns from: {item['path']} rows={len(other)}，"
                f"base rows={len(base)}"
            )
            continue

        missing_cols = [c for c in other.columns if c not in base.columns]
        if missing_cols:
            for c in missing_cols:
                base[c] = other[c].values
            used_files.append(str(item["path"]))

    return base, used_files, "; ".join(warnings)


def ensure_dirs(output_root, subject):
    sub_root = output_root / subject

    dirs = {
        "data": sub_root / "data",
        "gait": sub_root / "labels" / "label_gait",
        "ipsi": sub_root / "labels" / "label_terrain_ipsilateral",
        "contra": sub_root / "labels" / "label_terrain_contralateral",
        "figures": sub_root / "figures",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def missing_required(df, cols):
    return [c for c in cols if c not in df.columns]


def write_excel(df, path, overwrite=False):
    if path.exists() and not overwrite:
        return False, "exists"

    df.to_excel(path, index=False, engine="openpyxl")
    return True, "written"


def process_one_group(
    subject,
    task,
    file_infos,
    output_root,
    overwrite=False,
):
    report = {
        "subject": subject,
        "task": task,
        "status": "",
        "source_files": "",
        "rows": "",
        "time_mode": "",
        "data_file": "",
        "gait_file": "",
        "ipsi_file": "",
        "contra_file": "",
        "missing_data_columns": "",
        "missing_gait_columns": "",
        "missing_ipsi_columns": "",
        "missing_contra_columns": "",
        "detected_ipsi_columns": "",
        "detected_contra_columns": "",
        "warning": "",
    }

    df, used_files, warning = combine_group_files(file_infos, task_name=task)

    if df is None:
        report["status"] = "READ_FAILED"
        report["warning"] = warning
        return report

    report["source_files"] = " | ".join(used_files)
    report["rows"] = len(df)
    report["warning"] = warning

    if "timestamp" not in df.columns:
        report["status"] = "NO_TIMESTAMP"
        report["missing_data_columns"] = "timestamp"
        return report

    try:
        time_s, time_mode = calculate_time_s(df["timestamp"])
    except Exception as e:
        report["status"] = "TIME_FAILED"
        report["warning"] = (
            (report["warning"] + "; ") if report["warning"] else ""
        ) + str(e)
        return report

    # Set the first frame to 0 to reduce floating-point error
    time_s = pd.Series(time_s, index=df.index, dtype=float)
    if len(time_s):
        time_s.iloc[0] = 0.0

    df["time_s"] = time_s
    report["time_mode"] = time_mode

    dirs = ensure_dirs(output_root, subject)

    # ---------- DATA ----------
    data_missing = missing_required(df, DATA_COLUMNS)
    report["missing_data_columns"] = ",".join(data_missing)

    if not data_missing:
        data_out = df[DATA_COLUMNS].copy()

        data_path = dirs["data"] / f"{subject}_{task}_data.xlsx"
        _, data_status = write_excel(data_out, data_path, overwrite=overwrite)
        report["data_file"] = str(data_path)

        # The label time axis must come from the data
        master_time_s = data_out["time_s"].copy()
    else:
        # Retain the computed primary time_s for validation even when data columns are missing;
        # do not write incomplete data to the formal dataset.
        master_time_s = df["time_s"].copy()
        data_status = "missing_columns"

    # ---------- GAIT ----------
    gait_source_cols = ["l_label_gait", "r_label_gait"]
    gait_missing = missing_required(df, gait_source_cols)
    report["missing_gait_columns"] = ",".join(gait_missing)

    if not gait_missing:
        gait_out = pd.DataFrame({
            "time_s": master_time_s.values,
            "l_label_gait": df["l_label_gait"].values,
            "r_label_gait": df["r_label_gait"].values,
        })

        gait_path = dirs["gait"] / f"{subject}_{task}_label_gait.xlsx"
        write_excel(gait_out, gait_path, overwrite=overwrite)
        report["gait_file"] = str(gait_path)

    # ---------- TERRAIN LABELS ----------
    #
    # Rules:
    # walk_* / stair_*: no terrain labels; do not generate terrain Excel files.
    #
    # For other tasks, determine the theoretically expected axes from the task type:
    # x_*             -> x
    # y_*             -> y
    # z1~z4_*         -> z
    # xy1~xy6_*       -> x, y
    # xyz1~xyz3_*     -> x, y, z
    #
    # When exporting, extract only terrain columns actually present in the source:
    # - Extract available columns
    # - Do not create missing columns artificially
    # - Report missing expected labels in _build_report.csv
    #
    # normalize_columns() has completed:
    # terrain_x_single_label -> terrain_ipsilateral_x_label
    # terrain_y_single_label -> terrain_ipsilateral_y_label
    # terrain_z_single_label -> terrain_ipsilateral_z_label
    #
    # terrain_x_same_label   -> terrain_contralateral_x_label
    # terrain_y_same_label   -> terrain_contralateral_y_label
    # terrain_z_same_label   -> terrain_contralateral_z_label

    def expected_terrain_axes(task_name):
        """Determine theoretically expected terrain axes from task name."""
        t = task_name.lower()

        if t.startswith("walk_") or t.startswith("stair_"):
            return []

        if t.startswith("xyz"):
            return ["x", "y", "z"]

        if t.startswith("xy"):
            return ["x", "y"]

        if t.startswith("x_"):
            return ["x"]

        if t.startswith("y_"):
            return ["y"]

        if re.match(r"^z\d+_", t):
            return ["z"]

        # Unknown non-walk/stair tasks:
        # Do not guess axes; extract columns actually present in later steps.
        return []

    expected_axes = expected_terrain_axes(task)

    ipsi_by_axis = {
        "x": "terrain_ipsilateral_x_label",
        "y": "terrain_ipsilateral_y_label",
        "z": "terrain_ipsilateral_z_label",
    }

    contra_by_axis = {
        "x": "terrain_contralateral_x_label",
        "y": "terrain_contralateral_y_label",
        "z": "terrain_contralateral_z_label",
    }

    # Extract existing source labels in fixed x, y, z order.
    ipsi_available = [
        ipsi_by_axis[axis]
        for axis in ["x", "y", "z"]
        if ipsi_by_axis[axis] in df.columns
    ]

    contra_available = [
        contra_by_axis[axis]
        for axis in ["x", "y", "z"]
        if contra_by_axis[axis] in df.columns
    ]

    # Labels expected in principle but absent from the source.
    ipsi_missing = [
        ipsi_by_axis[axis]
        for axis in expected_axes
        if ipsi_by_axis[axis] not in df.columns
    ]

    contra_missing = [
        contra_by_axis[axis]
        for axis in expected_axes
        if contra_by_axis[axis] not in df.columns
    ]

    report["missing_ipsi_columns"] = ",".join(ipsi_missing)
    report["missing_contra_columns"] = ",".join(contra_missing)
    report["detected_ipsi_columns"] = ",".join(ipsi_available)
    report["detected_contra_columns"] = ",".join(contra_available)

    # Walk/stair tasks do not require terrain labels.
    terrain_required = not (
        task.lower().startswith("walk_")
        or task.lower().startswith("stair_")
    )

    # ---------- IPSILATERAL ----------
    if terrain_required and ipsi_available:
        ipsi_out = pd.DataFrame({
            "time_s": master_time_s.values
        })

        for col in ipsi_available:
            ipsi_out[col] = df[col].values

        ipsi_path = (
            dirs["ipsi"] /
            f"{subject}_{task}_label_terrain_ipsilateral.xlsx"
        )
        write_excel(
            ipsi_out,
            ipsi_path,
            overwrite=overwrite
        )
        report["ipsi_file"] = str(ipsi_path)

    # ---------- CONTRALATERAL ----------
    if terrain_required and contra_available:
        contra_out = pd.DataFrame({
            "time_s": master_time_s.values
        })

        for col in contra_available:
            contra_out[col] = df[col].values

        contra_path = (
            dirs["contra"] /
            f"{subject}_{task}_label_terrain_contralateral.xlsx"
        )
        write_excel(
            contra_out,
            contra_path,
            overwrite=overwrite
        )
        report["contra_file"] = str(contra_path)

    # ---------- TERRAIN QC ----------
    # For non-walk/stair tasks:
    # Explicitly report if no terrain labels are found for a side.
    if terrain_required:
        if not ipsi_available and not ipsi_missing:
            # expected_axes may be empty for unknown task names.
            ipsi_missing = ["ALL_TERRAIN_IPSILATERAL"]
            report["missing_ipsi_columns"] = ",".join(ipsi_missing)

        if not contra_available and not contra_missing:
            contra_missing = ["ALL_TERRAIN_CONTRALATERAL"]
            report["missing_contra_columns"] = ",".join(contra_missing)

        # Keep the file but warn when observed axes differ from expected task axes.
        actual_ipsi_axes = [
            axis for axis, col in ipsi_by_axis.items()
            if col in df.columns
        ]
        actual_contra_axes = [
            axis for axis, col in contra_by_axis.items()
            if col in df.columns
        ]

        if expected_axes:
            extra_ipsi = [
                axis for axis in actual_ipsi_axes
                if axis not in expected_axes
            ]
            extra_contra = [
                axis for axis in actual_contra_axes
                if axis not in expected_axes
            ]

            if extra_ipsi or extra_contra:
                msg = (
                    f"Unexpected terrain axes for task: "
                    f"expected={expected_axes}, "
                    f"ipsi_extra={extra_ipsi}, "
                    f"contra_extra={extra_contra}"
                )
                report["warning"] = (
                    (report["warning"] + "; ")
                    if report["warning"] else ""
                ) + msg

        # Also warn if same-side and contralateral available axes differ.
        actual_ipsi_set = set(actual_ipsi_axes)
        actual_contra_set = set(actual_contra_axes)

        if actual_ipsi_set != actual_contra_set:
            msg = (
                "Mismatch between ipsilateral and contralateral terrain axes: "
                f"ipsi={actual_ipsi_axes}, "
                f"contra={actual_contra_axes}"
            )
            report["warning"] = (
                (report["warning"] + "; ")
                if report["warning"] else ""
            ) + msg

    # ---------- STATUS ----------
    problems = []

    if data_missing:
        problems.append("DATA_MISSING")

    if gait_missing:
        problems.append("GAIT_MISSING")

    # Do not validate terrain labels for walk/stair tasks.
    if terrain_required:
        if ipsi_missing:
            problems.append("IPSI_MISSING")
        if contra_missing:
            problems.append("CONTRA_MISSING")

        # For non-walk/stair tasks, if a side has no terrain label at all,
        # this must also be reported.
        if not ipsi_available and "IPSI_MISSING" not in problems:
            problems.append("IPSI_MISSING")

        if not contra_available and "CONTRA_MISSING" not in problems:
            problems.append("CONTRA_MISSING")

    if problems:
        report["status"] = "+".join(problems)
    else:
        report["status"] = "OK"

    return report


def scan_source_files(source_root):
    files = sorted(
        p for p in source_root.rglob("*.xlsx")
        if not p.name.startswith("~$")
    )
    return files


def parse_int_list(text):
    """
    Supports:
      --subjects 1 2 3
    """
    if not text:
        return None
    return {f"sub_{int(x):03d}" for x in text}


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize label_3 into processed dataset"
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Source directory, default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory, default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Process selected subjects only, e.g. --subjects 1 7 36",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=TASKS,
        help="Process selected tasks only, e.g. --tasks walk_fwd xyz1_fwd",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if not source_root.exists():
        print(f"[ERROR] Source directory does not exist: {source_root}")
        sys.exit(1)

    selected_subjects = parse_int_list(args.subjects)
    selected_tasks = set(args.tasks) if args.tasks else None

    print("=" * 80)
    print("TerraCon dataset restructuring")
    print(f"Source directory: {source_root}")
    print(f"Output directory: {output_root}")
    print(f"Overwrite existing files: {args.overwrite}")
    print("=" * 80)

    files = scan_source_files(source_root)
    print(f"Excel files scanned: {len(files)}")

    groups = defaultdict(list)
    ignored = []

    for path in files:
        subject = parse_subject(path)
        task = parse_task(path)

        if subject is None or task is None:
            ignored.append(str(path))
            continue

        if selected_subjects and subject not in selected_subjects:
            continue

        if selected_tasks and task not in selected_tasks:
            continue

        groups[(subject, task)].append({"path": path})

    print(f"Subject-task pairs identified: {len(groups)}")

    if ignored:
        print(f"Excel files with unidentified subject/task: {len(ignored)}")

    reports = []

    for i, ((subject, task), file_infos) in enumerate(
        sorted(groups.items()), start=1
    ):
        print(
            f"[{i:03d}/{len(groups):03d}] "
            f"{subject} | {task} | Source files: {len(file_infos)}"
        )

        report = process_one_group(
            subject=subject,
            task=task,
            file_infos=file_infos,
            output_root=output_root,
            overwrite=args.overwrite,
        )
        reports.append(report)

        print(f"    status: {report['status']}")
        print(f"    rows:   {report['rows']}")
        print(f"    time:   {report['time_mode']}")

        if report["missing_data_columns"]:
            print(
                "    Missing DATA: "
                f"{report['missing_data_columns']}"
            )
        if report["missing_gait_columns"]:
            print(
                "    Missing GAIT: "
                f"{report['missing_gait_columns']}"
            )
        if report["missing_ipsi_columns"]:
            print(
                "    Missing IPSI: "
                f"{report['missing_ipsi_columns']}"
            )
        if report["missing_contra_columns"]:
            print(
                "    Missing CONTRA: "
                f"{report['missing_contra_columns']}"
            )

        if report.get("detected_ipsi_columns"):
            print(
                "    Recognized IPSI: "
                f"{report['detected_ipsi_columns']}"
            )

        if report.get("detected_contra_columns"):
            print(
                "    Recognized CONTRA: "
                f"{report['detected_contra_columns']}"
            )

        if report["warning"]:
            print(f"    WARNING: {report['warning']}")

    output_root.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(reports)
    report_path = output_root / "_build_report.csv"
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    if ignored:
        ignored_path = output_root / "_ignored_source_files.txt"
        ignored_path.write_text(
            "\n".join(ignored),
            encoding="utf-8"
        )
    else:
        ignored_path = None

    print("\n" + "=" * 80)
    print("Processing complete")
    print(f"Report: {report_path}")

    if len(report_df):
        print("\nStatus summary:")
        print(report_df["status"].value_counts(dropna=False).to_string())

    if ignored_path:
        print(f"List of unidentified files: {ignored_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
