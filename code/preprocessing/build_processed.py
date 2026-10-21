#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 label_3 中的源 Excel 重组为统一 processed 数据库结构。

目标结构：
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

核心规则：
1. 不删行、不截断。
2. time_s = timestamp - 当前任务第一帧 timestamp。
3. 三类 label 的 time_s 直接复制 data 的 time_s，保证逐行严格一致。
4. data 列顺序固定。
5. 自动递归扫描 source-root 下全部 xlsx。
6. walk/stair 不生成 terrain；其他任务按任务轴和源文件实际存在列动态拆分 terrain。
7. terrain 语义固定：single=ipsilateral，same=contralateral；单独 x 的 double=contralateral。
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


# 常见旧列名 -> 新列名
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
    """统一去掉首尾空格、换行、重复空白。"""
    s = str(col).strip()
    s = re.sub(r"\s+", "_", s)
    return s



def detect_terrain_column_mapping(columns, task_name=None):
    """
    自动识别 terrain 标签列，并统一为最终数据库列名。

    【最终语义】
    single -> ipsilateral
    same   -> contralateral

    多轴任务（xy / xyz）常见源列：
      terrain_x_single_label -> terrain_ipsilateral_x_label
      terrain_y_single_label -> terrain_ipsilateral_y_label
      terrain_z_single_label -> terrain_ipsilateral_z_label

      terrain_x_same_label   -> terrain_contralateral_x_label
      terrain_y_same_label   -> terrain_contralateral_y_label
      terrain_z_same_label   -> terrain_contralateral_z_label

    单轴任务的源列可能不带 x/y/z，需要根据 task_name 判断轴：

      x_fwd / x_bwd:
        terrain_single_label -> terrain_ipsilateral_x_label
        terrain_double_label -> terrain_contralateral_x_label

      y_fwd / y_bwd:
        terrain_single_label -> terrain_ipsilateral_y_label
        terrain_same_label   -> terrain_contralateral_y_label

      z1~z4_fwd / z1~z4_bwd:
        terrain_single_label -> terrain_ipsilateral_z_label
        terrain_same_label   -> terrain_contralateral_z_label

    允许源列中分隔符、大小写略有变化。
    """
    mapping = {}

    task_name = (task_name or "").lower()

    # 根据任务判断单轴任务的轴
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

        # 先查源列里是否明确写了 x/y/z
        axis = None
        for a in ("x", "y", "z"):
            if a in token_set:
                axis = a
                break

        # 单轴任务的 generic terrain_*_label 不带轴名
        if axis is None:
            axis = task_axis

        if axis is None:
            continue

        # corrected semantic mapping:
        # single = ipsilateral
        # same   = contralateral
        # double = contralateral（单独 x 任务）
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
    """规范列名，并根据 ALIASES 自动重命名。"""
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    # 先对 terrain 标签做宽松识别。
    # 这样即使源文件列名不是完全固定格式，也能识别 same/single + x/y/z。
    terrain_auto_map = detect_terrain_column_mapping(df.columns, task_name=task_name)
    if terrain_auto_map:
        df = df.rename(columns=terrain_auto_map)

    lower_to_real = {str(c).lower(): c for c in df.columns}
    rename_map = {}

    # 先处理特殊 alias
    for canonical, candidates in ALIASES.items():
        if canonical in df.columns:
            continue

        for cand in candidates:
            real = lower_to_real.get(cand.lower())
            if real is not None:
                rename_map[real] = canonical
                break

    df = df.rename(columns=rename_map)

    # data 传感器列按大小写不敏感匹配 canonical
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
    从文件名或上级路径中识别 sub_001 ~ sub_036。
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
    从路径中识别任务。
    使用预定义 TASKS，按长度降序避免 x_fwd 错匹配到 xyz1_fwd 等。
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
    time_s = timestamp - 第一帧 timestamp

    支持：
    - datetime
    - Unix 秒
    - Unix 毫秒
    - Unix 微秒
    - 普通数值时间

    返回：
        time_s, mode
    """
    s = timestamp_series.copy()

    # 1) 优先判断原列是否明显为 datetime 类型
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s)
        time_s = (dt - dt.iloc[0]).dt.total_seconds()
        return time_s.astype(float), "datetime"

    # 2) 尝试数值
    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = numeric.notna().mean()

    if numeric_ratio >= 0.95:
        if numeric.isna().iloc[0]:
            raise ValueError("timestamp 第一帧不是有效数值，无法计算 time_s")

        base = float(numeric.iloc[0])
        diff = numeric.astype(float) - base

        valid_vals = numeric.dropna().astype(float)
        valid_diff = valid_vals.diff().abs()
        valid_diff = valid_diff[valid_diff > 0]

        median_dt = float(valid_diff.median()) if len(valid_diff) else np.nan
        abs_base = abs(base)

        # 依据 epoch 数量级 + 相邻采样间隔判断单位
        # 秒级 epoch 通常 ~1e9
        # 毫秒级 epoch 通常 ~1e12
        # 微秒级 epoch 通常 ~1e15
        if abs_base >= 1e14:
            return diff / 1_000_000.0, "numeric_microseconds"
        elif abs_base >= 1e11:
            return diff / 1_000.0, "numeric_milliseconds"
        elif abs_base >= 1e8:
            return diff, "numeric_seconds"
        else:
            # 非 epoch 数值：根据相邻间隔做保守判断
            if np.isfinite(median_dt):
                if median_dt >= 1000:
                    return diff / 1_000_000.0, "relative_microseconds"
                elif median_dt >= 1:
                    # 常见 100 Hz 毫秒时间戳相邻约 10
                    return diff / 1_000.0, "relative_milliseconds"

            return diff, "relative_seconds"

    # 3) 最后尝试字符串 datetime
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().mean() >= 0.95:
        if pd.isna(dt.iloc[0]):
            raise ValueError("timestamp 第一帧无法解析为 datetime")
        time_s = (dt - dt.iloc[0]).dt.total_seconds()
        return time_s.astype(float), "datetime_string"

    raise ValueError(
        "timestamp 无法可靠解析。请检查该列是数值时间戳还是日期时间格式。"
    )


def count_useful_columns(df):
    wanted = set(DATA_COLUMNS + GAIT_COLUMNS + IPSI_COLUMNS + CONTRA_COLUMNS)
    return len(wanted.intersection(df.columns))


def combine_group_files(file_infos, task_name=None):
    """
    同一 subject + task 可能有多个 Excel。
    选择包含有效列最多的文件作为 base，
    再从其他同长度文件补充 base 中缺失的列。

    不做截断，不因长度不一致而硬合并。
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
        return None, [], "该 subject-task 没有可读取的 Excel"

    loaded.sort(key=lambda x: (x["score"], x["rows"]), reverse=True)
    base_info = loaded[0]
    base = base_info["df"].copy()
    used_files = [str(base_info["path"])]
    warnings = []

    for item in loaded[1:]:
        other = item["df"]

        if len(other) != len(base):
            warnings.append(
                f"跳过补列：{item['path']} 行数={len(other)}，"
                f"base 行数={len(base)}"
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

    # 强制第一帧为 0，减小浮点误差
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

        # label 的时间轴必须来自 data
        master_time_s = data_out["time_s"].copy()
    else:
        # 即使 data 缺列，也先保留计算出的主 time_s 做检查；
        # 但不会把不完整 data 写入正式数据库。
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
    # 规则：
    # walk_* / stair_*：没有 terrain 标签，不生成 terrain Excel。
    #
    # 其他任务根据任务类型确定“理论上应该存在”的轴：
    # x_*             -> x
    # y_*             -> y
    # z1~z4_*         -> z
    # xy1~xy6_*       -> x, y
    # xyz1~xyz3_*     -> x, y, z
    #
    # 但正式输出时仍然按照源文件“实际存在”的 terrain 列提取：
    # - 有什么提取什么
    # - 不人为补空列
    # - 若理论应有的标签缺失，在 _build_report.csv 中报告
    #
    # normalize_columns() 已完成：
    # terrain_x_single_label -> terrain_ipsilateral_x_label
    # terrain_y_single_label -> terrain_ipsilateral_y_label
    # terrain_z_single_label -> terrain_ipsilateral_z_label
    #
    # terrain_x_same_label   -> terrain_contralateral_x_label
    # terrain_y_same_label   -> terrain_contralateral_y_label
    # terrain_z_same_label   -> terrain_contralateral_z_label

    def expected_terrain_axes(task_name):
        """根据任务名判断理论上应该存在的 terrain 轴。"""
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

        # 未知的非 walk/stair 任务：
        # 不强行猜轴，但后面仍会按实际存在列提取。
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

    # 按 x, y, z 固定顺序提取源文件实际存在的标签。
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

    # 理论应有但源文件中没有的标签。
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

    # walk / stair 不需要 terrain。
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
    # 对非 walk/stair 任务：
    # 如果完全没有找到某一侧 terrain 标签，要明确报错。
    if terrain_required:
        if not ipsi_available and not ipsi_missing:
            # 未知任务名时，expected_axes 可能为空。
            ipsi_missing = ["ALL_TERRAIN_IPSILATERAL"]
            report["missing_ipsi_columns"] = ",".join(ipsi_missing)

        if not contra_available and not contra_missing:
            contra_missing = ["ALL_TERRAIN_CONTRALATERAL"]
            report["missing_contra_columns"] = ",".join(contra_missing)

        # 如果实际出现的轴与理论任务轴不符，保留文件但给 warning。
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
                    f"terrain出现任务预期之外的轴: "
                    f"expected={expected_axes}, "
                    f"ipsi_extra={extra_ipsi}, "
                    f"contra_extra={extra_contra}"
                )
                report["warning"] = (
                    (report["warning"] + "; ")
                    if report["warning"] else ""
                ) + msg

        # 同侧和对侧实际存在的轴不一致，也给 warning。
        actual_ipsi_set = set(actual_ipsi_axes)
        actual_contra_set = set(actual_contra_axes)

        if actual_ipsi_set != actual_contra_set:
            msg = (
                "ipsilateral/contralateral terrain轴不一致: "
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

    # walk/stair 不检查 terrain。
    if terrain_required:
        if ipsi_missing:
            problems.append("IPSI_MISSING")
        if contra_missing:
            problems.append("CONTRA_MISSING")

        # 对非 walk/stair，如果某一侧连一个 terrain 标签都没找到，
        # 也必须报告。
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
    支持：
      --subjects 1 2 3
    """
    if not text:
        return None
    return {f"sub_{int(x):03d}" for x in text}


def main():
    parser = argparse.ArgumentParser(
        description="将 label_3 重组为 processed 数据库"
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"源目录，默认：{DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"输出目录，默认：{DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="只处理指定受试者，例如：--subjects 1 7 36",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=TASKS,
        help="只处理指定任务，例如：--tasks walk_fwd xyz1_fwd",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在输出文件",
    )

    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if not source_root.exists():
        print(f"[ERROR] 源目录不存在：{source_root}")
        sys.exit(1)

    selected_subjects = parse_int_list(args.subjects)
    selected_tasks = set(args.tasks) if args.tasks else None

    print("=" * 80)
    print("Trigait 数据库重组")
    print(f"源目录: {source_root}")
    print(f"输出目录: {output_root}")
    print(f"覆盖已有文件: {args.overwrite}")
    print("=" * 80)

    files = scan_source_files(source_root)
    print(f"扫描到 Excel: {len(files)} 个")

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

    print(f"识别到 subject-task 组合: {len(groups)} 个")

    if ignored:
        print(f"无法识别 subject/task 的 Excel: {len(ignored)} 个")

    reports = []

    for i, ((subject, task), file_infos) in enumerate(
        sorted(groups.items()), start=1
    ):
        print(
            f"[{i:03d}/{len(groups):03d}] "
            f"{subject} | {task} | 源文件 {len(file_infos)} 个"
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
                "    缺少 DATA: "
                f"{report['missing_data_columns']}"
            )
        if report["missing_gait_columns"]:
            print(
                "    缺少 GAIT: "
                f"{report['missing_gait_columns']}"
            )
        if report["missing_ipsi_columns"]:
            print(
                "    缺少 IPSI: "
                f"{report['missing_ipsi_columns']}"
            )
        if report["missing_contra_columns"]:
            print(
                "    缺少 CONTRA: "
                f"{report['missing_contra_columns']}"
            )

        if report.get("detected_ipsi_columns"):
            print(
                "    已识别 IPSI: "
                f"{report['detected_ipsi_columns']}"
            )

        if report.get("detected_contra_columns"):
            print(
                "    已识别 CONTRA: "
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
    print("处理完成")
    print(f"报告: {report_path}")

    if len(report_df):
        print("\n状态统计:")
        print(report_df["status"].value_counts(dropna=False).to_string())

    if ignored_path:
        print(f"未识别文件列表: {ignored_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
