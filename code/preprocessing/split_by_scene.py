#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
已经拼接好的 7 个 IMU Excel -> 按指定场景直接切分 XLSX

特点
----
1. 支持直接指定受试者、场景、传感器。
2. 指定什么就只处理什么，不扫描/跳过其它目标。
3. 目标 XLSX 已存在时直接覆盖，不再做“已存在跳过”判断。
4. 默认场景开始前 0.5 s、结束后 0.5 s；运行时可分别用 --before 和 --after 指定。
5. 不重新拼接、不滤波、不插值、不重新对齐。
6. 保留拼接 Excel 原始行顺序，只按时间窗口截取。

示例
----
只重做 sub_025 的 stair_fwd pelvis:
    python split_stitched_imu_by_scene.py \
      --subject 25 \
      --scene stair_fwd \
      --sensor G

重做 sub_025 的 stair_fwd 全部 7 个传感器:
    python split_stitched_imu_by_scene.py \
      --subject 25 \
      --scene stair_fwd \
      --all-sensors

重做 sub_025 的两个场景、只处理 pelvis:
    python split_stitched_imu_by_scene.py \
      --subject 25 \
      --scenes stair_fwd stair_bwd \
      --sensor G

重做 sub_025 的全部场景、全部 7 个传感器:
    python split_stitched_imu_by_scene.py \
      --subject 25 \
      --all-scenes \
      --all-sensors

例如：场景开始前保留 1.0 s、结束后保留 2.0 s:
    python split_stitched_imu_by_scene.py \
      --subject 25 \
      --all-scenes \
      --all-sensors \
      --before 1.0 \
      --after 2.0
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


BASE_DIR = Path("/mnt/newdisk/Trigait/Trigait数据库_7.20目录示范_20260721")
STITCHED_ROOT = BASE_DIR / "original_data"
TIME_ROOT = BASE_DIR / "device_export"
OUTPUT_ROOT = BASE_DIR / "original_devide"

SENSOR_MAP = {
    "G": "pelvis",
    "L1": "L_thigh",
    "L2": "L_shank",
    "L3": "L_foot",
    "R1": "R_thigh",
    "R2": "R_shank",
    "R3": "R_foot",
}

SKIP_SHEETS = {"文件信息", "未解析字节", "info", "Info", "INFO"}

OUTPUT_COLUMNS = [
    "sample_index",
    "timestamp",
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
    "gyro_x_deg_s",
    "gyro_y_deg_s",
    "gyro_z_deg_s",
    "angle_x_deg",
    "angle_y_deg",
    "angle_z_deg",
    "timestamp_device",
    "source_file",
    "time_offset_seconds",
    "time_alignment_method",
    "time_alignment_confidence",
    "alignment_anchor_count",
    "alignment_residual_seconds",
]

COLUMN_CANDIDATES = {
    "timestamp": [
        "timestamp",
        "时间戳",
        "时间戳原始字符串",
        "timestamp_aligned",
        "aligned_timestamp",
        "timestamp_device",
    ],
    "timestamp_device": [
        "timestamp_device",
        "设备时间戳",
        "时间戳原始字符串",
        "时间戳",
        "timestamp",
    ],
    "accel_x_g": ["accel_x_g", "X轴加速度(g)", "X轴加速度（g）", "加速度X(g)"],
    "accel_y_g": ["accel_y_g", "Y轴加速度(g)", "Y轴加速度（g）", "加速度Y(g)"],
    "accel_z_g": ["accel_z_g", "Z轴加速度(g)", "Z轴加速度（g）", "加速度Z(g)"],
    "gyro_x_deg_s": [
        "gyro_x_deg_s", "X轴角速度(deg/s)", "X轴角速度（deg/s）", "角速度X(deg/s)"
    ],
    "gyro_y_deg_s": [
        "gyro_y_deg_s", "Y轴角速度(deg/s)", "Y轴角速度（deg/s）", "角速度Y(deg/s)"
    ],
    "gyro_z_deg_s": [
        "gyro_z_deg_s", "Z轴角速度(deg/s)", "Z轴角速度（deg/s）", "角速度Z(deg/s)"
    ],
    "angle_x_deg": ["angle_x_deg", "X轴角度(deg)", "X轴角度（deg）", "角度X(deg)"],
    "angle_y_deg": ["angle_y_deg", "Y轴角度(deg)", "Y轴角度（deg）", "角度Y(deg)"],
    "angle_z_deg": ["angle_z_deg", "Z轴角度(deg)", "Z轴角度（deg）", "角度Z(deg)"],
    "source_file": ["source_file", "源文件", "原始文件"],
    "time_offset_seconds": ["time_offset_seconds"],
    "time_alignment_method": ["time_alignment_method"],
    "time_alignment_confidence": ["time_alignment_confidence"],
    "alignment_anchor_count": ["alignment_anchor_count"],
    "alignment_residual_seconds": ["alignment_residual_seconds"],
}


def normalize_subject(value):
    """25 / 025 / sub_025 -> sub_025"""
    text = str(value).strip()
    m = re.fullmatch(r"sub[_-]?(\d+)", text, flags=re.I)
    if m:
        return f"sub_{int(m.group(1)):03d}"
    if re.fullmatch(r"\d+", text):
        return f"sub_{int(text):03d}"
    raise ValueError(f"无法识别受试者编号: {value}")


def find_scene_time_file(subject):
    subject_num = subject.split("_")[1]
    subject_dir = TIME_ROOT / subject

    exact_candidates = [
        subject_dir / f"场景时间记录表_sub{subject_num}.xlsm",
        subject_dir / f"场景时间记录表_{subject}.xlsm",
        subject_dir / f"场景时间记录表_sub{subject_num}.xlsx",
        subject_dir / f"场景时间记录表_{subject}.xlsx",
    ]

    for p in exact_candidates:
        if p.exists():
            return p

    candidates = sorted(subject_dir.glob("场景时间记录表*.xlsm"))
    if not candidates:
        candidates = sorted(subject_dir.glob("场景时间记录表*.xlsx"))

    if not candidates:
        raise FileNotFoundError(
            f"未找到场景时间记录表: {subject_dir}/场景时间记录表*.xlsm"
        )

    return candidates[0]


def read_scene_table(filepath):
    """读取场景 / 开始时间 / 结束时间。"""
    wb = load_workbook(filepath, data_only=True, read_only=True)

    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        header_row_idx = None
        scene_col = start_col = end_col = None

        for i, row in enumerate(rows[:20]):
            values = [str(v).strip() if v is not None else "" for v in row]
            if "场景" in values and "开始时间" in values and "结束时间" in values:
                header_row_idx = i
                scene_col = values.index("场景")
                start_col = values.index("开始时间")
                end_col = values.index("结束时间")
                break

        if header_row_idx is None:
            continue

        scenes = []
        for row in rows[header_row_idx + 1:]:
            if len(row) <= max(scene_col, start_col, end_col):
                continue

            scene = row[scene_col]
            start = row[start_col]
            end = row[end_col]

            if scene is None or start is None or end is None:
                continue

            start_ts = pd.to_datetime(start, errors="coerce")
            end_ts = pd.to_datetime(end, errors="coerce")

            if pd.isna(start_ts) or pd.isna(end_ts):
                continue
            if end_ts <= start_ts:
                continue

            scenes.append({
                "scene": str(scene).strip(),
                "start": pd.Timestamp(start_ts),
                "end": pd.Timestamp(end_ts),
            })

        if scenes:
            return scenes

    raise ValueError(
        f"时间表中没有找到有效的“场景 / 开始时间 / 结束时间”数据: {filepath}"
    )


def first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_stitched_excel(filepath):
    """
    读取拼接 Excel。
    如果因为 Excel 行数限制拆成多个数据 sheet，会按 sheet 顺序合并。
    """
    xls = pd.ExcelFile(filepath, engine="openpyxl")
    frames = []

    for sheet in xls.sheet_names:
        if sheet in SKIP_SHEETS:
            continue

        header = pd.read_excel(xls, sheet_name=sheet, nrows=0)
        ts_col = first_existing_column(header, COLUMN_CANDIDATES["timestamp"])
        if ts_col is None:
            continue

        df = pd.read_excel(xls, sheet_name=sheet)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(f"Excel 中没有找到带时间戳的 IMU 数据表: {filepath}")

    return pd.concat(frames, ignore_index=True)


def build_standard_frame(df, input_excel):
    """把拼接 Excel 映射成目标 CSV 布局，不修改 IMU 数值。"""
    out = pd.DataFrame(index=df.index)

    required = [
        "timestamp",
        "accel_x_g",
        "accel_y_g",
        "accel_z_g",
        "gyro_x_deg_s",
        "gyro_y_deg_s",
        "gyro_z_deg_s",
        "angle_x_deg",
        "angle_y_deg",
        "angle_z_deg",
    ]

    source_columns = {}
    for target in required:
        source = first_existing_column(df, COLUMN_CANDIDATES[target])
        if source is None:
            raise KeyError(
                f"{input_excel.name} 缺少 {target} 对应列。当前列: {list(df.columns)}"
            )
        source_columns[target] = source

    out["timestamp"] = pd.to_datetime(
        df[source_columns["timestamp"]], errors="coerce"
    )

    for target in required[1:]:
        out[target] = df[source_columns[target]]

    ts_dev_col = first_existing_column(df, COLUMN_CANDIDATES["timestamp_device"])
    if ts_dev_col is not None:
        out["timestamp_device"] = pd.to_datetime(df[ts_dev_col], errors="coerce")
    else:
        out["timestamp_device"] = out["timestamp"]

    source_file_col = first_existing_column(df, COLUMN_CANDIDATES["source_file"])
    if source_file_col is not None:
        out["source_file"] = df[source_file_col].astype(str)
    else:
        out["source_file"] = input_excel.name

    optional_defaults = {
        "time_offset_seconds": 0.0,
        "time_alignment_method": "none_scene_cut_only",
        "time_alignment_confidence": "not_applicable",
        "alignment_anchor_count": 0,
        "alignment_residual_seconds": 0.0,
    }

    for target, default in optional_defaults.items():
        source = first_existing_column(df, COLUMN_CANDIDATES[target])
        if source is not None:
            out[target] = df[source]
        else:
            out[target] = default

    return out


def format_timestamp_milliseconds(series):
    s = pd.to_datetime(series, errors="coerce")
    result = s.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    return result.where(s.notna(), "")


def select_scenes(all_scenes, wanted_scenes):
    by_name = {x["scene"]: x for x in all_scenes}

    missing = [s for s in wanted_scenes if s not in by_name]
    if missing:
        raise ValueError(
            "时间记录表中找不到这些场景: " + ", ".join(missing)
        )

    return [by_name[s] for s in wanted_scenes]


def cut_one_sensor(subject, sensor, body_name, scenes, before_padding, after_padding):
    input_excel = STITCHED_ROOT / subject / f"{subject}_{sensor}.xlsx"

    if not input_excel.exists():
        raise FileNotFoundError(f"拼接 Excel 不存在: {input_excel}")

    print(f"\n[{sensor} -> {body_name}] 读取 {input_excel}")
    raw = read_stitched_excel(input_excel)
    std = build_standard_frame(raw, input_excel)

    invalid_ts = int(std["timestamp"].isna().sum())
    if invalid_ts:
        print(f"  注意: {invalid_ts} 行时间戳无法解析，无法落入场景窗口")

    before_pad = pd.Timedelta(seconds=before_padding)
    after_pad = pd.Timedelta(seconds=after_padding)

    for sc in scenes:
        scene = sc["scene"]
        start = sc["start"] - before_pad
        end = sc["end"] + after_pad

        mask = (std["timestamp"] >= start) & (std["timestamp"] < end)
        part = std.loc[mask].copy()

        # 每个场景重新从 1 编号
        part.insert(0, "sample_index", range(1, len(part) + 1))

        part["timestamp"] = format_timestamp_milliseconds(part["timestamp"])
        part["timestamp_device"] = format_timestamp_milliseconds(
            part["timestamp_device"]
        )

        part = part[OUTPUT_COLUMNS]

        out_dir = OUTPUT_ROOT / subject / "imu" / scene
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / f"{subject}_imu_{body_name}.xlsx"

        # 直接覆盖，不再检查“是否已存在”
        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            part.to_excel(writer, sheet_name="imu", index=False)

            ws = writer.book["imu"]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            # 时间列前面已经格式化成完整毫秒字符串；
            # 在 Excel 中再明确设为文本，后续直接删行 + Ctrl+S 保存，
            # 不会自动变成 08:08.5 之类的格式。
            header_map = {cell.value: cell.column for cell in ws[1]}

            for col_name in ("timestamp", "timestamp_device"):
                if col_name in header_map:
                    col_idx = header_map[col_name]
                    for row_idx in range(2, ws.max_row + 1):
                        ws.cell(row=row_idx, column=col_idx).number_format = "@"

            from openpyxl.utils import get_column_letter

            widths = {
                "sample_index": 14,
                "timestamp": 25,
                "accel_x_g": 14,
                "accel_y_g": 14,
                "accel_z_g": 14,
                "gyro_x_deg_s": 18,
                "gyro_y_deg_s": 18,
                "gyro_z_deg_s": 18,
                "angle_x_deg": 16,
                "angle_y_deg": 16,
                "angle_z_deg": 16,
                "timestamp_device": 25,
                "source_file": 35,
                "time_offset_seconds": 20,
                "time_alignment_method": 24,
                "time_alignment_confidence": 24,
                "alignment_anchor_count": 22,
                "alignment_residual_seconds": 26,
            }

            for name, col_idx in header_map.items():
                ws.column_dimensions[get_column_letter(col_idx)].width = widths.get(name, 16)

        print(
            f"  [覆盖写入] {scene:<14} "
            f"{start.strftime('%H:%M:%S.%f')[:-3]} ~ "
            f"{end.strftime('%H:%M:%S.%f')[:-3]} | "
            f"{len(part)} 行 | {out_file}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="直接指定受试者/场景/传感器，从拼接 Excel 重建场景 XLSX。"
    )

    parser.add_argument(
        "--subject",
        required=True,
        help="受试者，例如 25 / 025 / sub_025",
    )

    scene_group = parser.add_mutually_exclusive_group(required=True)
    scene_group.add_argument(
        "--scene",
        help="只处理一个场景，例如 stair_fwd",
    )
    scene_group.add_argument(
        "--scenes",
        nargs="+",
        help="处理多个场景，例如 --scenes stair_fwd stair_bwd",
    )
    scene_group.add_argument(
        "--all-scenes",
        action="store_true",
        help="处理时间记录表中的全部场景",
    )

    sensor_group = parser.add_mutually_exclusive_group(required=True)
    sensor_group.add_argument(
        "--sensor",
        choices=list(SENSOR_MAP.keys()),
        help="只处理一个传感器：G/L1/L2/L3/R1/R2/R3",
    )
    sensor_group.add_argument(
        "--sensors",
        nargs="+",
        choices=list(SENSOR_MAP.keys()),
        help="处理指定多个传感器，例如 --sensors G L1 L2",
    )
    sensor_group.add_argument(
        "--all-sensors",
        action="store_true",
        help="处理全部 7 个传感器",
    )

    parser.add_argument(
        "--before",
        type=float,
        default=None,
        help="场景开始时间之前额外保留多少秒；默认 0.5",
    )
    parser.add_argument(
        "--after",
        type=float,
        default=None,
        help="场景结束时间之后额外保留多少秒；默认 0.5",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=None,
        help="兼容旧用法：前后统一预留多少秒。例如 --padding 1。若同时给 --before/--after，则对应的 --before/--after 优先。",
    )

    args = parser.parse_args()

    base_padding = 0.5 if args.padding is None else args.padding
    before_padding = base_padding if args.before is None else args.before
    after_padding = base_padding if args.after is None else args.after

    if args.padding is not None and args.padding < 0:
        parser.error("--padding 不能小于 0")
    if before_padding < 0:
        parser.error("--before 不能小于 0")
    if after_padding < 0:
        parser.error("--after 不能小于 0")

    subject = normalize_subject(args.subject)

    if args.scene:
        wanted_scenes = [args.scene]
    elif args.scenes:
        wanted_scenes = args.scenes
    else:
        wanted_scenes = None

    if args.sensor:
        sensors = [args.sensor]
    elif args.sensors:
        sensors = args.sensors
    else:
        sensors = list(SENSOR_MAP.keys())

    time_file = find_scene_time_file(subject)
    all_scenes = read_scene_table(time_file)

    if args.all_scenes:
        scenes = all_scenes
        wanted_scenes = [x["scene"] for x in all_scenes]
    else:
        scenes = select_scenes(all_scenes, wanted_scenes)

    print("=" * 100)
    print("指定场景直接重建 IMU XLSX")
    print(f"受试者:       {subject}")
    print(f"场景:         {', '.join(wanted_scenes)}")
    print(f"传感器:       {', '.join(sensors)}")
    print(f"拼接 Excel:   {STITCHED_ROOT / subject}")
    print(f"时间记录表:   {time_file}")
    print(f"输出目录:     {OUTPUT_ROOT / subject / 'imu'}")
    print(f"时间窗口:     开始前 {before_padding:.3f} 秒，结束后 {after_padding:.3f} 秒")
    print("已有目标 XLSX: 直接覆盖")
    print("=" * 100)

    for sensor in sensors:
        cut_one_sensor(
            subject=subject,
            sensor=sensor,
            body_name=SENSOR_MAP[sensor],
            scenes=scenes,
            before_padding=before_padding,
            after_padding=after_padding,
        )

    print("\n完成。")


if __name__ == "__main__":
    main()
