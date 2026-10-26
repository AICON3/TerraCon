#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量转换 device_export/sub_001 ~ sub_036 中 G/L1/L2/L3/R1/R2/R3 的 WIT 二进制文件。

目标：
1. 不修改、不删除 device_export 中任何原始文件。
2. 每个 WIT 文件单独转换成一个 Excel。
3. 每个传感器目录内的 WIT0、WIT1、WIT2... 按编号顺序再拼接成一个 Excel。
4. 不因为同步头错误、时间非法等原因删除原始数据：
   - 正常 28 字节帧写入“数据”工作表；
   - 同步头之间出现的错位/异常字节写入“未解析字节”工作表；
   - 文件尾不足 28 字节的内容也写入“未解析字节”工作表。
5. 不对数值做 round()；同时保存每帧 raw_hex，便于回查原始字节。
6. 拼接时不按时间重排，只按 WIT 文件编号 + 文件内原始顺序拼接。

默认输入：
/mnt/newdisk/Trigait/Trigait数据库_7.20目录示范_20260721/device_export

默认输出：
/mnt/newdisk/Trigait/Trigait数据库_7.20目录示范_20260721/original_picture

输出结构示例：
original_picture/
  sub_001/
    G/
      sub_001_G_WIT0.xlsx
      sub_001_G_WIT1.xlsx
      sub_001_G_拼接.xlsx
    L1/
      ...
"""

import os
import re
import sys
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Tuple

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font, PatternFill, Alignment


# =========================
# Path configuration
# =========================
INPUT_ROOT = "/mnt/newdisk/Trigait/Trigait数据库_7.20目录示范_20260721/device_export"
OUTPUT_ROOT = "/mnt/newdisk/Trigait/Trigait数据库_7.20目录示范_20260721/original_picture"

SUBJECT_START = 11
SUBJECT_END = 11
TARGET_DIR_NAMES = ["G", "L1", "L2", "L3", "R1", "R2", "R3"]

# =========================
# Binary format configuration
# =========================
RECORD_SIZE = 28
SYNC_HEADER = bytes([0x55, 0x61])

# Excel worksheets have a limit of 1,048,576 rows; reserve one for the header
MAX_DATA_ROWS_PER_SHEET = 1_048_575
# Excel text cells are limited to about 32,767 characters; chunk raw_hex data at ~3 characters per byte
MAX_UNPARSED_BYTES_PER_ROW = 8000

DATA_COLUMNS = [
    "拼接序号",
    "源文件",
    "文件内帧序号",
    "源字节起始偏移",
    "同步头正确",
    "时间是否合法",
    "时间戳原始字符串",
    "年份",
    "月份",
    "日期",
    "小时",
    "分钟",
    "秒",
    "毫秒",
    "X轴加速度(g)",
    "Y轴加速度(g)",
    "Z轴加速度(g)",
    "X轴角速度(deg/s)",
    "Y轴角速度(deg/s)",
    "Z轴角速度(deg/s)",
    "X轴角度(deg)",
    "Y轴角度(deg)",
    "Z轴角度(deg)",
    "raw_hex",
]

UNPARSED_COLUMNS = [
    "源文件",
    "起始偏移",
    "结束偏移_不含",
    "字节数",
    "原因",
    "raw_hex",
]

INFO_COLUMNS = ["项目", "内容"]

HEADER_FONT = Font(name="Microsoft YaHei", bold=True, color="FFFFFF", size=11)
HEADER_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
HEADER_ALIGN = Alignment(horizontal="center", vertical="center")


def to_signed(raw: int) -> int:
    """16 位无符号整数 -> 有符号整数。"""
    return raw - 65536 if raw & 0x8000 else raw


def read_i16_le(buf: bytes, offset: int) -> int:
    raw = buf[offset] | (buf[offset + 1] << 8)
    return to_signed(raw)


def check_time_valid(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    millisecond: int,
) -> bool:
    """只做标记，不会因为时间非法删除该帧。"""
    if not (2000 <= year <= 2099):
        return False
    if not (0 <= millisecond <= 999):
        return False
    try:
        datetime(year, month, day, hour, minute, second, millisecond * 1000)
        return True
    except Exception:
        return False


def parse_frame(chunk: bytes, byte_offset: int, source_filename: str, file_frame_index: int) -> Dict:
    """解析完整 28 字节帧。即使时间非法，也保留并返回。"""
    if len(chunk) != RECORD_SIZE:
        raise ValueError(f"帧长度不是 {RECORD_SIZE} 字节")

    sync_ok = chunk[:2] == SYNC_HEADER
    data = chunk[2:]

    accel_x = read_i16_le(data, 0) / 32768.0 * 16.0
    accel_y = read_i16_le(data, 2) / 32768.0 * 16.0
    accel_z = read_i16_le(data, 4) / 32768.0 * 16.0

    gyro_x = read_i16_le(data, 6) / 32768.0 * 2000.0
    gyro_y = read_i16_le(data, 8) / 32768.0 * 2000.0
    gyro_z = read_i16_le(data, 10) / 32768.0 * 2000.0

    angle_x = read_i16_le(data, 12) / 32768.0 * 180.0
    angle_y = read_i16_le(data, 14) / 32768.0 * 180.0
    angle_z = read_i16_le(data, 16) / 32768.0 * 180.0

    year = 2000 + data[18]
    month = data[19]
    day = data[20]
    hour = data[21]
    minute = data[22]
    second = data[23]
    millisecond = data[24] | (data[25] << 8)

    timestamp_raw = (
        f"{year:04d}/{month:02d}/{day:02d} "
        f"{hour:02d}:{minute:02d}:{second:02d}.{millisecond:03d}"
    )

    time_valid = check_time_valid(year, month, day, hour, minute, second, millisecond)

    return {
        "source_filename": source_filename,
        "file_frame_index": file_frame_index,
        "byte_offset": byte_offset,
        "sync_ok": sync_ok,
        "time_valid": time_valid,
        "timestamp_raw": timestamp_raw,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "millisecond": millisecond,
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "angle_x": angle_x,
        "angle_y": angle_y,
        "angle_z": angle_z,
        "raw_hex": chunk.hex(" ").upper(),
    }


def iter_file_content(filepath: str) -> Iterator[Tuple[str, Dict]]:
    """
    逐文件扫描：
    - 遇到 55 61 且后面至少 28 字节：解析为 frame；
    - 其它字节不丢弃，连续收集为 unparsed；
    - 文件尾不足 28 字节也进入 unparsed。

    产出：("frame", record) 或 ("unparsed", segment)
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    source_filename = os.path.basename(filepath)
    total = len(raw)
    pos = 0
    file_frame_index = 0

    unparsed_start: Optional[int] = None

    def flush_unparsed(end_pos: int, reason: str):
        """
        返回一个列表。异常原始字节过长时分块写入多行，避免超过 Excel 单元格文本上限。
        分块只影响 Excel 展示，不丢字节、不改变字节顺序。
        """
        nonlocal unparsed_start
        if unparsed_start is None or end_pos <= unparsed_start:
            return []

        start = unparsed_start
        outputs = []
        while start < end_pos:
            chunk_end = min(start + MAX_UNPARSED_BYTES_PER_ROW, end_pos)
            segment = raw[start:chunk_end]
            outputs.append({
                "source_filename": source_filename,
                "start_offset": start,
                "end_offset": chunk_end,
                "byte_count": len(segment),
                "reason": reason,
                "raw_hex": segment.hex(" ").upper(),
            })
            start = chunk_end

        unparsed_start = None
        return outputs

    while pos < total:
        remain = total - pos

        # If 28 bytes remain and the current position is a sync header, read a complete frame
        if remain >= RECORD_SIZE and raw[pos:pos + 2] == SYNC_HEADER:
            pending_list = flush_unparsed(pos, "同步头之间的异常/错位原始字节")
            for pending in pending_list:
                yield "unparsed", pending

            chunk = raw[pos:pos + RECORD_SIZE]
            file_frame_index += 1
            rec = parse_frame(chunk, pos, source_filename, file_frame_index)
            yield "frame", rec
            pos += RECORD_SIZE
            continue

        # If fewer than one frame remains, preserve all remaining bytes
        if remain < RECORD_SIZE:
            if unparsed_start is None:
                unparsed_start = pos
            pos = total
            break

        # If not at a sync header, remember the offset and scan byte by byte for the next sync header
        if unparsed_start is None:
            unparsed_start = pos
        pos += 1

    pending_list = flush_unparsed(total, "文件尾不足28字节或末尾异常/错位原始字节")
    for pending in pending_list:
        yield "unparsed", pending


def is_wit_file(filename: str) -> bool:
    """识别 WIT0/WIT1/...，允许无扩展名或 .txt/.bin/.dat。"""
    if filename.startswith("."):
        return False

    name, ext = os.path.splitext(filename)
    if ext.upper() not in ("", ".TXT", ".BIN", ".DAT"):
        return False
    return re.fullmatch(r"WIT\d+", name.upper()) is not None


def wit_sort_key(filename: str) -> int:
    name, _ = os.path.splitext(filename)
    m = re.fullmatch(r"WIT(\d+)", name.upper())
    return int(m.group(1)) if m else 10**12


def make_header_cell(ws, value: str):
    cell = WriteOnlyCell(ws, value=value)
    cell.font = HEADER_FONT
    cell.fill = HEADER_FILL
    cell.alignment = HEADER_ALIGN
    return cell


class ExcelStreamWriter:
    """支持数据超过 Excel 单 sheet 行数时自动分 sheet。"""

    def __init__(self, output_path: str, title_prefix: str = "数据"):
        self.output_path = output_path
        self.wb = Workbook(write_only=True)
        self.title_prefix = title_prefix
        self.data_sheet_index = 0
        self.current_data_rows = 0
        self.data_ws = None
        self.unparsed_ws = None
        self.info_ws = None
        self.total_frames = 0
        self.total_unparsed_segments = 0
        self.total_unparsed_bytes = 0
        self._new_data_sheet()
        self._create_unparsed_sheet()

    def _new_data_sheet(self):
        self.data_sheet_index += 1
        title = f"{self.title_prefix}_{self.data_sheet_index:03d}"
        self.data_ws = self.wb.create_sheet(title=title)
        self.data_ws.append([make_header_cell(self.data_ws, x) for x in DATA_COLUMNS])
        self.current_data_rows = 0

    def _create_unparsed_sheet(self):
        self.unparsed_ws = self.wb.create_sheet(title="未解析字节")
        self.unparsed_ws.append([make_header_cell(self.unparsed_ws, x) for x in UNPARSED_COLUMNS])

    def append_frame(self, rec: Dict, concat_index: int):
        if self.current_data_rows >= MAX_DATA_ROWS_PER_SHEET:
            self._new_data_sheet()

        row = [
            concat_index,
            rec["source_filename"],
            rec["file_frame_index"],
            rec["byte_offset"],
            rec["sync_ok"],
            rec["time_valid"],
            rec["timestamp_raw"],
            rec["year"],
            rec["month"],
            rec["day"],
            rec["hour"],
            rec["minute"],
            rec["second"],
            rec["millisecond"],
            rec["accel_x"],
            rec["accel_y"],
            rec["accel_z"],
            rec["gyro_x"],
            rec["gyro_y"],
            rec["gyro_z"],
            rec["angle_x"],
            rec["angle_y"],
            rec["angle_z"],
            rec["raw_hex"],
        ]
        self.data_ws.append(row)
        self.current_data_rows += 1
        self.total_frames += 1

    def append_unparsed(self, seg: Dict):
        self.unparsed_ws.append([
            seg["source_filename"],
            seg["start_offset"],
            seg["end_offset"],
            seg["byte_count"],
            seg["reason"],
            seg["raw_hex"],
        ])
        self.total_unparsed_segments += 1
        self.total_unparsed_bytes += seg["byte_count"]

    def add_info_sheet(self, info_items: List[Tuple[str, str]]):
        self.info_ws = self.wb.create_sheet(title="文件信息")
        self.info_ws.append([make_header_cell(self.info_ws, x) for x in INFO_COLUMNS])
        for key, value in info_items:
            self.info_ws.append([key, value])

    def save(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.wb.save(self.output_path)


def convert_one_wit_file(filepath: str, output_path: str, sub_name: str, sensor_name: str) -> Dict[str, int]:
    """一个 WIT -> 一个 Excel。"""
    writer = ExcelStreamWriter(output_path)
    concat_index = 0

    for kind, obj in iter_file_content(filepath):
        if kind == "frame":
            concat_index += 1
            writer.append_frame(obj, concat_index)
        else:
            writer.append_unparsed(obj)

    file_size = os.path.getsize(filepath)
    writer.add_info_sheet([
        ("受试者", sub_name),
        ("传感器目录", sensor_name),
        ("源文件", filepath),
        ("源文件字节数", str(file_size)),
        ("完整帧数", str(writer.total_frames)),
        ("未解析片段数", str(writer.total_unparsed_segments)),
        ("未解析字节数", str(writer.total_unparsed_bytes)),
        ("记录大小", f"{RECORD_SIZE} 字节/帧"),
        ("同步头", SYNC_HEADER.hex(" ").upper()),
        ("说明", "没有因为时间非法或错位而删除原始字节；完整帧保存 raw_hex，异常/尾部字节保存在未解析字节工作表。"),
    ])
    writer.save()

    return {
        "frames": writer.total_frames,
        "unparsed_segments": writer.total_unparsed_segments,
        "unparsed_bytes": writer.total_unparsed_bytes,
        "file_bytes": file_size,
    }


def convert_and_merge_sensor_dir(input_dir: str, output_dir: str, sub_name: str, sensor_name: str) -> None:
    """
    处理一个 G/L1/... 目录：
    1) 每个 WIT 单独 Excel；
    2) 所有 WIT 按编号顺序拼接成一个 Excel。
    """
    if not os.path.isdir(input_dir):
        print(f"  [跳过] 目录不存在: {input_dir}")
        return

    wit_files = [
        fn for fn in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, fn)) and is_wit_file(fn)
    ]
    wit_files.sort(key=wit_sort_key)

    if not wit_files:
        print(f"  [跳过] 未找到 WIT 文件: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"  [{sensor_name}] 找到 {len(wit_files)} 个 WIT 文件")

    # Convert each source file first
    for fn in wit_files:
        src = os.path.join(input_dir, fn)
        wit_stem = os.path.splitext(fn)[0]
        out_name = f"{sub_name}_{sensor_name}_{wit_stem}.xlsx"
        out_path = os.path.join(output_dir, out_name)

        print(f"    单文件转换: {fn} -> {out_name}")
        stats = convert_one_wit_file(src, out_path, sub_name, sensor_name)
        print(
            f"      完整帧={stats['frames']}，"
            f"未解析字节={stats['unparsed_bytes']}，"
            f"源文件={stats['file_bytes']} 字节"
        )

    # Then concatenate at the directory level
    merge_name = f"{sub_name}_{sensor_name}_拼接.xlsx"
    merge_path = os.path.join(output_dir, merge_name)
    merge_writer = ExcelStreamWriter(merge_path)
    concat_index = 0
    total_source_bytes = 0

    for fn in wit_files:
        src = os.path.join(input_dir, fn)
        total_source_bytes += os.path.getsize(src)
        print(f"    拼接读取: {fn}")

        for kind, obj in iter_file_content(src):
            if kind == "frame":
                concat_index += 1
                merge_writer.append_frame(obj, concat_index)
            else:
                merge_writer.append_unparsed(obj)

    merge_writer.add_info_sheet([
        ("受试者", sub_name),
        ("传感器目录", sensor_name),
        ("源目录", input_dir),
        ("拼接文件顺序", ", ".join(wit_files)),
        ("源文件数量", str(len(wit_files))),
        ("源文件总字节数", str(total_source_bytes)),
        ("拼接完整帧数", str(merge_writer.total_frames)),
        ("未解析片段数", str(merge_writer.total_unparsed_segments)),
        ("未解析字节数", str(merge_writer.total_unparsed_bytes)),
        ("拼接规则", "按 WIT 数字编号升序；每个文件内部保持原始顺序；不按时间排序。"),
        ("说明", "没有因为时间非法或同步异常删除原始字节；异常字节统一保存在未解析字节工作表。"),
    ])
    merge_writer.save()
    print(f"    已生成拼接文件: {merge_path}")


def main(input_root: str = INPUT_ROOT, output_root: str = OUTPUT_ROOT) -> None:
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)

    print("=" * 90)
    print("device_export 批量原样转换 + 拼接")
    print(f"输入根目录: {input_root}")
    print(f"输出根目录: {output_root}")
    print(f"受试者范围: sub_{SUBJECT_START:03d} ~ sub_{SUBJECT_END:03d}")
    print(f"传感器目录: {', '.join(TARGET_DIR_NAMES)}")
    print("原则: 不删除原始文件；不按时间筛帧；不 round；异常字节保留到 Excel。")
    print("=" * 90)

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"输入根目录不存在: {input_root}")

    os.makedirs(output_root, exist_ok=True)

    for subject_no in range(SUBJECT_START, SUBJECT_END + 1):
        sub_name = f"sub_{subject_no:03d}"
        sub_input = os.path.join(input_root, sub_name)
        sub_output = os.path.join(output_root, sub_name)

        print(f"\n{'#' * 90}")
        print(f"处理 {sub_name}")

        if not os.path.isdir(sub_input):
            print(f"[跳过] 受试者目录不存在: {sub_input}")
            continue

        for sensor_name in TARGET_DIR_NAMES:
            sensor_input = os.path.join(sub_input, sensor_name)
            sensor_output = os.path.join(sub_output, sensor_name)
            convert_and_merge_sensor_dir(
                input_dir=sensor_input,
                output_dir=sensor_output,
                sub_name=sub_name,
                sensor_name=sensor_name,
            )

    print("\n全部处理结束。")


if __name__ == "__main__":
    # Optional CLI arguments:
    # python convert_device_export_no_trim.py [input_root] [output_root]
    in_root = sys.argv[1] if len(sys.argv) >= 2 else INPUT_ROOT
    out_root = sys.argv[2] if len(sys.argv) >= 3 else OUTPUT_ROOT
    main(in_root, out_root)
