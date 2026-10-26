#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-convert WIT binary files for G/L1/L2/L3/R1/R2/R3 under device_export/sub_001–sub_036.

Goals:
1. Do not change or delete any source files in device_export.
2. Convert each WIT file into a separate Excel workbook.
3. Concatenate WIT0/WIT1/WIT2... files numerically into one workbook per sensor.
4. Do not discard raw bytes due to sync-header or invalid timestamp errors:
   - Valid 28-byte frames go to the data worksheet;
   - Misaligned or invalid bytes between sync headers go to the unparsed-bytes worksheet;
   - Trailing fragments shorter than 28 bytes also go to the unparsed-bytes worksheet.
5. Do not round numeric values; preserve raw_hex per frame for traceability.
6. Concatenate by WIT file number and original byte order, without resorting by time.

Default input:
/mnt/newdisk/Trigait/Trigait\u6570\u636e\u5e93_7.20\u76ee\u5f55\u793a\u8303_20260721/device_export

Default output:
/mnt/newdisk/Trigait/Trigait\u6570\u636e\u5e93_7.20\u76ee\u5f55\u793a\u8303_20260721/original_picture

Example output structure:
original_picture/
  sub_001/
    G/
      sub_001_G_WIT0.xlsx
      sub_001_G_WIT1.xlsx
      sub_001_G_merged.xlsx
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
INPUT_ROOT = "/mnt/newdisk/Trigait/Trigait\u6570\u636e\u5e93_7.20\u76ee\u5f55\u793a\u8303_20260721/device_export"
OUTPUT_ROOT = "/mnt/newdisk/Trigait/Trigait\u6570\u636e\u5e93_7.20\u76ee\u5f55\u793a\u8303_20260721/original_picture"

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
    "Merge index",
    "Source file",
    "Frame index in file",
    "Source byte offset",
    "Sync header valid",
    "Timestamp valid",
    "Raw timestamp string",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Second",
    "Millisecond",
    "X acceleration (g)",
    "Y acceleration (g)",
    "Z acceleration (g)",
    "X angular velocity (deg/s)",
    "Y angular velocity (deg/s)",
    "Z angular velocity (deg/s)",
    "X angle (deg)",
    "Y angle (deg)",
    "Z angle (deg)",
    "raw_hex",
]

UNPARSED_COLUMNS = [
    "Source file",
    "Start offset",
    "End offset (exclusive)",
    "Byte count",
    "Reason",
    "raw_hex",
]

INFO_COLUMNS = ["Field", "Value"]

HEADER_FONT = Font(name="Microsoft YaHei", bold=True, color="FFFFFF", size=11)
HEADER_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
HEADER_ALIGN = Alignment(horizontal="center", vertical="center")


def to_signed(raw: int) -> int:
    """Convert unsigned 16-bit integers to signed integers."""
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
    """Flag invalid timestamps without dropping frames."""
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
    """Parse a complete 28-byte frame, retaining frames with invalid timestamps."""
    if len(chunk) != RECORD_SIZE:
        raise ValueError(f"Frame length is not {RECORD_SIZE} bytes")

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
    Scan each file:
    - At 55 61 with at least 28 remaining bytes, parse a frame;
    - Collect all other bytes into an unparsed segment;
    - Trailing fragments smaller than 28 bytes are also unparsed.

    Yield: ("frame", record) or ("unparsed", segment)
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
        Return a list; chunk long invalid-byte sequences into multiple Excel rows to respect cell limits.
        Chunking affects only Excel presentation; bytes and byte order are preserved.
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
            pending_list = flush_unparsed(pos, "Malformed or misaligned raw bytes between sync headers")
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

    pending_list = flush_unparsed(total, "Trailing bytes shorter than 28 bytes or malformed/misaligned bytes")
    for pending in pending_list:
        yield "unparsed", pending


def is_wit_file(filename: str) -> bool:
    """Identify WIT0/WIT1/... with no extension or .txt/.bin/.dat."""
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
    """Automatically use additional worksheets when exceeding Excel row limits."""

    def __init__(self, output_path: str, title_prefix: str = "Data"):
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
        self.unparsed_ws = self.wb.create_sheet(title="Unparsed Bytes")
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
        self.info_ws = self.wb.create_sheet(title="File Info")
        self.info_ws.append([make_header_cell(self.info_ws, x) for x in INFO_COLUMNS])
        for key, value in info_items:
            self.info_ws.append([key, value])

    def save(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.wb.save(self.output_path)


def convert_one_wit_file(filepath: str, output_path: str, sub_name: str, sensor_name: str) -> Dict[str, int]:
    """One WIT file -> one Excel workbook."""
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
        ("Participant", sub_name),
        ("Sensor directory", sensor_name),
        ("Source file", filepath),
        ("Source file size (bytes)", str(file_size)),
        ("Complete frame count", str(writer.total_frames)),
        ("Unparsed segment count", str(writer.total_unparsed_segments)),
        ("Unparsed byte count", str(writer.total_unparsed_bytes)),
        ("Record size", f"{RECORD_SIZE} bytes/frame"),
        ("Sync header", SYNC_HEADER.hex(" ").upper()),
        ("Notes", "Raw bytes are retained even when timestamps are invalid or data are misaligned. Complete frames are stored as raw_hex; malformed or trailing bytes are stored in the Unparsed Bytes worksheet."),
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
    Process one G/L1/... directory:
    1) Generate one workbook per WIT file;
    2) Concatenate all WIT files in numeric order.
    """
    if not os.path.isdir(input_dir):
        print(f"  [Skipped] Directory does not exist: {input_dir}")
        return

    wit_files = [
        fn for fn in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, fn)) and is_wit_file(fn)
    ]
    wit_files.sort(key=wit_sort_key)

    if not wit_files:
        print(f"  [Skipped] No WIT files found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"  [{sensor_name}] found {len(wit_files)} WIT files")

    # Convert each source file first
    for fn in wit_files:
        src = os.path.join(input_dir, fn)
        wit_stem = os.path.splitext(fn)[0]
        out_name = f"{sub_name}_{sensor_name}_{wit_stem}.xlsx"
        out_path = os.path.join(output_dir, out_name)

        print(f"    Converting one file: {fn} -> {out_name}")
        stats = convert_one_wit_file(src, out_path, sub_name, sensor_name)
        print(
            f"      complete frames={stats['frames']}，"
            f"unparsed bytes={stats['unparsed_bytes']}，"
            f"source file={stats['file_bytes']} bytes"
        )

    # Then concatenate at the directory level
    merge_name = f"{sub_name}_{sensor_name}_merged.xlsx"
    merge_path = os.path.join(output_dir, merge_name)
    merge_writer = ExcelStreamWriter(merge_path)
    concat_index = 0
    total_source_bytes = 0

    for fn in wit_files:
        src = os.path.join(input_dir, fn)
        total_source_bytes += os.path.getsize(src)
        print(f"    Reading for merge: {fn}")

        for kind, obj in iter_file_content(src):
            if kind == "frame":
                concat_index += 1
                merge_writer.append_frame(obj, concat_index)
            else:
                merge_writer.append_unparsed(obj)

    merge_writer.add_info_sheet([
        ("Participant", sub_name),
        ("Sensor directory", sensor_name),
        ("Source directory", input_dir),
        ("Input file order", ", ".join(wit_files)),
        ("Source file count", str(len(wit_files))),
        ("Total source bytes", str(total_source_bytes)),
        ("Merged complete frame count", str(merge_writer.total_frames)),
        ("Unparsed segment count", str(merge_writer.total_unparsed_segments)),
        ("Unparsed byte count", str(merge_writer.total_unparsed_bytes)),
        ("Merge rule", "Files are concatenated in ascending WIT numeric order; original order within each file is preserved and records are not sorted by time."),
        ("Notes", "Raw bytes are retained even when timestamps are invalid or synchronization fails. Malformed bytes are stored in the Unparsed Bytes worksheet."),
    ])
    merge_writer.save()
    print(f"    Generated merged workbook: {merge_path}")


def main(input_root: str = INPUT_ROOT, output_root: str = OUTPUT_ROOT) -> None:
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)

    print("=" * 90)
    print("device_export batch conversion and concatenation")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Subject range: sub_{SUBJECT_START:03d} ~ sub_{SUBJECT_END:03d}")
    print(f"Sensor directories: {', '.join(TARGET_DIR_NAMES)}")
    print("Principle: preserve source files, frame order, exact values, and invalid bytes in Excel.")
    print("=" * 90)

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    os.makedirs(output_root, exist_ok=True)

    for subject_no in range(SUBJECT_START, SUBJECT_END + 1):
        sub_name = f"sub_{subject_no:03d}"
        sub_input = os.path.join(input_root, sub_name)
        sub_output = os.path.join(output_root, sub_name)

        print(f"\n{'#' * 90}")
        print(f"Processing {sub_name}")

        if not os.path.isdir(sub_input):
            print(f"[Skipped] Subject directory does not exist: {sub_input}")
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

    print("\nAll processing completed.")


if __name__ == "__main__":
    # Optional CLI arguments:
    # python convert_device_export_no_trim.py [input_root] [output_root]
    in_root = sys.argv[1] if len(sys.argv) >= 2 else INPUT_ROOT
    out_root = sys.argv[2] if len(sys.argv) >= 3 else OUTPUT_ROOT
    main(in_root, out_root)
