# Seven-IMU absolute-time alignment v6
# Do not restrict the data to the common time interval or trim the first three seconds.
# Build a 100 Hz timeline from the union of all IMU timestamps.

import os
import argparse
import numpy as np
import pandas as pd

SENSORS = [
    "pelvis","l_thigh","l_shank","l_foot",
    "r_thigh","r_shank","r_foot"
]


def clean_timestamp(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    return df


def remove_large_jump_before(df, threshold=0.1):
    t = df["timestamp"].astype("int64") / 1e9
    dt = np.diff(t.values)
    idx = np.where(dt > threshold)[0]
    if len(idx):
        df = df.iloc[idx[0]+1:].copy()
    return df


def resample_100hz(df):
    df = df.set_index("timestamp")
    cols = df.select_dtypes(include=[np.number]).columns
    df = df[cols].resample("10ms").mean()
    df = df.interpolate(method="time", limit_direction="both")
    return df.reset_index()


def find_sensor_file(folder, sensor):
    sensor = sensor.lower()
    for f in os.listdir(folder):
        name = f.lower()
        if "imu" in name and sensor in name and f.endswith(".xlsx"):
            return os.path.join(folder, f)
    return None


def process_sensor(path, threshold):
    df = pd.read_excel(path)
    qc = {"raw": len(df)}
    df = clean_timestamp(df)
    qc["clean"] = len(df)
    df = remove_large_jump_before(df, threshold)
    qc["after_jump"] = len(df)
    df = resample_100hz(df)
    qc["100hz"] = len(df)
    return df, qc


def process_task(input_root, output_root, subject, task, threshold):

    folder = os.path.join(input_root, subject, "imu", task)

    data = {}
    qc = {}

    for sensor in SENSORS:
        f = find_sensor_file(folder, sensor)
        if f is None:
            raise FileNotFoundError(f"missing {sensor}")

        df, info = process_sensor(f, threshold)

        df = df.rename(columns={
            c: f"{sensor}_{c}"
            for c in df.columns
            if c != "timestamp"
        })

        data[sensor] = df
        qc[sensor] = info

    # Use the union of sensor timestamps so valid data outside the common interval is retained.
    timeline = pd.DataFrame({
        "timestamp": pd.date_range(
            min(x.timestamp.min() for x in data.values()),
            max(x.timestamp.max() for x in data.values()),
            freq="10ms"
        )
    })

    result = timeline

    for sensor, df in data.items():
        result = pd.merge(
            result, df,
            on="timestamp",
            how="left"
        )

    result["time_s"] = (
        result["timestamp"] -
        result["timestamp"].iloc[0]
    ).dt.total_seconds()

    out_dir = os.path.join(output_root, subject)
    os.makedirs(out_dir, exist_ok=True)

    out = os.path.join(
        out_dir,
        f"{subject}_{task}_data.xlsx"
    )

    with pd.ExcelWriter(out) as writer:
        result.to_excel(writer, sheet_name="data", index=False)
        pd.DataFrame(qc).T.to_excel(writer, sheet_name="qc")

        pd.DataFrame({
            "parameter":[
                "alignment",
                "resample",
                "common_time_crop",
                "motion_trim"
            ],
            "value":[
                "absolute timestamp",
                "100Hz",
                "disabled",
                "disabled"
            ]
        }).to_excel(writer, sheet_name="processing", index=False)

    print("[OK]", out)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gap-threshold", type=float, default=0.1)

    args = parser.parse_args()

    process_task(
        args.input_root,
        args.output_root,
        f"sub_{args.subject:03d}",
        args.task,
        args.gap_threshold
    )


if __name__ == "__main__":
    main()
