
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class TrigaitMultiTaskDataset(Dataset):
    """
    Trigait multi-task dataset.

    Input:
        imu: [200, 42]

    Future prediction targets:
        angle: [30, 8]
        gait:  [30, 2]

    Additional observations for Persistence baseline:
        angle_last: [8]
        gait_last:  [2]

    Expected pickle structure:
        {
            subject_id: {
                task_name: {
                    "imu": ...,
                    "angle": ...,
                    "gait": ...
                }
            }
        }
    """

    def __init__(
        self,
        pkl_path,
        window_size=200,
        pred_len=30,
        stride=10
    ):
        self.window_size = window_size
        self.pred_len = pred_len
        self.stride = stride

        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(
                "Dataset file not found: {}".format(pkl_path)
            )

        print("Loading dataset:", pkl_path)

        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.samples = []
        self.build_index()

        print("Total samples:", len(self.samples))

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid sliding-window samples were generated. "
                "Check sequence lengths, dataset structure, "
                "window_size, pred_len, and stride."
            )

    # ==================================================
    # Build sliding-window index
    # ==================================================

    def build_index(self):

        for sub, tasks in self.data.items():

            for task, record in tasks.items():

                if not all(
                    key in record
                    for key in ("imu", "angle", "gait")
                ):
                    print(
                        "Warning: missing required field; "
                        "skipping subject={}, task={}".format(
                            sub, task
                        )
                    )
                    continue

                imu = np.asarray(record["imu"])
                angle = np.asarray(record["angle"])
                gait = np.asarray(record["gait"])

                # All signals must have the same time length.
                if not (
                    len(imu) == len(angle) == len(gait)
                ):
                    print(
                        "Warning: inconsistent sequence lengths; "
                        "skipping subject={}, task={}".format(
                            sub, task
                        )
                    )
                    continue

                total_length = len(imu)

                # Need 200 input frames and 30 future frames.
                # start + window_size + pred_len must not exceed T.
                max_start = (
                    total_length
                    - self.window_size
                    - self.pred_len
                )

                if max_start < 0:
                    continue

                # Include the last valid window.
                for start in range(
                    0,
                    max_start + 1,
                    self.stride
                ):
                    self.samples.append(
                        (sub, task, start)
                    )

    def __len__(self):
        return len(self.samples)

    # ==================================================
    # Convert one index into a sample
    # ==================================================

    def __getitem__(self, index):

        sub, task, start = self.samples[index]

        record = self.data[sub][task]

        # --------------------------------------------------
        # 1. Historical IMU input
        # --------------------------------------------------

        x1 = start
        x2 = start + self.window_size

        imu = np.asarray(
            record["imu"][x1:x2],
            dtype=np.float32
        )

        imu = np.nan_to_num(
            imu,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Preserve the original normalization.
        imu = imu / 100.0

        # --------------------------------------------------
        # 2. Future prediction targets
        # --------------------------------------------------

        y1 = x2
        y2 = x2 + self.pred_len

        angle = np.asarray(
            record["angle"][y1:y2],
            dtype=np.float32
        )

        gait = np.asarray(
            record["gait"][y1:y2],
            dtype=np.float32
        )

        angle = np.nan_to_num(
            angle,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        gait = np.nan_to_num(
            gait,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # --------------------------------------------------
        # 3. Last observed values for Persistence baseline
        # --------------------------------------------------
        # x2 - 1 is the last frame of the input window.
        # It is not part of the future prediction interval.

        last_idx = x2 - 1

        angle_last = np.asarray(
            record["angle"][last_idx],
            dtype=np.float32
        )

        gait_last = np.asarray(
            record["gait"][last_idx],
            dtype=np.float32
        )

        angle_last = np.nan_to_num(
            angle_last,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        gait_last = np.nan_to_num(
            gait_last,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # --------------------------------------------------
        # 4. Shape validation
        # --------------------------------------------------

        if imu.shape != (self.window_size, 42):
            raise ValueError(
                "Unexpected IMU shape for subject={}, task={}, "
                "start={}: {}".format(
                    sub, task, start, imu.shape
                )
            )

        if angle.shape[0] != self.pred_len:
            raise ValueError(
                "Unexpected angle sequence length: {}".format(
                    angle.shape
                )
            )

        if gait.shape != (self.pred_len, 2):
            raise ValueError(
                "Unexpected gait shape: {}".format(
                    gait.shape
                )
            )

        # --------------------------------------------------
        # 5. Return sample
        # --------------------------------------------------

        return {
            "imu": torch.tensor(
                imu,
                dtype=torch.float32
            ),

            "angle": torch.tensor(
                angle,
                dtype=torch.float32
            ),

            "gait": torch.tensor(
                gait,
                dtype=torch.float32
            ),

            "angle_last": torch.tensor(
                angle_last,
                dtype=torch.float32
            ),

            "gait_last": torch.tensor(
                gait_last,
                dtype=torch.float32
            ),

            "subject": sub,
            "task": task
        }


# ==================================================
# Standalone test
# ==================================================

if __name__ == "__main__":

    BASE_DIR = (
        "/mnt/newdisk/dateset_0909/"
        "code/model_train_angle_gait"
    )

    # Change this path to test another split.
    pkl_path = os.path.join(
        BASE_DIR,
        "splits",
        "scene_split_5",
        "train.pkl"
    )

    dataset = TrigaitMultiTaskDataset(
        pkl_path,
        window_size=200,
        pred_len=30,
        stride=10
    )

    sample = dataset[0]

    print("\nSample information:")

    for key, value in sample.items():

        if torch.is_tensor(value):
            print(key, tuple(value.shape))
        else:
            print(key, value)
