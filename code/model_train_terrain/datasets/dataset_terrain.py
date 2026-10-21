
import pickle
import numpy as np
import torch

from torch.utils.data import Dataset


SCENE_MAP = {

    "walk": 0,
    "stair": 1,

    "x": 2,
    "y": 3,

    "z1": 4,
    "z2": 5,
    "z3": 6,
    "z4": 7,

    "xy1": 8,
    "xy2": 9,
    "xy3": 10,
    "xy4": 11,
    "xy5": 12,
    "xy6": 13,

    "xyz1": 14,
    "xyz2": 15,
    "xyz3": 16,
}


def get_scene(task):

    parts = str(task).split("_")

    if len(parts) >= 4:
        return parts[2]

    return "unknown"



class TrigaitTerrainDataset(Dataset):

    """
    Trigait terrain regression only dataset

    Input:
        IMU history
        [200,42]

    Output:
        future terrain regression
        [30,6]

    Terrain:
        ipsilateral xyz
        contralateral xyz

    Removed:
        angle
        gait
        terrain_cls
        terrain_condition
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

        print(
            "Loading dataset:",
            pkl_path
        )

        with open(
            pkl_path,
            "rb"
        ) as f:
            self.data = pickle.load(f)

        self.samples = []

        self.build_index()

        print(
            "Total samples:",
            len(self.samples)
        )


    def build_index(self):

        for sub, tasks in self.data.items():

            for task, record in tasks.items():

                T = len(record["imu"])

                max_start = (
                    T
                    -
                    self.window_size
                    -
                    self.pred_len
                )

                if max_start <= 0:
                    continue


                for start in range(
                    0,
                    max_start,
                    self.stride
                ):

                    self.samples.append(
                        (
                            sub,
                            task,
                            start
                        )
                    )


    def __len__(self):

        return len(self.samples)



    def __getitem__(self,index):

        sub, task, start = self.samples[index]

        record = self.data[sub][task]


        # IMU

        imu = record["imu"][
            start:start+self.window_size
        ]

        imu = np.asarray(
            imu,
            dtype=np.float32
        )

        imu = np.nan_to_num(imu)

        imu = imu / 100.0



        # future terrain

        y1 = (
            start
            +
            self.window_size
        )

        y2 = (
            y1
            +
            self.pred_len
        )


        ipsi = record[
            "terrain_ipsilateral"
        ][y1:y2]


        contra = record[
            "terrain_contralateral"
        ][y1:y2]


        terrain = np.concatenate(
            [
                ipsi,
                contra
            ],
            axis=1
        )



        ipsi_mask = record[
            "terrain_ipsilateral_mask"
        ][y1:y2]


        contra_mask = record[
            "terrain_contralateral_mask"
        ][y1:y2]


        terrain_mask = np.concatenate(
            [
                ipsi_mask,
                contra_mask
            ],
            axis=1
        )



        terrain = np.nan_to_num(
            terrain
        )

        terrain_mask = np.nan_to_num(
            terrain_mask
        )


        scene = get_scene(task)

        scene_id = SCENE_MAP.get(
            scene,
            -1
        )


        return {

            "imu":
                torch.tensor(
                    imu,
                    dtype=torch.float32
                ),

            "terrain":
                torch.tensor(
                    terrain,
                    dtype=torch.float32
                ),

            "terrain_mask":
                torch.tensor(
                    terrain_mask,
                    dtype=torch.float32
                ),

            "scene_id":
                torch.tensor(
                    scene_id,
                    dtype=torch.long
                ),

            "subject":
                sub,

            "task":
                task
        }



if __name__ == "__main__":

    pkl = (
        "/mnt/newdisk/dateset_0909/"
        "code/model_train_terrain/"
        "splits/subject_split/train.pkl"
    )


    dataset = TrigaitTerrainDataset(
        pkl_path=pkl,
        window_size=200,
        pred_len=30,
        stride=10
    )


    sample = dataset[0]


    print("\nSample output:")

    for k,v in sample.items():

        if torch.is_tensor(v):

            print(
                k,
                v.shape,
                v.dtype
            )

        else:

            print(
                k,
                v
            )
