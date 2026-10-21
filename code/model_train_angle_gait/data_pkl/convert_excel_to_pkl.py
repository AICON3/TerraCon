import os
import pickle
import argparse
import pandas as pd
import numpy as np


# ======================================================
# Path
# ======================================================

BASE_DIR = (
    "/mnt/newdisk/"
    "dateset_0909"
)


DEFAULT_INPUT = os.path.join(
    BASE_DIR,
    "processed"
)


DEFAULT_OUTPUT = os.path.join(
    BASE_DIR,
    "model_train",
    "data_pkl",
    "multitask_dataset.pkl"
)



# ======================================================
# IMU 42 channels
# ======================================================

IMU_COLUMNS = [

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



# ======================================================
# Angle target
# ======================================================


ANGLE_COLUMNS = [

    "l_thigh_angle_x_deg",
    "l_thigh_angle_y_deg",

    "l_shank_angle_x_deg",
    "l_foot_angle_y_deg",

    "r_thigh_angle_x_deg",
    "r_thigh_angle_y_deg",

    "r_shank_angle_x_deg",
    "r_foot_angle_y_deg",

]



# ======================================================
# utils
# ======================================================


def read_excel(path):

    if not os.path.exists(path):

        return None

    return pd.read_excel(path)



def get_task_from_filename(filename):

    return filename.replace(
        "_data.xlsx",
        ""
    )



def build_terrain(df,prefix):


    value=np.zeros(
        (len(df),3),
        dtype=np.float32
    )


    mask=np.zeros(
        (len(df),3),
        dtype=np.float32
    )


    cols=[

        f"{prefix}_x_label",
        f"{prefix}_y_label",
        f"{prefix}_z_label"

    ]


    for i,c in enumerate(cols):

        if c in df.columns:

            arr=df[c].fillna(0).values


            value[:,i]=arr


            mask[:,i]=(
                np.abs(arr)>1e-6
            )


    return value,mask



# ======================================================
# process one task
# ======================================================


def process_task(
        sub_dir,
        data_file,
        missing_report
):


    subject=os.path.basename(
        sub_dir
    )


    task=get_task_from_filename(
        data_file
    )


    data_path=os.path.join(
        sub_dir,
        "data",
        data_file
    )


    data_df=pd.read_excel(
        data_path
    )



    imu=data_df[
        IMU_COLUMNS
    ].values.astype(
        np.float32
    )


    time_s=data_df[
        "time_s"
    ].values.astype(
        np.float32
    )


    angle=data_df[
        ANGLE_COLUMNS
    ].values.astype(
        np.float32
    )



    labels_dir=os.path.join(
        sub_dir,
        "labels"
    )


    # -------------------------
    # gait
    # -------------------------

    gait_path=os.path.join(

        labels_dir,

        "label_gait",

        f"{task}_label_gait.xlsx"

    )


    gait_df=read_excel(
        gait_path
    )


    if gait_df is None:


        missing_report.append(
            [
                subject,
                task,
                "gait"
            ]
        )


        gait=np.zeros(
            (len(data_df),2),
            dtype=np.float32
        )


    else:


        gait=gait_df[

            [
                "l_label_gait",
                "r_label_gait"
            ]

        ].values.astype(
            np.float32
        )




    # -------------------------
    # terrain ipsi
    # -------------------------

    ipsi_path=os.path.join(

        labels_dir,

        "label_terrain_ipsilateral",

        f"{task}_label_terrain_ipsilateral.xlsx"

    )


    ipsi_df=read_excel(
        ipsi_path
    )


    if ipsi_df is None:


        missing_report.append(
            [
                subject,
                task,
                "terrain_ipsilateral"
            ]
        )


        terrain_ipsi=np.zeros(
            (len(data_df),3),
            dtype=np.float32
        )


        mask_ipsi=np.zeros(
            (len(data_df),3),
            dtype=np.float32
        )


    else:


        terrain_ipsi,mask_ipsi=build_terrain(
            ipsi_df,
            "terrain_ipsilateral"
        )




    # -------------------------
    # terrain contra
    # -------------------------

    contra_path=os.path.join(

        labels_dir,

        "label_terrain_contralateral",

        f"{task}_label_terrain_contralateral.xlsx"

    )


    contra_df=read_excel(
        contra_path
    )


    if contra_df is None:


        missing_report.append(
            [
                subject,
                task,
                "terrain_contralateral"
            ]
        )


        terrain_contra=np.zeros(
            (len(data_df),3),
            dtype=np.float32
        )


        mask_contra=np.zeros(
            (len(data_df),3),
            dtype=np.float32
        )


    else:


        terrain_contra,mask_contra=build_terrain(
            contra_df,
            "terrain_contralateral"
        )




    return {


        "imu":imu,

        "time_s":time_s,

        "angle":angle,

        "gait":gait,


        "terrain_ipsilateral":
            terrain_ipsi,


        "terrain_ipsilateral_mask":
            mask_ipsi,


        "terrain_contralateral":
            terrain_contra,


        "terrain_contralateral_mask":
            mask_contra,


    }




# ======================================================
# build dataset
# ======================================================


def build_dataset(input_root):


    dataset={}

    missing_report=[]



    subjects=sorted(

        [

            x for x in os.listdir(input_root)

            if x.startswith("sub_")

            and os.path.isdir(
                os.path.join(
                    input_root,
                    x
                )
            )

        ]

    )



    print(
        "Found subjects:",
        len(subjects)
    )



    for sub in subjects:


        print(
            "\nProcessing:",
            sub
        )


        sub_dir=os.path.join(
            input_root,
            sub
        )


        dataset[sub]={}



        data_dir=os.path.join(
            sub_dir,
            "data"
        )


        if not os.path.exists(data_dir):

            print(
                "WARNING:",
                sub,
                "missing data folder"
            )

            continue



        files=sorted(

            [

                f for f in os.listdir(data_dir)

                if f.endswith(
                    "_data.xlsx"
                )

            ]

        )



        for f in files:


            try:


                task=get_task_from_filename(
                    f
                )


                dataset[sub][task]=process_task(

                    sub_dir,

                    f,

                    missing_report

                )


            except Exception as e:


                print(
                    "ERROR:",
                    sub,
                    f,
                    e
                )



        if len(dataset[sub])==0:

            print(
                "WARNING:",
                sub,
                "has no task"
            )



    return dataset,missing_report




# ======================================================
# main
# ======================================================


if __name__=="__main__":


    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT
    )


    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT
    )


    args=parser.parse_args()



    dataset,missing=build_dataset(
        args.input
    )



    os.makedirs(

        os.path.dirname(
            args.output
        ),

        exist_ok=True

    )



    with open(
        args.output,
        "wb"
    ) as f:


        pickle.dump(
            dataset,
            f
        )



    missing_file=os.path.join(

        os.path.dirname(
            args.output
        ),

        "missing_report.csv"

    )


    pd.DataFrame(

        missing,

        columns=[

            "subject",
            "task",
            "missing"

        ]

    ).to_csv(

        missing_file,

        index=False

    )



    print("\n====================")

    print(
        "Finished"
    )


    print(
        "Subjects:",
        len(dataset)
    )


    print(
        "Tasks:",
        sum(
            len(v)
            for v in dataset.values()
        )
    )


    print(
        "Saved:",
        args.output
    )


    print(
        "Missing:",
        missing_file
    )