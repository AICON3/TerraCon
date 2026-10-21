import os
import pickle
import random
import argparse
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold


# =====================================================
# Path
# =====================================================

BASE_DIR = "/mnt/newdisk/dateset_0909"


DEFAULT_INPUT = "/mnt/newdisk/dateset_0909/code/model_train_terrain/data_pkl/multitask_dataset.pkl"


DEFAULT_OUTPUT = os.path.join(
    BASE_DIR,
    "code",
    "model_train_terrain",
    "splits"
)


RANDOM_SEED = 2026



# =====================================================
# Scene definitions
# =====================================================

# 基础/单方向场景
ONE_D_SCENES = [

    "walk",
    "stair",

    "x",
    "y",

    "z1",
    "z2",
    "z3",
    "z4",

]


# 二维组合场景
XY_SCENES = [

    "xy1",
    "xy2",
    "xy3",
    "xy4",
    "xy5",
    "xy6",

]


# 三维组合场景
XYZ_SCENES = [

    "xyz1",
    "xyz2",
    "xyz3",

]


# 所有已定义地形场景
ALL_SCENES = (

    ONE_D_SCENES
    +
    XY_SCENES
    +
    XYZ_SCENES

)



# =====================================================
# Basic utils
# =====================================================

def get_scene(task):
    """
    从 task 名称解析 scene。

    例如:
        sub_001_x_fwd
        -> x

        sub_001_xy3_bwd
        -> xy3

        sub_001_xyz2_fwd
        -> xyz2
    """

    parts = str(task).split("_")

    if len(parts) >= 4:
        return parts[2]

    return None



def count_tasks(data):
    """
    统计 split 中任务文件数量。
    """

    return sum(
        len(tasks)
        for tasks in data.values()
    )



def collect_scene_counts(data):
    """
    统计一个 split 中各 scene 的任务文件数量。
    """

    counts = Counter()

    for sub, tasks in data.items():

        for task in tasks.keys():

            scene = get_scene(task)

            counts[scene] += 1

    return counts



def print_split_summary(
        data,
        name
):
    """
    打印某个 split 的 subject / task / scene 数量。
    """

    print("\n----------------------------------------")
    print("Split:", name)
    print("----------------------------------------")

    print(
        "Subjects:",
        len(data)
    )

    print(
        "Tasks:",
        count_tasks(data)
    )

    scene_counts = collect_scene_counts(data)

    print(
        "Scenes:",
        dict(
            sorted(
                scene_counts.items(),
                key=lambda x: str(x[0])
            )
        )
    )



# =====================================================
# Save split
# =====================================================

def save_split(
        data,
        folder,
        name
):
    """
    保存:
        train.pkl / val.pkl / test.pkl

    同时保存:
        train_list.csv / val_list.csv / test_list.csv
    """

    os.makedirs(
        folder,
        exist_ok=True
    )


    # -------------------------
    # 保存 pkl
    # -------------------------

    pkl_path = os.path.join(
        folder,
        name + ".pkl"
    )


    with open(
        pkl_path,
        "wb"
    ) as f:

        pickle.dump(
            data,
            f
        )


    # -------------------------
    # 保存 CSV 索引
    # -------------------------

    rows = []


    for sub, tasks in data.items():

        for task in tasks.keys():

            rows.append(
                {
                    "subject": sub,
                    "task": task,
                    "scene": get_scene(task)
                }
            )


    csv_path = os.path.join(
        folder,
        name + "_list.csv"
    )


    pd.DataFrame(
        rows,
        columns=[
            "subject",
            "task",
            "scene"
        ]
    ).to_csv(
        csv_path,
        index=False
    )


    print_split_summary(
        data,
        name
    )


    print(
        "Saved PKL:",
        pkl_path
    )

    print(
        "Saved CSV:",
        csv_path
    )



# =====================================================
# Dataset validation
# =====================================================

def validate_dataset_structure(dataset):
    """
    对输入 multitask_dataset_terrain.pkl 做基本检查。

    主要确认:
    1. dataset 非空
    2. 至少存在一个 subject / task
    3. terrain 回归标签是否已经进入 pkl
    """

    if not isinstance(
        dataset,
        dict
    ):

        raise TypeError(
            "dataset must be dict"
        )


    if len(dataset) == 0:

        raise ValueError(
            "dataset is empty"
        )


    sample_sub = next(
        iter(dataset.keys())
    )


    if len(
        dataset[sample_sub]
    ) == 0:

        raise ValueError(
            f"{sample_sub} has no tasks"
        )


    sample_task = next(
        iter(
            dataset[sample_sub].keys()
        )
    )


    sample = dataset[
        sample_sub
    ][
        sample_task
    ]


    print("\n========================================")
    print("Dataset structure check")
    print("========================================")

    print(
        "Sample subject:",
        sample_sub
    )

    print(
        "Sample task:",
        sample_task
    )

    print(
        "Sample keys:"
    )

    print(
        list(sample.keys())
    )


    expected_keys = [

        "imu",

        "time_s",

        "angle",

        "gait",

        "terrain_ipsilateral",

        "terrain_ipsilateral_mask",

        "terrain_contralateral",

        "terrain_contralateral_mask",

    ]


    missing_keys = [

        key
        for key in expected_keys

        if key not in sample

    ]


    if len(
        missing_keys
    ) > 0:

        print(
            "\nWARNING: sample missing keys:"
        )

        for key in missing_keys:

            print(
                "  -",
                key
            )

    else:

        print(
            "\nAll expected regression/classification keys found."
        )



# =====================================================
# Subject split
# =====================================================

def subject_split(dataset):
    """
    跨受试者划分。

    Train:
        sub_001 ~ sub_034
        全部 scene

    Val:
        sub_035 ~ sub_041
        全部 scene

    Test:
        sub_042 ~ sub_048
        全部 scene
    """

    train = {}

    val = {}

    test = {}


    for sub, tasks in dataset.items():

        sid = int(
            sub.split("_")[1]
        )


        if sid <= 34:

            train[sub] = tasks


        elif sid <= 41:

            val[sub] = tasks


        else:

            test[sub] = tasks


    return (
        train,
        val,
        test
    )



# =====================================================
# Subject + Scene split
# =====================================================

def subject_scene_split(dataset):
    """
    同时测试:
        新受试者 + 新三维场景

    Train:
        sub_001 ~ sub_034
        ONE_D_SCENES + XY_SCENES

    Val:
        sub_035 ~ sub_041
        ONE_D_SCENES + XY_SCENES

    Test:
        sub_042 ~ sub_048
        XYZ_SCENES

    注意:
        Test 同时换 subject 和 scene。
    """

    train = {}

    val = {}

    test = {}


    train_scenes = (

        ONE_D_SCENES
        +
        XY_SCENES

    )


    test_scenes = (

        XYZ_SCENES

    )


    for sub, tasks in dataset.items():

        sid = int(
            sub.split("_")[1]
        )


        for task, value in tasks.items():

            scene = get_scene(task)


            if (
                sid <= 34
                and
                scene in train_scenes
            ):

                train.setdefault(
                    sub,
                    {}
                )[task] = value


            elif (
                35 <= sid <= 41
                and
                scene in train_scenes
            ):

                val.setdefault(
                    sub,
                    {}
                )[task] = value


            elif (
                sid >= 42
                and
                scene in test_scenes
            ):

                test.setdefault(
                    sub,
                    {}
                )[task] = value


    return (
        train,
        val,
        test
    )



# =====================================================
# Generic Scene split
# =====================================================

def scene_split_generator(
        dataset,
        train_scenes,
        test_scenes,
        val_ratio=0.10,
        seed=2026
):
    """
    通用 scene 划分函数。

    特点:
        - 所有受试者都可以进入 train / val / test
        - scene 决定属于 train 还是 test
        - val 从 train scenes 中按每个 subject 随机抽取一定比例

    这样验证集仍来自“已见 scene”分布。
    """

    train = {}

    val = {}

    test = {}


    # -------------------------
    # 先按 scene 分 train/test
    # -------------------------

    for sub, tasks in dataset.items():

        for task, value in tasks.items():

            scene = get_scene(task)


            if scene in train_scenes:

                train.setdefault(
                    sub,
                    {}
                )[task] = value


            elif scene in test_scenes:

                test.setdefault(
                    sub,
                    {}
                )[task] = value


    # -------------------------
    # 从 train 中抽 val
    # -------------------------

    rng = random.Random(
        seed
    )


    for sub in list(
        train.keys()
    ):

        tasks = list(
            train[sub].keys()
        )


        if len(
            tasks
        ) == 0:

            continue


        rng.shuffle(
            tasks
        )


        n_val = max(
            1,
            int(
                len(tasks)
                *
                val_ratio
            )
        )


        # 避免把该 subject 的 train 全抽空
        if (
            len(tasks) > 1
            and
            n_val >= len(tasks)
        ):

            n_val = (
                len(tasks)
                -
                1
            )


        selected = tasks[
            :n_val
        ]


        for task in selected:

            val.setdefault(
                sub,
                {}
            )[task] = train[
                sub
            ][
                task
            ]


            del train[
                sub
            ][
                task
            ]


        if len(
            train[sub]
        ) == 0:

            del train[sub]


    return (
        train,
        val,
        test
    )



# =====================================================
# Scene Split 1
# =====================================================

def scene_split_1(dataset):
    """
    单方向/基础地形 -> 组合地形

    Train scenes:
        walk
        stair
        x
        y
        z1-z4

    Val:
        从上述 train scenes 中按每个 subject 抽 10%

    Test scenes:
        xy1-xy6
        xyz1-xyz3

    目的:
        测试从基础/单方向约束泛化到二维和三维组合约束。
    """

    train_scenes = (

        ONE_D_SCENES

    )


    test_scenes = (

        XY_SCENES
        +
        XYZ_SCENES

    )


    return scene_split_generator(
        dataset=dataset,
        train_scenes=train_scenes,
        test_scenes=test_scenes,
        val_ratio=0.10,
        seed=RANDOM_SEED + 1
    )



# =====================================================
# Scene Split 2
# =====================================================

def scene_split_2(dataset):
    """
    基础场景 + 部分 XY -> 新 XY + XYZ

    Train scenes:
        walk
        stair
        x
        y
        z1-z4
        xy1
        xy2
        xy3

    Val:
        从上述 train scenes 中抽 10%

    Test scenes:
        xy4
        xy5
        xy6
        xyz1
        xyz2
        xyz3

    目的:
        测试学习部分二维组合后，
        对未见二维组合和三维组合的泛化。
    """

    train_scenes = (

        ONE_D_SCENES
        +
        [
            "xy1",
            "xy2",
            "xy3",
        ]

    )


    test_scenes = (

        [
            "xy4",
            "xy5",
            "xy6",
        ]
        +
        XYZ_SCENES

    )


    return scene_split_generator(
        dataset=dataset,
        train_scenes=train_scenes,
        test_scenes=test_scenes,
        val_ratio=0.10,
        seed=RANDOM_SEED + 2
    )



# =====================================================
# Scene Split 3
# =====================================================

def scene_split_3(dataset):
    """
    二维组合内部泛化。

    Train scenes:
        walk
        stair
        x
        y
        z1-z4
        xy1
        xy2
        xy3

    Val:
        从上述 train scenes 中抽 10%

    Test scenes:
        xy4
        xy5
        xy6

    XYZ 不参与该 split 的 test。

    目的:
        专门检查模型是否学习了 XY 组合规律。
    """

    train_scenes = (

        ONE_D_SCENES
        +
        [
            "xy1",
            "xy2",
            "xy3",
        ]

    )


    test_scenes = [

        "xy4",
        "xy5",
        "xy6",

    ]


    return scene_split_generator(
        dataset=dataset,
        train_scenes=train_scenes,
        test_scenes=test_scenes,
        val_ratio=0.10,
        seed=RANDOM_SEED + 3
    )



# =====================================================
# Scene Split 4
# =====================================================

def scene_split_4(dataset):
    """
    一维 + 全部二维 -> 三维

    Train scenes:
        walk
        stair
        x
        y
        z1-z4
        xy1-xy6

    Val:
        从上述 train scenes 中抽 10%

    Test scenes:
        xyz1
        xyz2
        xyz3

    目的:
        作为主要 terrain scene generalization 实验，
        检查模型能否从一维和二维约束
        泛化到未见三维组合约束。
    """

    train_scenes = (

        ONE_D_SCENES
        +
        XY_SCENES

    )


    test_scenes = (

        XYZ_SCENES

    )


    return scene_split_generator(
        dataset=dataset,
        train_scenes=train_scenes,
        test_scenes=test_scenes,
        val_ratio=0.10,
        seed=RANDOM_SEED + 4
    )



# =====================================================
# Subject 5-fold split
# =====================================================

def subject_5fold_split(
        dataset,
        output_dir
):
    """
    Subject-level 5-fold。

    每个 fold:
        test subjects:
            KFold 当前 test fold

        trainval subjects:
            其余 subjects

        val subjects:
            trainval 中随机约 15%

        train subjects:
            trainval 剩余 subjects

    保证同一 subject 不同时出现在 train/val/test。
    """

    subjects = np.array(
        sorted(
            dataset.keys()
        )
    )


    if len(
        subjects
    ) < 5:

        raise ValueError(
            "Need at least 5 subjects for 5-fold split."
        )


    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED
    )


    for fold, (
        trainval_idx,
        test_idx
    ) in enumerate(
        kf.split(
            subjects
        ),
        start=1
    ):

        print(
            "\n========================================"
        )

        print(
            "Subject 5-fold:",
            fold
        )

        print(
            "========================================"
        )


        test_subjects = subjects[
            test_idx
        ]


        trainval_subjects = list(
            subjects[
                trainval_idx
            ]
        )


        rng = random.Random(
            RANDOM_SEED
            +
            100
            +
            fold
        )


        rng.shuffle(
            trainval_subjects
        )


        val_num = max(
            1,
            int(
                len(trainval_subjects)
                *
                0.15
            )
        )


        if (
            len(trainval_subjects) > 1
            and
            val_num >= len(trainval_subjects)
        ):

            val_num = (
                len(trainval_subjects)
                -
                1
            )


        val_subjects = trainval_subjects[
            :val_num
        ]


        train_subjects = trainval_subjects[
            val_num:
        ]


        train = {

            sub:
                dataset[sub]

            for sub in train_subjects

        }


        val = {

            sub:
                dataset[sub]

            for sub in val_subjects

        }


        test = {

            sub:
                dataset[sub]

            for sub in test_subjects

        }


        fold_dir = os.path.join(
            output_dir,
            "subject_5fold_split",
            f"fold{fold}"
        )


        save_split(
            train,
            fold_dir,
            "train"
        )


        save_split(
            val,
            fold_dir,
            "val"
        )


        save_split(
            test,
            fold_dir,
            "test"
        )


        print(
            "\nFold subject counts:"
        )

        print(
            "Train subjects:",
            len(train)
        )

        print(
            "Val subjects:",
            len(val)
        )

        print(
            "Test subjects:",
            len(test)
        )



# =====================================================
# Save standard three-way split
# =====================================================

def save_three_way_split(
        dataset,
        split_name,
        split_func,
        output_root
):
    """
    执行一个 train/val/test split 并保存。
    """

    print(
        "\n\n========================================"
    )

    print(
        "Running:",
        split_name
    )

    print(
        "========================================"
    )


    train, val, test = split_func(
        dataset
    )


    folder = os.path.join(
        output_root,
        split_name
    )


    save_split(
        train,
        folder,
        "train"
    )


    save_split(
        val,
        folder,
        "val"
    )


    save_split(
        test,
        folder,
        "test"
    )



# =====================================================
# Main
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate subject and scene splits "
            "for Trigait terrain regression dataset."
        )
    )


    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input multitask_dataset_terrain.pkl"
    )


    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output split directory"
    )


    args = parser.parse_args()


    # -------------------------
    # 检查输入文件
    # -------------------------

    if not os.path.exists(
        args.input
    ):

        raise FileNotFoundError(
            f"Input pkl does not exist: {args.input}"
        )


    print(
        "Loading:",
        args.input
    )


    # -------------------------
    # 读取 dataset
    # -------------------------

    with open(
        args.input,
        "rb"
    ) as f:

        dataset = pickle.load(
            f
        )


    print(
        "Subjects:",
        len(dataset)
    )


    print(
        "Total tasks:",
        sum(
            len(tasks)
            for tasks in dataset.values()
        )
    )


    # -------------------------
    # 检查 dataset 内容
    # -------------------------

    validate_dataset_structure(
        dataset
    )


    # -------------------------
    # 输出目录
    # -------------------------

    os.makedirs(
        args.output,
        exist_ok=True
    )


    # =================================================
    # 1. Subject split
    # =================================================

    save_three_way_split(
        dataset=dataset,
        split_name="subject_split",
        split_func=subject_split,
        output_root=args.output
    )


    # =================================================
    # 2. Subject + Scene split
    # =================================================

    save_three_way_split(
        dataset=dataset,
        split_name="subject_scene_split",
        split_func=subject_scene_split,
        output_root=args.output
    )


    # =================================================
    # 3. Scene split 1
    # =================================================

    save_three_way_split(
        dataset=dataset,
        split_name="scene_split_1",
        split_func=scene_split_1,
        output_root=args.output
    )


    # =================================================
    # 4. Scene split 2
    # =================================================

    save_three_way_split(
        dataset=dataset,
        split_name="scene_split_2",
        split_func=scene_split_2,
        output_root=args.output
    )


    # =================================================
    # 5. Scene split 3
    # =================================================

    save_three_way_split(
        dataset=dataset,
        split_name="scene_split_3",
        split_func=scene_split_3,
        output_root=args.output
    )


    # =================================================
    # 6. Scene split 4
    # =================================================

    save_three_way_split(
        dataset=dataset,
        split_name="scene_split_4",
        split_func=scene_split_4,
        output_root=args.output
    )


    # =================================================
    # 7. Subject 5-fold
    # =================================================

    print(
        "\n\n========================================"
    )

    print(
        "Running: subject_5fold_split"
    )

    print(
        "========================================"
    )


    subject_5fold_split(
        dataset=dataset,
        output_dir=args.output
    )


    # =================================================
    # Finished
    # =================================================

    print(
        "\n\n========================================"
    )

    print(
        "All split generation finished."
    )

    print(
        "Output:",
        args.output
    )

    print(
        "Generated split groups:"
    )

    print(
        "  - subject_split"
    )

    print(
        "  - subject_scene_split"
    )

    print(
        "  - subject_5fold_split"
    )

    print(
        "  - scene_split_1"
    )

    print(
        "  - scene_split_2"
    )

    print(
        "  - scene_split_3"
    )

    print(
        "  - scene_split_4"
    )

    print(
        "========================================"
    )



if __name__ == "__main__":

    main()
