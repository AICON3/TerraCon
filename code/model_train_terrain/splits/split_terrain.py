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

# Basic or single-direction scenes
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


# Two-dimensional combined scenes
XY_SCENES = [

    "xy1",
    "xy2",
    "xy3",
    "xy4",
    "xy5",
    "xy6",

]


# Three-dimensional combined scenes
XYZ_SCENES = [

    "xyz1",
    "xyz2",
    "xyz3",

]


# All defined terrain scenes
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
    Parse scene from the task name.

    For example:
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
    Count task files in a split.
    """

    return sum(
        len(tasks)
        for tasks in data.values()
    )



def collect_scene_counts(data):
    """
    Count task files by scene in a split.
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
    Print subject/task/scene counts for a split.
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
    Save:
        train.pkl / val.pkl / test.pkl

    Also save:
        train_list.csv / val_list.csv / test_list.csv
    """

    os.makedirs(
        folder,
        exist_ok=True
    )


    # -------------------------
    # Save the pkl file
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
    # Save the CSV index
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
    Perform basic validation of input multitask_dataset_terrain.pkl.

    Verify:
    1. The dataset is nonempty
    2. At least one subject/task exists
    3. Terrain regression labels are included in the pkl
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
    Split across subjects.

    Train:
        sub_001 ~ sub_034
        all scenes

    Val:
        sub_035 ~ sub_041
        all scenes

    Test:
        sub_042 ~ sub_048
        all scenes
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
    Evaluate both:
        unseen subjects + unseen 3D scenes

    Train:
        sub_001 ~ sub_034
        ONE_D_SCENES + XY_SCENES

    Val:
        sub_035 ~ sub_041
        ONE_D_SCENES + XY_SCENES

    Test:
        sub_042 ~ sub_048
        XYZ_SCENES

    Note:
        Test changes both subject and scene.
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
    General scene-splitting function.

    Characteristics:
        - All subjects may appear in train/val/test
        - Scene determines train/test membership
        - Validation samples are drawn randomly from training scenes per subject

    Thus validation remains within the observed-scene distribution.
    """

    train = {}

    val = {}

    test = {}


    # -------------------------
    # Split train/test by scene first
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
    # Draw validation samples from training
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


        # Avoid exhausting the participant's training samples
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
    Single-direction/basic terrain -> combined terrain

    Train scenes:
        walk
        stair
        x
        y
        z1-z4

    Val:
        Sample 10% of the training scenes per subject

    Test scenes:
        xy1-xy6
        xyz1-xyz3

    Purpose:
        Test generalization from basic/single-direction constraints to 2D/3D combinations.
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
    Basic scenes + some XY -> unseen XY + XYZ

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
        Sample 10% from the training scenes above

    Test scenes:
        xy4
        xy5
        xy6
        xyz1
        xyz2
        xyz3

    Purpose:
        After learning some 2D combinations, test
        generalization to unseen 2D and 3D combinations.
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
    Generalization within 2D combinations.

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
        Sample 10% from the training scenes above

    Test scenes:
        xy4
        xy5
        xy6

    XYZ is excluded from test in this split.

    Purpose:
        Assess whether the model learned XY combination patterns.
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
    1D + all 2D -> 3D

    Train scenes:
        walk
        stair
        x
        y
        z1-z4
        xy1-xy6

    Val:
        Sample 10% from the training scenes above

    Test scenes:
        xyz1
        xyz2
        xyz3

    Purpose:
        As the primary terrain-scene generalization experiment,
        test whether the model generalizes from 1D/2D constraints
        to unseen 3D combined constraints.
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

    Each fold:
        test subjects:
            KFold current test fold

        trainval subjects:
            remaining subjects

        val subjects:
            approximately 15% randomly sampled from trainval

        train subjects:
            remaining trainval subjects

    Ensure each subject belongs to only one of train/val/test.
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
    Create and save a train/val/test split.
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
    # Validate input files
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
    # Load the dataset
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
    # Validate dataset contents
    # -------------------------

    validate_dataset_structure(
        dataset
    )


    # -------------------------
    # Output directory
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
