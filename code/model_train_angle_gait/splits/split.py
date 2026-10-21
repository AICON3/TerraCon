
import os
import pickle
import random
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold


# =====================================================
# Path
# =====================================================

BASE_DIR = (
    "/mnt/newdisk/dateset_0909/"
    "code/model_train_angle_gait"
)


DEFAULT_INPUT = os.path.join(
    BASE_DIR,
    "data_pkl",
    "multitask_dataset.pkl"
)


DEFAULT_OUTPUT = os.path.join(
    BASE_DIR,
    "splits"
)


SEED = 2026

random.seed(SEED)
np.random.seed(SEED)


# =====================================================
# Scene parsing
# =====================================================

def get_scene(task):
    """
    Extract scene name from task name.

    Preserve the original parsing rule:
    use the third component when splitting the
    task name by underscores.
    """

    parts = str(task).split("_")

    if len(parts) >= 4:
        return parts[2].lower()

    return None


# =====================================================
# Save split
# =====================================================

def save_split(data, folder, name):

    os.makedirs(
        folder,
        exist_ok=True
    )

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

    print(
        "{}: {} samples, {} subjects".format(
            os.path.join(folder, name),
            len(rows),
            len(data)
        )
    )


# =====================================================
# Subject split
# Original logic retained
# =====================================================

def subject_split(dataset):

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

    return train, val, test


# =====================================================
# Original scene definitions
# =====================================================

BASE_SCENES = [

    "walk",
    "stair",

    "x",
    "y",

    "z1",
    "z2",
    "z3",
    "z4"

]


COMPLEX_SCENES = [

    "xy1",
    "xy2",
    "xy3",
    "xy4",
    "xy5",
    "xy6",

    "xyz1",
    "xyz2",
    "xyz3"

]


# =====================================================
# Scene split
# Original logic retained
# =====================================================

def scene_split(dataset):

    train = {}
    test = {}

    for sub, tasks in dataset.items():

        for task, value in tasks.items():

            scene = get_scene(task)

            if scene in BASE_SCENES:

                train.setdefault(
                    sub,
                    {}
                )[task] = value

            elif scene in COMPLEX_SCENES:

                test.setdefault(
                    sub,
                    {}
                )[task] = value

    val = {}

    for sub in list(train.keys()):

        tasks = list(
            train[sub].keys()
        )

        random.shuffle(tasks)

        n = max(
            1,
            int(len(tasks) * 0.1)
        )

        for task in tasks[:n]:

            val.setdefault(
                sub,
                {}
            )[task] = train[sub][task]

            del train[sub][task]

    return train, val, test


# =====================================================
# Subject Scene split
# Original logic retained
# =====================================================

def subject_scene_split(dataset):

    train = {}
    val = {}
    test = {}

    for sub, tasks in dataset.items():

        sid = int(
            sub.split("_")[1]
        )

        for task, value in tasks.items():

            scene = get_scene(task)

            if (
                sid <= 34
                and scene in BASE_SCENES
            ):

                train.setdefault(
                    sub,
                    {}
                )[task] = value

            elif (
                35 <= sid <= 41
                and scene in BASE_SCENES
            ):

                val.setdefault(
                    sub,
                    {}
                )[task] = value

            elif (
                sid >= 42
                and scene in COMPLEX_SCENES
            ):

                test.setdefault(
                    sub,
                    {}
                )[task] = value

    return train, val, test


# =====================================================
# Subject 5-fold
# Original logic retained
# =====================================================

def subject_5fold_split(
        dataset,
        output_dir
):

    subjects = np.array(
        sorted(
            dataset.keys()
        )
    )

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    for fold, (trainval_idx, test_idx) in enumerate(
        kf.split(subjects),
        start=1
    ):

        print(
            "\n========== Fold {} ==========".format(
                fold
            )
        )

        test_subjects = subjects[test_idx]

        trainval_subjects = subjects[
            trainval_idx
        ].copy()

        random.shuffle(
            trainval_subjects
        )

        val_num = max(
            1,
            int(len(trainval_subjects) * 0.15)
        )

        val_subjects = trainval_subjects[:val_num]

        train_subjects = trainval_subjects[val_num:]

        train = {}
        val = {}
        test = {}

        for s in train_subjects:
            train[s] = dataset[s]

        for s in val_subjects:
            val[s] = dataset[s]

        for s in test_subjects:
            test[s] = dataset[s]

        fold_dir = os.path.join(
            output_dir,
            "subject_5fold_split",
            "fold{}".format(fold)
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

        print("train subjects:", len(train))
        print("val subjects:", len(val))
        print("test subjects:", len(test))


# =====================================================
# Scene split 5
# New split only
# =====================================================

def scene_split_5(dataset):
    """
    Scene split 5.

    Train:
        WALK, STAIR

    Validation:
        A random 10% of WALK/STAIR tasks
        for each subject, where possible.

    Test:
        XY1-XY6, XYZ1-XYZ3

    All subjects are eligible.

    This split evaluates scene generalization.
    It does not enforce subject-level separation.
    """

    train = {}
    val = {}
    test = {}

    train_scenes = {
        "walk",
        "stair"
    }

    test_scenes = {
        "xy1",
        "xy2",
        "xy3",
        "xy4",
        "xy5",
        "xy6",
        "xyz1",
        "xyz2",
        "xyz3"
    }

    # -------------------------------------------------
    # 1. Assign tasks by scene
    # -------------------------------------------------

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

    # -------------------------------------------------
    # 2. Split validation tasks from training scenes
    # -------------------------------------------------

    # Use an independent RNG to avoid changing the
    # random state used by the original split functions.
    rng = random.Random(SEED)

    for sub in list(train.keys()):

        subject_tasks = list(
            train[sub].keys()
        )

        rng.shuffle(
            subject_tasks
        )

        if len(subject_tasks) >= 2:

            n_val = max(
                1,
                int(len(subject_tasks) * 0.1)
            )

            # Retain at least one training task.
            n_val = min(
                n_val,
                len(subject_tasks) - 1
            )

        else:

            n_val = 0

        val_tasks = subject_tasks[:n_val]

        for task in val_tasks:

            val.setdefault(
                sub,
                {}
            )[task] = train[sub][task]

            del train[sub][task]

    # Remove empty subject entries.
    train = {
        sub: tasks
        for sub, tasks in train.items()
        if tasks
    }

    val = {
        sub: tasks
        for sub, tasks in val.items()
        if tasks
    }

    test = {
        sub: tasks
        for sub, tasks in test.items()
        if tasks
    }

    return train, val, test


# =====================================================
# Generate all splits
# =====================================================

def generate_all_splits(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT):

    print("=" * 60)
    print("Generate all dataset splits")
    print("=" * 60)
    print("Input:", input_path)
    print("Output:", output_dir)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            "Dataset file not found: {}".format(input_path)
        )

    with open(input_path, "rb") as f:
        dataset = pickle.load(f)

    print("Subjects in dataset:", len(dataset))

    # 1. Subject split
    print("\n===== Subject Split =====")
    train, val, test = subject_split(dataset)
    folder = os.path.join(output_dir, "subject_split")
    save_split(train, folder, "train")
    save_split(val, folder, "val")
    save_split(test, folder, "test")

    # 2. Original scene split, now saved as scene_split_1
    print("\n===== Scene Split 1 =====")
    train, val, test = scene_split(dataset)
    folder = os.path.join(output_dir, "scene_split_1")
    save_split(train, folder, "train")
    save_split(val, folder, "val")
    save_split(test, folder, "test")

    # 3. Subject-scene split
    print("\n===== Subject Scene Split =====")
    train, val, test = subject_scene_split(dataset)
    folder = os.path.join(output_dir, "subject_scene_split")
    save_split(train, folder, "train")
    save_split(val, folder, "val")
    save_split(test, folder, "test")

    # 4. Scene split 5
    print("\n===== Scene Split 5 =====")
    train, val, test = scene_split_5(dataset)
    folder = os.path.join(output_dir, "scene_split_5")
    save_split(train, folder, "train")
    save_split(val, folder, "val")
    save_split(test, folder, "test")

    print("\nScene Split 5 definition:")
    print("Train scenes: WALK, STAIR")
    print("Validation scenes: WALK, STAIR")
    print("Test scenes: XY1-XY6, XYZ1-XYZ3")

    # 5. Subject 5-fold cross-validation
    print("\n===== Subject 5 Fold =====")
    subject_5fold_split(dataset, output_dir)

    print("\n" + "=" * 60)
    print("All splits finished.")
    print("=" * 60)


# =====================================================
# Main
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate all dataset splits, including "
            "scene_split_1 and scene_split_5."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to multitask_dataset.pkl"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output directory for dataset splits"
    )

    args = parser.parse_args()

    generate_all_splits(
        input_path=args.input,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
