############################################################
# Trigait Persistence Baseline Evaluation
#
# No training and no checkpoint are required.
# Prediction: future = last observed state in the input window.
#
# Examples:
#   Evaluate all single splits and all 5 folds:
#     python evaluation/evaluate_persistence2.py
#
#   Evaluate only scene_split_5:
#     python evaluation/evaluate_persistence2.py --splits scene_split_5
#
#   Evaluate selected single splits:
#     python evaluation/evaluate_persistence2.py --splits scene_split_1 scene_split_5
#
#   Evaluate only subject 5-fold splits:
#     python evaluation/evaluate_persistence2.py --only_5fold
#
#   Evaluate all single splits and selected folds:
#     python evaluation/evaluate_persistence2.py --folds fold1 fold3
############################################################

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.stats import pearsonr


############################################################
# PATH
############################################################

BASE_DIR = (
    "/mnt/newdisk/dateset_0909/"
    "code/"
    "model_train_angle_gait"
)

EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation_results")

sys.path.append(os.path.join(BASE_DIR, "datasets"))

from dataset_angle_gait import TrigaitMultiTaskDataset


############################################################
# Metrics
############################################################

def regression_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))

    try:
        r2 = r2_score(true, pred)
    except Exception:
        r2 = np.nan

    try:
        corr = pearsonr(true, pred)[0]
    except Exception:
        corr = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Pearson": corr,
    }


def gait_metrics(true, pred):
    true = true.reshape(-1)
    pred = pred.reshape(-1)

    return {
        "Accuracy": accuracy_score(true, pred),
        "Precision": precision_score(
            true, pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            true, pred, average="macro", zero_division=0
        ),
        "F1": f1_score(
            true, pred, average="macro", zero_division=0
        ),
    }


############################################################
# Horizon
############################################################

def horizon_error(true, pred):
    results = []

    for t in range(true.shape[1]):
        mae = mean_absolute_error(
            true[:, t, :].reshape(-1),
            pred[:, t, :].reshape(-1)
        )
        results.append({
            "frame": t + 1,
            "angle_MAE": mae
        })

    return results


############################################################
# Persistence Evaluation
############################################################

def evaluate(loader):
    angle_records = []
    gait_records = []
    subject_task_records = []
    horizon_records = []

    for batch_id, batch in enumerate(loader):
        angle_true = batch["angle"].numpy()
        gait_true = batch["gait"].numpy()

        # Repeat the final observed input frame across the future horizon.
        # Future ground-truth labels are not used to construct predictions.
        last_angle = batch["angle_last"].numpy()
        angle_pred = np.repeat(
            last_angle[:, None, :],
            angle_true.shape[1],
            axis=1
        )

        last_gait = batch["gait_last"].numpy()
        gait_pred_prob = np.repeat(
            last_gait[:, None, :],
            gait_true.shape[1],
            axis=1
        )

        gait_true_cls = np.argmax(gait_true, axis=-1)
        gait_pred_cls = np.argmax(gait_pred_prob, axis=-1)

        subjects = batch["subject"]
        tasks = batch["task"]

        for i in range(len(subjects)):
            angle_metric = regression_metrics(
                angle_true[i].reshape(-1),
                angle_pred[i].reshape(-1)
            )
            gait_metric = gait_metrics(
                gait_true_cls[i],
                gait_pred_cls[i]
            )

            angle_metric.update({
                "subject": subjects[i],
                "task": tasks[i]
            })
            gait_metric.update({
                "subject": subjects[i],
                "task": tasks[i]
            })

            angle_records.append(angle_metric)
            gait_records.append(gait_metric)

            subject_task_records.append({
                "subject": subjects[i],
                "task": tasks[i],
                "angle_MAE": angle_metric["MAE"],
                "angle_RMSE": angle_metric["RMSE"],
                "angle_R2": angle_metric["R2"],
                "gait_F1": gait_metric["F1"]
            })

        horizon_records.extend(horizon_error(angle_true, angle_pred))

        if batch_id % 100 == 0:
            print("Evaluation batch:", batch_id)

    return (
        subject_task_records,
        angle_records,
        gait_records,
        horizon_records
    )


############################################################
# Save
############################################################

def save_results(output_dir, results):
    os.makedirs(output_dir, exist_ok=True)

    (
        subject_task_records,
        angle_records,
        gait_records,
        horizon_records
    ) = results

    subject_task_df = pd.DataFrame(subject_task_records)
    angle_df = pd.DataFrame(angle_records)
    gait_df = pd.DataFrame(gait_records)
    horizon_df = pd.DataFrame(horizon_records)

    subject_task_df.to_excel(
        os.path.join(output_dir, "subject_task_metrics.xlsx"),
        index=False
    )
    angle_df.to_excel(
        os.path.join(output_dir, "angle_detail.xlsx"),
        index=False
    )
    gait_df.to_excel(
        os.path.join(output_dir, "gait_detail.xlsx"),
        index=False
    )
    horizon_df.to_excel(
        os.path.join(output_dir, "horizon_error.xlsx"),
        index=False
    )

    subject_task_df.groupby("subject").mean(
        numeric_only=True
    ).reset_index().to_excel(
        os.path.join(output_dir, "subject_metrics.xlsx"),
        index=False
    )

    subject_task_df.groupby("task").mean(
        numeric_only=True
    ).reset_index().to_excel(
        os.path.join(output_dir, "task_metrics.xlsx"),
        index=False
    )

    summary = subject_task_df.mean(numeric_only=True).to_dict()
    with open(
        os.path.join(output_dir, "summary.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Saved results to:", output_dir)


############################################################
# Main
############################################################

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=[
            "subject_split",
            "scene_split_1",
            "scene_split_5",
            "subject_scene_split",
            "subject_5fold_split",
        ],
        default=None,
        help=(
            "Choose split(s). For example --splits scene_split_5. "
            "If omitted, all single splits and all five folds are evaluated."
        )
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help=(
            "Choose subject_5fold_split folds, e.g. --folds fold1 fold3. "
            "When used without --splits, all single splits are also evaluated."
        )
    )

    parser.add_argument(
        "--only_5fold",
        action="store_true",
        help="Only evaluate subject_5fold_split."
    )

    args = parser.parse_args()

    single_split_names = [
        "subject_split",
        "scene_split_1",
        "scene_split_5",
        "subject_scene_split",
    ]

    # Each run is (relative split path, output directory name, output subdirectory).
    runs = []

    if args.only_5fold:
        selected_single_splits = []
        include_folds = True
    elif args.splits is not None:
        selected_single_splits = [
            split for split in args.splits
            if split != "subject_5fold_split"
        ]
        include_folds = "subject_5fold_split" in args.splits
    else:
        selected_single_splits = single_split_names
        include_folds = True

    if not args.only_5fold and args.splits is None:
        selected_single_splits = single_split_names

    split_output_names = {
        "subject_split": "subject_split_angle_gait_Persistence",
        "scene_split_1": "scene_split_1_angle_gait_Persistence",
        "scene_split_5": "scene_split_5_angle_gait_Persistence",
        "subject_scene_split": "subject_scene_split_angle_gait_Persistence",
    }

    for split_name in selected_single_splits:
        runs.append((
            split_name,
            split_output_names[split_name],
            ""
        ))

    if include_folds:
        fold_root = "subject_5fold_split"
        folds = args.folds or [f"fold{i}" for i in range(1, 6)]

        for fold in folds:
            runs.append((
                os.path.join(fold_root, fold),
                f"{fold_root}_angle_gait_Persistence",
                f"{fold}_angle_gait_Persistence"
            ))

    for rel_path, name, sub_name in runs:
        print("\n" + "=" * 60)
        print("Evaluate Persistence:", rel_path)
        print("=" * 60)

        test_path = os.path.join(
            BASE_DIR, "splits", rel_path, "test.pkl"
        )

        if not os.path.isfile(test_path):
            print("[SKIP] test.pkl does not exist:", test_path)
            continue

        dataset = TrigaitMultiTaskDataset(test_path)

        if len(dataset) == 0:
            print("[SKIP] Dataset has no valid windows:", test_path)
            continue

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        results = evaluate(loader)

        output_dir = os.path.join(
            EVALUATION_DIR, "Persistence", name
        )

        if sub_name:
            output_dir = os.path.join(output_dir, sub_name)

        save_results(output_dir, results)

    print("\nPersistence evaluation finished.")


if __name__ == "__main__":
    main()