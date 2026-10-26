############################################################
# Trigait Multi-task 5-Fold Evaluation Framework
#
# Dataset:
#   IMU [B,200,42]
#
# Output:
#   Angle
#   Gait
#
# This file was adapted from evaluation/evaluate_angel_gait.py
# to evaluate subject five-fold training outputs directly.
#
# Original path convention (one split layer):
#   data  : splits/<split_name>/test.pkl
#   ckpt  : train/checkpoints/<model>/<split_name>_angle_gait_<model>/best_model.pth
#
# Five-fold data layout (additional fold layer):
#   data  : splits/<fold_root>/<fold>/test.pkl
#           Example: splits/subject_5fold_split/fold1/test.pkl
#   ckpt  : train/checkpoints/<model>/<fold>/best_model.pth
#           Example: train/checkpoints/GPT4TS/fold1/best_model.pth
#
# Therefore dataset path and checkpoint directory name are configured independently:
#   --fold_root   affects the test.pkl path only (default subject_5fold_split)
#   --fold(s)     fold names, default fold1 ... fold5
#   --ckpt_style  checkpoint naming convention: fold / experiment / auto
#
# Evaluation metrics and outputs match evaluate_angel_gait.py,
# with additional cross-fold mean/std summaries.
############################################################


import os
import sys
import json
import argparse
import importlib

import numpy as np
import pandas as pd

import torch
import torch.multiprocessing

from torch.utils.data import DataLoader

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score,

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix

)

from scipy.stats import pearsonr


############################################################
# PATH
############################################################


# Remote project root consistent with evaluate_angel_gait.py

DEFAULT_BASE_DIR = (

    "/mnt/newdisk/dateset_0909/"

    "code/"

    "model_train_angle_gait"

)


BASE_DIR = DEFAULT_BASE_DIR


# Script directory for locating datasets/ and models/ when running locally

CURRENT_DIR = os.path.dirname(

    os.path.abspath(__file__)

)


LOCAL_PROJECT_DIR = os.path.dirname(

    CURRENT_DIR

)


torch.multiprocessing.set_sharing_strategy(

    "file_system"

)


############################################################
# Import path helper
#
# Remote: BASE_DIR/datasets, BASE_DIR/models
# Local: <project directory>/datasets, <project directory>/models
#
# Both roots are added to sys.path; missing paths are ignored.
############################################################


def add_sys_paths(base_dir):

    for sub in ("datasets", "models"):

        for root in (base_dir, LOCAL_PROJECT_DIR):

            path = os.path.join(

                root,

                sub

            )

            if path not in sys.path:

                sys.path.append(

                    path

                )


def import_dataset_class(base_dir):

    add_sys_paths(

        base_dir

    )

    from dataset_angle_gait import TrigaitMultiTaskDataset

    return TrigaitMultiTaskDataset


add_sys_paths(

    BASE_DIR

)


############################################################
# Model Map
############################################################


MODEL_MAP = {

    "TCN":

        (

            "tcn_multitask_angle_gait",

            "TCNMultiTask"

        ),

    "LSTM":

        (

            "lstm_multitask_angle_gait",

            "LSTMMultiTask"

        ),

    "Transformer":

        (

            "transformer_multitask_angle_gait",

            "TransformerMultiTask"

        ),

    "GPT4TS":

        (

            "gpt4ts_multitask_angle_gait",

            "GPT4TSMultiTask"

        ),

    "TimesNet":

        (

            "timesnet_multitask_angle_gait",

            "TimesNetMultiTask"

        ),

    "PatchTST":

        (

            "patchtst_multitask_angle_gait",

            "PatchTSTMultiTask"

        ),

    "TimeMixer":

        (

            "timemixer_multitask_angle_gait",

            "TimeMixerMultiTask"

        ),

    "Mamba":

        (

            "mamba_multitask_angle_gait",

            "MambaMultiTask"

        ),

    "Persistence":

        (

            "persistence_multitask_angle_gait",

            "PersistenceMultiTask"

        )

}


# Default folds for five-fold evaluation

DEFAULT_FOLDS = [

    "fold1",

    "fold2",

    "fold3",

    "fold4",

    "fold5"

]


# Default data-directory hierarchy

DEFAULT_FOLD_ROOT = "subject_5fold_split"


# Metric columns for cross-fold summaries

SUMMARY_METRICS = [

    "angle_MAE",

    "angle_RMSE",

    "angle_R2",

    "gait_F1"

]


############################################################
# Model Loader
#
# Consistent with evaluate_angel_gait.py,
# so a given checkpoint uses the same model architecture as in single-split evaluation.
############################################################


def load_model(

        model_name,

        device

):
    if model_name not in MODEL_MAP:
        raise ValueError(

            f"Unknown model {model_name}"

        )

    module_name, class_name = MODEL_MAP[model_name]

    module = importlib.import_module(

        module_name

    )

    model_class = getattr(

        module,

        class_name

    )

    if model_name == "Transformer":

        model = model_class(

            input_dim=42,

            d_model=128,

            nhead=4,

            num_layers=3,

            dim_feedforward=256,

            output_len=30,

            dropout=0.2

        )

    elif model_name == "TimesNet":

        model = model_class(

            input_len=200,

            pred_len=30,

            input_dim=42,

            d_model=64,

            d_ff=128,

            num_layers=2,

            dropout=0.1

        )

    elif model_name == "PatchTST":

        model = model_class(

            input_len=200,

            pred_len=30,

            input_dim=42,

            patch_len=16,

            stride=8

        )

    elif model_name == "TimeMixer":

        model = model_class(

            input_len=200,

            input_dim=42,

            pred_len=30,

            d_model=64,

            d_ff=128,

            num_layers=2,

            dropout=0.1

        )

    elif model_name == "GPT4TS":

        model = model_class(

            input_dim=42,

            pred_len=30,

            freeze=True

        )

    elif model_name == "Mamba":

        # Must match training configuration:
        # config.yaml:
        # Mamba d_model: 64
        #
        # Training checkpoint:
        # embedding.weight [64,42]

        model = model_class(

            input_dim=42,

            d_model=64,

            num_layers=2,

            pred_len=30

        )

    else:

        model = model_class()

    model = model.to(

        device

    )

    return model


############################################################
# Path resolution for 5-fold protocol
############################################################


def get_checkpoint_path(

        checkpoint_dir,

        model_name,

        ckpt_name

):
    return os.path.join(

        checkpoint_dir,

        model_name,

        ckpt_name,

        "best_model.pth"

    )


def candidate_ckpt_names(

        model_name,

        fold_root,

        fold,

        ckpt_style

):
    """
    Return candidate checkpoint directory names in priority order.

    fold       : fold1
    experiment : subject_5fold_split_fold1_angle_gait_GPT4TS
                 fold1_angle_gait_GPT4TS
    auto       : prefer fold, then experiment

    Fold naming is preferred because the five-fold training script
    uses args.name=foldN in checkpoints/<model>/<args.name>.
    """

    names = []

    if ckpt_style in ("auto", "fold"):

        names.append(

            fold

        )

    if ckpt_style in ("auto", "experiment"):

        names.append(

            f"{fold_root}_{fold}_angle_gait_{model_name}"

        )

        names.append(

            f"{fold}_angle_gait_{model_name}"

        )

    # Remove duplicates while preserving order

    unique = []

    for name in names:

        if name and name not in unique:

            unique.append(

                name

            )

    return unique


def resolve_checkpoint(

        checkpoint_dir,

        model_name,

        fold_root,

        fold,

        ckpt_style

):
    """
    Return (checkpoint path, directory name);
    return (None, None) if no candidate exists.
    """

    for ckpt_name in candidate_ckpt_names(

            model_name,

            fold_root,

            fold,

            ckpt_style

    ):

        path = get_checkpoint_path(

            checkpoint_dir,

            model_name,

            ckpt_name

        )

        if os.path.isfile(

                path

        ):
            return path, ckpt_name

    return None, None


def candidate_test_paths(

        base_dir,

        fold_root,

        fold

):
    """
    Prefer five-fold layout while supporting the original flat layout.
    """

    paths = []

    if fold_root:

        paths.append(

            os.path.join(

                base_dir,

                "splits",

                fold_root,

                fold,

                "test.pkl"

            )

        )

    paths.append(

        os.path.join(

            base_dir,

            "splits",

            fold,

            "test.pkl"

        )

    )

    return paths


def resolve_test_path(

        base_dir,

        fold_root,

        fold

):
    """
    Return the first existing test.pkl, or None if none exists.
    """

    for path in candidate_test_paths(

            base_dir,

            fold_root,

            fold

    ):

        if os.path.isfile(

                path

        ):
            return path

    return None


############################################################
# Checkpoint
############################################################


def load_checkpoint(

        model,

        checkpoint_path,

        device

):
    print(

        "Loading checkpoint:",

        checkpoint_path

    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )

    if "model_state_dict" in checkpoint:

        checkpoint = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        checkpoint = checkpoint["state_dict"]

    new_state = {}

    for k, v in checkpoint.items():

        if k.startswith(

                "module."

        ):
            k = k.replace(

                "module.",

                ""

            )

        new_state[k] = v

    model.load_state_dict(

        new_state,

        strict=True

    )

    return model


############################################################
# Regression Metrics
############################################################


def calc_regression_metrics(

        true,

        pred

):
    mae = mean_absolute_error(

        true,

        pred

    )

    rmse = np.sqrt(

        mean_squared_error(

            true,

            pred

        )

    )

    try:

        r2 = r2_score(

            true,

            pred

        )

    except:

        r2 = np.nan

    try:

        corr = pearsonr(

            true,

            pred

        )[0]

    except:

        corr = np.nan

    mape = np.mean(

        np.abs(

            (true - pred)

            /

            (true + 1e-8)

        )

    ) * 100

    return {

        "MAE":

            mae,

        "RMSE":

            rmse,

        "R2":

            r2,

        "Pearson":

            corr,

        "MAPE":

            mape

    }


############################################################
# Angle Metrics
############################################################


def calc_angle_metrics(

        true,

        pred

):
    names = [

        "angle_0",

        "angle_1",

        "angle_2",

        "angle_3",

        "angle_4",

        "angle_5",

        "angle_6",

        "angle_7"

    ]

    results = []

    for i, name in enumerate(names):

        metric = calc_regression_metrics(

            true[:, :, i].reshape(-1),

            pred[:, :, i].reshape(-1)

        )

        metric["name"] = name

        results.append(

            metric

        )

    return results


############################################################
# Gait Metrics
############################################################


def calc_gait_metrics(

        true,

        pred

):
    true = true.reshape(-1)

    pred = pred.reshape(-1)

    cm = confusion_matrix(

        true,

        pred

    )

    return {

        "Accuracy":

            accuracy_score(

                true,

                pred

            ),

        "Precision":

            precision_score(

                true,

                pred,

                average="macro",

                zero_division=0

            ),

        "Recall":

            recall_score(

                true,

                pred,

                average="macro",

                zero_division=0

            ),

        "F1":

            f1_score(

                true,

                pred,

                average="macro",

                zero_division=0

            ),

        "Confusion":

            cm.tolist()

    }


############################################################
# Horizon Error
############################################################


def calculate_horizon_error(

        true_angle,

        pred_angle

):
    results = []

    pred_len = true_angle.shape[1]

    for t in range(pred_len):

        angle_mae = mean_absolute_error(

            true_angle[:, t, :].reshape(-1),

            pred_angle[:, t, :].reshape(-1)

        )

        results.append(

            {

                "frame": t + 1,

                "angle_MAE": angle_mae

            }

        )

    return results


############################################################
# Main Evaluation Loop
############################################################


def evaluate_subject_task(

        model,

        loader,

        device

):
    model.eval()

    subject_task_records = []

    angle_records = []

    gait_records = []

    horizon_records = []

    with torch.no_grad():

        for batch_id, batch in enumerate(loader):

            imu = batch["imu"].to(

                device

            )

            output = model(

                imu

            )

            # =============================
            # Angle
            # =============================

            angle_pred = output["angle"].cpu().numpy()

            angle_true = batch["angle"].numpy()

            # =============================
            # Gait
            # =============================

            gait_pred = torch.argmax(

                output["gait"],

                dim=-1

            ).cpu().numpy()

            gait_true = torch.argmax(

                batch["gait"],

                dim=-1

            ).numpy()

            subjects = batch["subject"]

            tasks = batch["task"]

            # =====================================
            # Sample level metrics
            # =====================================

            for i in range(len(subjects)):

                subject = subjects[i]

                task = tasks[i]

                # -------------------------
                # angle
                # -------------------------

                angle_metric = calc_regression_metrics(

                    angle_true[i].reshape(-1),

                    angle_pred[i].reshape(-1)

                )

                angle_metric.update(

                    {

                        "subject":

                            subject,

                        "task":

                            task

                    }

                )

                angle_records.append(

                    angle_metric

                )

                # -------------------------
                # gait
                # -------------------------

                gait_metric = calc_gait_metrics(

                    gait_true[i],

                    gait_pred[i]

                )

                gait_metric.pop(

                    "Confusion",

                    None

                )

                gait_metric.update(

                    {

                        "subject":

                            subject,

                        "task":

                            task

                    }

                )

                gait_records.append(

                    gait_metric

                )

                # -------------------------
                # summary
                # -------------------------

                subject_task_records.append(

                    {

                        "subject":

                            subject,

                        "task":

                            task,

                        "angle_MAE":

                            angle_metric["MAE"],

                        "angle_RMSE":

                            angle_metric["RMSE"],

                        "angle_R2":

                            angle_metric["R2"],

                        "gait_F1":

                            gait_metric["F1"]

                    }

                )

            # =====================================
            # Horizon
            # =====================================

            horizon_records.extend(

                calculate_horizon_error(

                    angle_true,

                    angle_pred

                )

            )

            if batch_id % 100 == 0:

                print(

                    "evaluation batch:",

                    batch_id

                )

    return (

        subject_task_records,

        angle_records,

        gait_records,

        horizon_records

    )


############################################################
# Save Results
#
# Per-fold output matches the original script (seven files).
############################################################


def save_results(

        output_dir,

        subject_task_records,

        angle_records,

        gait_records,

        horizon_records

):
    os.makedirs(

        output_dir,

        exist_ok=True

    )

    subject_task_df = pd.DataFrame(

        subject_task_records

    )

    angle_df = pd.DataFrame(

        angle_records

    )

    gait_df = pd.DataFrame(

        gait_records

    )

    horizon_df = pd.DataFrame(

        horizon_records

    )

    ########################################################
    # Detail Results
    ########################################################

    subject_task_df.to_excel(

        os.path.join(

            output_dir,

            "subject_task_metrics.xlsx"

        ),

        index=False

    )

    angle_df.to_excel(

        os.path.join(

            output_dir,

            "angle_detail.xlsx"

        ),

        index=False

    )

    gait_df.to_excel(

        os.path.join(

            output_dir,

            "gait_detail.xlsx"

        ),

        index=False

    )

    horizon_df.to_excel(

        os.path.join(

            output_dir,

            "horizon_error.xlsx"

        ),

        index=False

    )

    ########################################################
    # Subject Summary
    ########################################################

    subject_summary = (

        subject_task_df

        .groupby(

            "subject"

        )

        .mean(

            numeric_only=True

        )

        .reset_index()

    )

    subject_summary.to_excel(

        os.path.join(

            output_dir,

            "subject_metrics.xlsx"

        ),

        index=False

    )

    ########################################################
    # Task Summary
    ########################################################

    task_summary = (

        subject_task_df

        .groupby(

            "task"

        )

        .mean(

            numeric_only=True

        )

        .reset_index()

    )

    task_summary.to_excel(

        os.path.join(

            output_dir,

            "task_metrics.xlsx"

        ),

        index=False

    )

    ########################################################
    # Overall Summary
    ########################################################

    overall = (

        subject_task_df

        .mean(

            numeric_only=True

        )

        .to_dict()

    )

    with open(

            os.path.join(

                output_dir,

                "summary.json"

            ),

            "w",

            encoding="utf-8"

    ) as f:

        json.dump(

            to_builtin(

                overall

            ),

            f,

            indent=4,

            ensure_ascii=False

        )

    print(

        "Evaluation results saved:",

        output_dir

    )

    return overall


############################################################
# Cross-fold Summary
############################################################


def to_builtin(value):
    """
    Convert NumPy/Pandas scalars into native JSON-serializable types.
    """

    if isinstance(

            value,

            dict

    ):
        return {

            str(k):

                to_builtin(v)

            for k, v in value.items()

        }

    if isinstance(

            value,

            (list, tuple)

    ):
        return [

            to_builtin(v)

            for v in value

        ]

    if isinstance(

            value,

            (np.integer,)

    ):
        return int(

            value

        )

    if isinstance(

            value,

            (np.floating,)

    ):
        return float(

            value

        )

    if isinstance(

            value,

            (np.bool_,)

    ):
        return bool(

            value

        )

    return value


def save_fold_summary(

        summary_dir,

        model_name,

        fold_root,

        fold_rows,

        skipped,

        ckpt_style

):
    """
    Summarize overall metrics across folds in a table with mean/std.

    Each fold_rows entry:
        fold / checkpoint / ckpt_dir / test_path / angle_MAE / ...
    """

    os.makedirs(

        summary_dir,

        exist_ok=True

    )

    fold_df = pd.DataFrame(

        fold_rows

    )

    fold_metrics_path = os.path.join(

        summary_dir,

        "fold_metrics.xlsx"

    )

    fold_df.to_excel(

        fold_metrics_path,

        index=False

    )

    metric_cols = [

        c

        for c in SUMMARY_METRICS

        if c in fold_df.columns

    ]

    mean_row = {}

    std_row = {}

    if metric_cols and len(fold_df) > 0:

        mean_row = fold_df[metric_cols].mean(

        ).to_dict()

        if len(fold_df) > 1:

            std_row = fold_df[metric_cols].std(

                ddof=1

            ).to_dict()

        else:

            std_row = {

                c: 0.0

                for c in metric_cols

            }

    summary_df = pd.DataFrame(

        {

            "metric": metric_cols,

            "mean": [

                mean_row.get(

                    c,

                    np.nan

                )

                for c in metric_cols

            ],

            "std": [

                std_row.get(

                    c,

                    np.nan

                )

                for c in metric_cols

            ]

        }

    )

    summary_df.to_excel(

        os.path.join(

            summary_dir,

            "fold_summary.xlsx"

        ),

        index=False

    )

    payload = {

        "model":

            model_name,

        "fold_root":

            fold_root,

        "ckpt_style":

            ckpt_style,

        "num_folds":

            len(fold_df),

        "folds":

            fold_rows,

        "skipped":

            skipped,

        "mean":

            mean_row,

        "std":

            std_row

    }

    with open(

            os.path.join(

                summary_dir,

                "fold_summary.json"

            ),

            "w",

            encoding="utf-8"

    ) as f:

        json.dump(

            to_builtin(

                payload

            ),

            f,

            indent=4,

            ensure_ascii=False

        )

    print(

        "Fold summary saved:",

        summary_dir

    )

    for c in metric_cols:

        print(

            f"  {c}: "

            f"{mean_row.get(c, float('nan')):.6f} "

            f"± {std_row.get(c, float('nan')):.6f}"

        )

    return {

        "num_folds":

            len(fold_df),

        "mean":

            mean_row,

        "std":

            std_row

    }


############################################################
# One Fold
############################################################


def evaluate_one_fold(

        model_name,

        fold,

        test_path,

        checkpoint_path,

        ckpt_name,

        output_dir,

        base_dir,

        device,

        batch_size

):
    dataset_class = import_dataset_class(

        base_dir

    )

    model = load_model(

        model_name,

        device

    )

    model = load_checkpoint(

        model,

        checkpoint_path,

        device

    )

    dataset = dataset_class(

        test_path

    )

    print(

        "Test samples:",

        len(dataset)

    )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0

    )

    results = evaluate_subject_task(

        model,

        loader,

        device

    )

    overall = save_results(

        output_dir,

        *results

    )

    del model

    torch.cuda.empty_cache()

    row = {

        "fold":

            fold,

        "checkpoint":

            checkpoint_path,

        "ckpt_dir":

            ckpt_name,

        "test_path":

            test_path,

        "output_dir":

            output_dir

    }

    for c in SUMMARY_METRICS:

        if c in overall:

            row[c] = float(

                overall[c]

            )

    return row


############################################################
# Main
############################################################


def main():
    parser = argparse.ArgumentParser(

        description=(

            "Trigait Angle+Gait 5-fold Evaluation"

        )

    )

    parser.add_argument(

        "--model",

        required=False,

        default="Transformer",

        choices=list(

            MODEL_MAP.keys()

        )

    )

    parser.add_argument(

        "--models",

        nargs="+",

        default=None,

        help=(

            "Evaluate multiple models, e.g. "

            "--models TCN LSTM GPT4TS；"

            "Pass all to include every model in MODEL_MAP"

        )

    )

    parser.add_argument(

        "--folds",

        nargs="+",

        default=list(

            DEFAULT_FOLDS

        )

    )

    parser.add_argument(

        "--fold_root",

        default=DEFAULT_FOLD_ROOT,

        help=(

            "Parent directory of test.pkl,"

            "default subject_5fold_split"

        )

    )

    parser.add_argument(

        "--ckpt_style",

        default="auto",

        choices=[

            "auto",

            "fold",

            "experiment"

        ],

        help=(

            "Checkpoint directory naming: fold=fold1;"

            "experiment=<fold_root>_<fold>_angle_gait_<model>"

        )

    )

    parser.add_argument(

        "--base_dir",

        default=None,

        help=(

            "Project root, defaulting to the configured "

            "/mnt/newdisk/dateset_0909/code/model_train_angle_gait"

        )

    )

    parser.add_argument(

        "--checkpoint_dir",

        default=None,

        help="Checkpoint root, default <base_dir>/train/checkpoints"

    )

    parser.add_argument(

        "--eval_dir",

        default=None,

        help="Result root, default <base_dir>/evaluation_results"

    )

    parser.add_argument(

        "--test",

        default=None,

        help=(

            "Evaluate one specified test.pkl,"

            "used with --model/--folds (ignore fold_root construction)"

        )

    )

    parser.add_argument(

        "--batch_size",

        type=int,

        default=256

    )

    parser.add_argument(

        "--strict",

        action="store_true",

        help="Raise instead of skipping when checkpoint or test.pkl is missing"

    )

    parser.add_argument(

        "--skip_existing",

        action="store_true",

        help="Skip folds with existing summary.json to resume interrupted runs"

    )

    args = parser.parse_args()

    ########################################################
    # Paths
    ########################################################

    base_dir = args.base_dir or DEFAULT_BASE_DIR

    checkpoint_dir = (

        args.checkpoint_dir

        or os.path.join(

            base_dir,

            "train",

            "checkpoints"

        )

    )

    eval_dir = (

        args.eval_dir

        or os.path.join(

            base_dir,

            "evaluation_results"

        )

    )

    add_sys_paths(

        base_dir

    )

    ########################################################
    # Device
    ########################################################

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print(

        "Device:",

        device

    )

    print(

        "Base dir:",

        base_dir

    )

    ########################################################
    # Models / folds
    ########################################################

    if args.models:

        if "all" in args.models:

            models_to_eval = [

                m

                for m in MODEL_MAP

                if m != "Persistence"

            ]

        else:

            models_to_eval = list(

                args.models

            )

    else:

        models_to_eval = [

            args.model

        ]

    folds_to_eval = list(

        args.folds

    )

    ########################################################
    # Multi-model / multi-fold evaluation
    ########################################################

    all_models_summary = []

    for model_name in models_to_eval:

        summary_dir = os.path.join(

            eval_dir,

            model_name,

            args.fold_root

        )

        fold_rows = []

        skipped = []

        for fold in folds_to_eval:

            print("=" * 60)

            print(

                "Evaluate:",

                model_name,

                fold

            )

            print("=" * 60)

            output_dir = os.path.join(

                summary_dir,

                fold

            )

            ####################################################
            # Resume an interrupted run
            ####################################################

            existing_summary = os.path.join(

                output_dir,

                "summary.json"

            )

            if args.skip_existing and os.path.isfile(

                    existing_summary

            ):

                print(

                    "Skip (existing summary):",

                    output_dir

                )

                with open(

                        existing_summary,

                        "r",

                        encoding="utf-8"

                ) as f:

                    overall = json.load(

                        f

                    )

                row = {

                    "fold":

                        fold,

                    "checkpoint":

                        "",

                    "ckpt_dir":

                        "",

                    "test_path":

                        "",

                    "output_dir":

                        output_dir

                }

                for c in SUMMARY_METRICS:

                    if c in overall:

                        row[c] = float(

                            overall[c]

                        )

                fold_rows.append(

                    row

                )

                continue

            ####################################################
            # Data paths
            ####################################################

            if args.test:

                test_path = args.test

            else:

                test_path = resolve_test_path(

                    base_dir,

                    args.fold_root,

                    fold

                )

            if test_path is None or not os.path.isfile(

                    test_path

            ):

                message = (

                    f"test.pkl not found for fold {fold}: "

                    + str(

                        candidate_test_paths(

                            base_dir,

                            args.fold_root,

                            fold

                        )

                    )

                )

                if args.strict:

                    raise FileNotFoundError(

                        message

                    )

                print(

                    "[SKIP]",

                    message

                )

                skipped.append(

                    {

                        "fold":

                            fold,

                        "reason":

                            "test.pkl missing",

                        "detail":

                            message

                    }

                )

                continue

            ####################################################
            # Checkpoint paths
            ####################################################

            checkpoint_path, ckpt_name = resolve_checkpoint(

                checkpoint_dir,

                model_name,

                args.fold_root,

                fold,

                args.ckpt_style

            )

            if checkpoint_path is None:

                message = (

                    f"checkpoint not found for {model_name}/{fold}, "

                    "tried: "

                    + str(

                        [

                            get_checkpoint_path(

                                checkpoint_dir,

                                model_name,

                                name

                            )

                            for name in candidate_ckpt_names(

                                model_name,

                                args.fold_root,

                                fold,

                                args.ckpt_style

                            )

                        ]

                    )

                )

                if args.strict:

                    raise FileNotFoundError(

                        message

                    )

                print(

                    "[SKIP]",

                    message

                )

                skipped.append(

                    {

                        "fold":

                            fold,

                        "reason":

                            "checkpoint missing",

                        "detail":

                            message

                    }

                )

                continue

            ####################################################
            # Evaluation
            ####################################################

            row = evaluate_one_fold(

                model_name,

                fold,

                test_path,

                checkpoint_path,

                ckpt_name,

                output_dir,

                base_dir,

                device,

                args.batch_size

            )

            fold_rows.append(

                row

            )

        ########################################################
        # Cross-fold summary
        ########################################################

        if fold_rows:

            model_summary = save_fold_summary(

                summary_dir,

                model_name,

                args.fold_root,

                fold_rows,

                skipped,

                args.ckpt_style

            )

            entry = {

                "model":

                    model_name,

                "num_folds":

                    model_summary["num_folds"]

            }

            for c in SUMMARY_METRICS:

                entry[c + "_mean"] = model_summary["mean"].get(

                    c,

                    np.nan

                )

                entry[c + "_std"] = model_summary["std"].get(

                    c,

                    np.nan

                )

            all_models_summary.append(

                entry

            )

        else:

            print(

                "[WARN] no fold evaluated for",

                model_name

            )

    ########################################################
    # All-model summary
    ########################################################

    if all_models_summary:

        os.makedirs(

            eval_dir,

            exist_ok=True

        )

        all_df = pd.DataFrame(

            all_models_summary

        )

        all_path = os.path.join(

            eval_dir,

            "fold_summary_all_models.xlsx"

        )

        all_df.to_excel(

            all_path,

            index=False

        )

        with open(

                os.path.join(

                    eval_dir,

                    "fold_summary_all_models.json"

                ),

                "w",

                encoding="utf-8"

        ) as f:

            json.dump(

                to_builtin(

                    all_models_summary

                ),

                f,

                indent=4,

                ensure_ascii=False

            )

        print(

            "All-model fold summary saved:",

            all_path

        )

    print("All evaluation finished")


if __name__ == "__main__":
    main()
