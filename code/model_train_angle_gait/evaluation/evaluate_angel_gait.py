############################################################
# Trigait Multi-task Evaluation Framework
#
# Dataset:
#   IMU [B,200,42]
#
# Output:
#   Angle
#   Gait

#
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


# Current project root directory
#
# model_train/
# ├── datasets
# ├── models
# ├── train/checkpoints
# ├── evaluation
#

BASE_DIR = (

    "/mnt/newdisk/dateset_0909/"

    "code/"

    "model_train_angle_gait"

)

# checkpoint

CHECKPOINT_DIR = os.path.join(

    BASE_DIR,

    "train",

    "checkpoints"

)

# evaluation output

EVALUATION_DIR = os.path.join(

    BASE_DIR,

    "evaluation_results"

)

CURRENT_DIR = os.path.dirname(

    os.path.abspath(__file__)

)

# dataset path

sys.path.append(

    os.path.join(

        BASE_DIR,

        "datasets"

    )

)

# model path

sys.path.append(

    os.path.join(

        BASE_DIR,

        "models"

    )

)

from dataset_angle_gait import TrigaitMultiTaskDataset

torch.multiprocessing.set_sharing_strategy(

    "file_system"

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


############################################################
# Model Loader
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
# Checkpoint
############################################################


def get_checkpoint_path(

        model_name,

        split_name

):
    return os.path.join(

        CHECKPOINT_DIR,

        model_name,

        split_name,

        "best_model.pth"

    )


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

            overall,

            f,

            indent=4,

            ensure_ascii=False

        )

    print(

        "Evaluation results saved:",

        output_dir

    )


############################################################
# Main
############################################################


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--model",

        required=False,

        default="Transformer",

        choices=[

            "TCN",

            "LSTM",

            "Transformer",

            "GPT4TS",

            "TimesNet",

            "PatchTST",

            "TimeMixer",

            "Mamba"

        ]

    )

    parser.add_argument(

        "--name",

        default="subject_scene_split"

    )

    parser.add_argument(

        "--models",

        nargs="+",

        default=None

    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=None
    )



    parser.add_argument(

        "--test",

        required=False,

        default=None

    )

    parser.add_argument(

        "--batch_size",

        type=int,

        default=256

    )

    args = parser.parse_args()

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

    ########################################################
    # Multi-model / multi-split evaluation
    ########################################################

    models_to_eval = args.models if args.models else [args.model]

    if args.splits is not None:

        split_list = args.splits

    else:

        # Select automatically by name
        if "subject_scene_split" in args.name:

            split_list = [
                "subject_scene_split"
            ]

        elif "subject_split" in args.name:

            split_list = [
                "subject_split"
            ]

        elif "scene_split" in args.name:

            split_list = [
                "scene_split"
            ]

        else:

            raise ValueError(
                "Cannot determine split from name"
            )

    for model_name in models_to_eval:

        for split_name in split_list:
            experiment_name = (
                f"{split_name}_angle_gait_{model_name}"
            )

            test_path = os.path.join(
                BASE_DIR,
                "splits",
                split_name,
                "test.pkl"
            )

            print("=" * 60)
            print("Evaluate:", model_name, split_name)
            print("=" * 60)

            model = load_model(
                model_name,
                device
            )

            checkpoint = get_checkpoint_path(
                model_name,
                experiment_name
            )

            model = load_checkpoint(
                model,
                checkpoint,
                device
            )

            dataset = TrigaitMultiTaskDataset(
                test_path
            )

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0
            )

            results = evaluate_subject_task(
                model,
                loader,
                device
            )

            output_dir = os.path.join(
                EVALUATION_DIR,
                model_name,
                experiment_name
            )

            save_results(
                output_dir,
                *results
            )

            del model
            torch.cuda.empty_cache()

    print("All evaluation finished")
    return

    ########################################################
    # Load model
    ########################################################

    model = load_model(

        args.model,

        device

    )

    ########################################################
    # Load checkpoint
    ########################################################

    checkpoint = get_checkpoint_path(

        args.model,

        args.name

    )

    model = load_checkpoint(

        model,

        checkpoint,

        device

    )

    ########################################################
    # Dataset
    ########################################################

    print(

        "Loading test dataset:",

        args.test

    )

    dataset = TrigaitMultiTaskDataset(

        args.test

    )

    loader = DataLoader(

        dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=8,

        pin_memory=True

    )

    print(

        "Test samples:",

        len(dataset)

    )

    ########################################################
    # Evaluation
    ########################################################

    results = evaluate_subject_task(

        model,

        loader,

        device

    )

    ########################################################
    # Save
    ########################################################

    output_dir = os.path.join(

        EVALUATION_DIR,

        args.model,

        args.name

    )

    save_results(

        output_dir,

        *results

    )

    print(

        "Finished!"

    )


if __name__ == "__main__":
    main()