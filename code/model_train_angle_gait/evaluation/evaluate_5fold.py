############################################################
# Trigait Multi-task 5-Fold Evaluation Framework  (v2)
#
# 基于 evaluation/evaluate_angel_gait.py 改写，用于 5 折训练产物的评估。
#
# 数据路径:
#   <base>/splits/<fold_root>/<fold>/test.pkl
#   例: splits/subject_5fold_split/fold3/test.pkl
#
# 权重路径:
#   <base>/train/checkpoints/<Model>/<fold>_angle_gait_<Model>/best_model.pth
#   例: train/checkpoints/GPT4TS/fold3_angle_gait_GPT4TS/best_model.pth
#
# 相对 v1 的变化:
#   1. 权重目录默认按 <fold>_angle_gait_<Model> 解析（v1 以 foldN 优先），
#      同时保留 <fold_root>_<fold>_angle_gait_<Model> 与裸 <fold> 作为回退
#   2. --folds 默认自动发现（先看 splits/<fold_root>/，再看 checkpoints/<Model>/）
#   3. 新增 --preflight: 只检查数据/权重路径与 state_dict 是否和当前模型代码
#      匹配，不加载数据集、不跑推理，几秒内给出结论
#   4. 加载权重前先比对键名与张量形状，不匹配时给出可读提示并跳过该折，
#      而不是抛出上百行 RuntimeError（--strict 可改回直接报错）
#   5. BASE_DIR 默认自动定位到脚本所在工程，本机与服务器都能直接跑
#
# 评估指标与单折输出文件同 evaluate_angel_gait.py，另外增加跨折 mean/std 汇总。
############################################################


import os
import re
import sys
import csv
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


# 服务器上的工程根目录（脚本被放到别处运行时作为兜底）

REMOTE_BASE_DIR = (

    "/mnt/newdisk/dateset_0909/"

    "code/"

    "model_train_angle_gait"

)


# 脚本自身位置: <base>/evaluation/evaluate_5fold.py

CURRENT_DIR = os.path.dirname(

    os.path.abspath(__file__)

)


LOCAL_PROJECT_DIR = os.path.dirname(

    CURRENT_DIR

)


def _default_base_dir():

    """
    优先使用脚本所在工程（本机下载后直接能跑），
    否则回退到服务器上的绝对路径。
    """

    if os.path.isdir(

            os.path.join(

                LOCAL_PROJECT_DIR,

                "splits"

            )

    ):

        return LOCAL_PROJECT_DIR

    return REMOTE_BASE_DIR


BASE_DIR = _default_base_dir()


torch.multiprocessing.set_sharing_strategy(

    "file_system"

)


############################################################
# Import path helper
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


# 训练时使用的权重目录名模板

DEFAULT_CKPT_TEMPLATE = "{fold}_angle_gait_{model}"


# 数据目录默认层级

DEFAULT_FOLD_ROOT = "subject_5fold_split"


# 跨折汇总指标

SUMMARY_METRICS = [

    "angle_MAE",

    "angle_RMSE",

    "angle_R2",

    "gait_F1"

]


FOLD_RE = re.compile(

    r"^(fold\d+)",

    re.IGNORECASE

)


############################################################
# Model Builder
#
# 与 evaluate_angel_gait.py 完全一致，保证同一 checkpoint
# 得到与单 split 评估相同的模型结构。
############################################################


def build_model(

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
# Fold / checkpoint path resolution
############################################################


def fold_sort_key(name):

    match = re.search(

        r"(\d+)",

        name

    )

    return (

        int(

            match.group(1)

        ) if match else 0,

        name.lower()

    )


def discover_folds(

        base_dir,

        checkpoint_dir,

        fold_root,

        model_name

):
    """
    自动发现折名。优先看数据目录，其次看权重目录。
    返回规范化后的折名列表，例如 ["fold1", ..., "fold5"]。
    """

    found = set()

    if fold_root:

        group_dir = os.path.join(

            base_dir,

            "splits",

            fold_root

        )

        if os.path.isdir(

                group_dir

        ):

            for name in os.listdir(

                    group_dir

            ):

                match = FOLD_RE.match(

                    name

                )

                if match and os.path.isdir(

                        os.path.join(

                            group_dir,

                            name

                        )

                ):

                    found.add(

                        match.group(1).lower()

                    )

    if not found and model_name:

        model_dir = os.path.join(

            checkpoint_dir,

            model_name

        )

        if os.path.isdir(

                model_dir

        ):

            for name in os.listdir(

                    model_dir

            ):

                match = FOLD_RE.match(

                    name

                )

                if match and os.path.isdir(

                        os.path.join(

                            model_dir,

                            name

                        )

                ):

                    found.add(

                        match.group(1).lower()

                    )

    return sorted(

        found,

        key=fold_sort_key

    )


def ckpt_name_candidates(

        model_name,

        fold_root,

        fold,

        ckpt_style,

        ckpt_template

):
    """
    返回候补权重目录名（按优先级）。

    new    : ckpt_template，默认 fold3_angle_gait_GPT4TS
    legacy : fold3
    auto   : new -> <fold_root>_<fold>_angle_gait_<model> -> legacy
    """

    names = []

    def add(name):

        if name and name not in names:

            names.append(

                name

            )

    def expand(template):

        return template.format(

            fold=fold,

            model=model_name,

            fold_root=fold_root

        )

    if ckpt_style in ("auto", "new"):

        add(

            expand(

                ckpt_template

            )

        )

    if ckpt_style == "auto":

        add(

            expand(

                "{fold_root}_{fold}_angle_gait_{model}"

            )

        )

    if ckpt_style in ("auto", "legacy"):

        add(

            fold

        )

    return names


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


def resolve_checkpoint(

        checkpoint_dir,

        model_name,

        fold_root,

        fold,

        ckpt_style,

        ckpt_template

):
    """
    返回 (权重文件路径, 权重目录名, 尝试过的候选列表)。
    都没找到时路径为 None。
    """

    names = ckpt_name_candidates(

        model_name,

        fold_root,

        fold,

        ckpt_style,

        ckpt_template

    )

    tried = [

        get_checkpoint_path(

            checkpoint_dir,

            model_name,

            name

        )

        for name in names

    ]

    for name, path in zip(

            names,

            tried
    ):

        if os.path.isfile(

                path

        ):
            return path, name, tried

    return None, None, tried


def candidate_test_paths(

        base_dir,

        fold_root,

        fold

):
    """
    5 折布局优先，同时兼容平铺的 splits/<fold>/test.pkl。
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
# Checkpoint / architecture compatibility
############################################################


def load_state_dict_only(

        checkpoint_path,

        map_location

):
    """
    读出 checkpoint，返回 (state_dict, epoch, best_loss)。
    与 evaluate_angel_gait.py 的 load_checkpoint 相比，
    这里不做 load_state_dict，只取内容，便于先做结构比对。
    """

    checkpoint = torch.load(

        checkpoint_path,

        map_location=map_location

    )

    epoch = None

    best_loss = None

    if isinstance(

            checkpoint,

            dict

    ):

        epoch = checkpoint.get(

            "epoch"

        )

        best_loss = checkpoint.get(

            "best_loss"

        )

        if "model_state_dict" in checkpoint:

            checkpoint = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            checkpoint = checkpoint["state_dict"]

    state = {}

    for key, value in checkpoint.items():

        if key.startswith(

                "module."

        ):
            key = key.replace(

                "module.",

                ""

            )

        state[key] = value

    return state, epoch, best_loss


def compare_state_dict(

        model,

        state

):
    """
    比对 checkpoint 与当前模型代码的结构。

    返回 dict:
        missing     模型需要但 checkpoint 没有的键
        unexpected  checkpoint 多出来的键
        shape_diff  同名但形状不同的键
        compatible 能否直接 strict 加载
    """

    model_state = model.state_dict()

    model_keys = set(

        model_state.keys()

    )

    ckpt_keys = set(

        state.keys()

    )

    missing = sorted(

        model_keys - ckpt_keys

    )

    unexpected = sorted(

        ckpt_keys - model_keys

    )

    shape_diff = []

    for key in sorted(

            model_keys & ckpt_keys
    ):

        try:

            a = tuple(

                model_state[key].shape

            )

            b = tuple(

                state[key].shape

            )

        except AttributeError:

            continue

        if a != b:

            shape_diff.append(

                (

                    key,

                    a,

                    b

                )

            )

    return {

        "missing":

            missing,

        "unexpected":

            unexpected,

        "shape_diff":

            shape_diff,

        "compatible":

            not missing and not unexpected and not shape_diff

    }


def describe_mismatch(

        report,

        limit=4

):
    """
    把不匹配情况压成一行可读说明。
    """

    parts = []

    if report["missing"]:

        parts.append(

            f"缺少 {len(report['missing'])} 个键 "

            f"(如 {', '.join(report['missing'][:limit])})"

        )

    if report["unexpected"]:

        parts.append(

            f"多出 {len(report['unexpected'])} 个键 "

            f"(如 {', '.join(report['unexpected'][:limit])})"

        )

    if report["shape_diff"]:

        sample = "; ".join(

            f"{k}: model{a} vs ckpt{b}"

            for k, a, b in report["shape_diff"][:limit]

        )

        parts.append(

            f"形状不同 {len(report['shape_diff'])} 处 ({sample})"

        )

    if not parts:

        return "state_dict 与当前模型代码匹配"

    return "；".join(

        parts

    )


ARCH_HINT = (

    "该 checkpoint 的 state_dict 与 models/ 当前实现不是同一套结构，"

    "通常说明这份权重是用另一版模型代码训练出来的。"
    "请确认训练时的 models/ 与本仓库一致，或用当前代码重新训练该折。"

)


class StateDictMismatch(

        RuntimeError

):
    """
    权重结构与当前模型代码不匹配。
    单独成类，便于与 OOM 等其他 RuntimeError 区分。
    """


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

        device,

        max_batches=0

):
    model.eval()

    subject_task_records = []

    angle_records = []

    gait_records = []

    horizon_records = []

    with torch.no_grad():

        for batch_id, batch in enumerate(loader):

            if max_batches and batch_id >= max_batches:

                print(

                    "Reached --max_batches, stop early"

                )

                break

            imu = batch["imu"].to(

                device

            )

            output = model(

                imu

            )

            angle_pred = output["angle"].cpu().numpy()

            angle_true = batch["angle"].numpy()

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
# 单折输出的 7 个文件与 evaluate_angel_gait.py 保持一致。
############################################################


def to_builtin(value):

    """
    numpy / pandas 标量转原生类型，便于写 json。
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


def save_fold_summary(

        summary_dir,

        model_name,

        fold_root,

        fold_rows,

        skipped,

        ckpt_style,

        ckpt_template

):
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

        "ckpt_template":

            ckpt_template,

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

            f"+- {std_row.get(c, float('nan')):.6f}"

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


def import_dataset_class(base_dir):

    add_sys_paths(

        base_dir

    )

    from dataset_angle_gait import TrigaitMultiTaskDataset

    return TrigaitMultiTaskDataset


def evaluate_one_fold(

        model_name,

        fold,

        test_path,

        checkpoint_path,

        ckpt_name,

        output_dir,

        base_dir,

        device,

        batch_size,

        max_batches=0

):
    dataset_class = import_dataset_class(

        base_dir

    )

    model = build_model(

        model_name,

        device

    )

    state, epoch, best_loss = load_state_dict_only(

        checkpoint_path,

        device

    )

    report = compare_state_dict(

        model,

        state

    )

    if not report["compatible"]:

        raise StateDictMismatch(

            f"{model_name}/{fold} state_dict 与当前模型代码不匹配: "

            + describe_mismatch(

                report

            )

        )

    model.load_state_dict(

        state,

        strict=True

    )

    print(

        f"Checkpoint loaded (epoch={epoch}, best_loss={best_loss})"

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

        device,

        max_batches=max_batches

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

            output_dir,

        "epoch":

            epoch

    }

    for c in SUMMARY_METRICS:

        if c in overall:

            row[c] = float(

                overall[c]

            )

    return row


############################################################
# Preflight
#
# 只检查「路径 + state_dict 兼容性」，不加载数据集、不跑推理。
# 对 200MB 级别的 test.pkl 来说，这是最省时间的排查方式。
############################################################


def run_preflight(

        args,

        model_names,

        folds,

        base_dir,

        checkpoint_dir,

        eval_dir,

        device

):
    rows = []

    model_cache = {}

    print("=" * 78)

    print("PREFLIGHT: 只检查路径与 state_dict 兼容性，不运行推理")

    print("=" * 78)

    for model_name in model_names:

        for fold in folds:

            row = {

                "model":

                    model_name,

                "fold":

                    fold,

                "test_path":

                    "",

                "data_ok":

                    False,

                "ckpt_dir":

                    "",

                "ckpt_path":

                    "",

                "ckpt_ok":

                    False,

                "compatible":

                    False,

                "detail":

                    ""

            }

            ####################################################
            # 数据
            ####################################################

            test_path = (

                args.test

                if args.test

                else resolve_test_path(

                    base_dir,

                    args.fold_root,

                    fold

                )

            )

            row["test_path"] = test_path or ""

            row["data_ok"] = bool(

                test_path

                and os.path.isfile(

                    test_path

                )

            )

            ####################################################
            # 权重路径
            ####################################################

            checkpoint_path, ckpt_name, tried = resolve_checkpoint(

                checkpoint_dir,

                model_name,

                args.fold_root,

                fold,

                args.ckpt_style,

                args.ckpt_template

            )

            if checkpoint_path is None:

                row["detail"] = (

                    "权重缺失, 已尝试: "

                    + "; ".join(

                        tried

                    )

                )

                rows.append(

                    row

                )

                print(

                    f"[MISS] {model_name:<11} {fold:<6} "
                    f"data={'ok' if row['data_ok'] else 'MISSING'} "
                    f"ckpt=MISSING"

                )

                continue

            row["ckpt_dir"] = ckpt_name

            row["ckpt_path"] = checkpoint_path

            row["ckpt_ok"] = True

            ####################################################
            # 结构比对
            ####################################################

            try:

                if model_name not in model_cache:

                    model_cache[model_name] = build_model(

                        model_name,

                        device

                    )

                model = model_cache[model_name]

                state, epoch, best_loss = load_state_dict_only(

                    checkpoint_path,

                    device

                )

                report = compare_state_dict(

                    model,

                    state

                )

                row["compatible"] = report["compatible"]

                row["detail"] = describe_mismatch(

                    report

                )

                row["epoch"] = epoch

                flag = "OK  " if report["compatible"] else "BAD "

                print(

                    f"[{flag}] {model_name:<11} {fold:<6} "

                    f"data={'ok' if row['data_ok'] else 'MISSING'} "
                    f"ckpt={ckpt_name} "
                    f"keys={len(state)}/{len(model.state_dict())} "
                    f"epoch={epoch}"

                )

                if not report["compatible"]:

                    print(

                        f"        -> {row['detail']}"

                    )

                    print(

                        f"        -> {ARCH_HINT}"

                    )

            except Exception as exc:  # noqa: BLE001

                row["detail"] = f"{type(exc).__name__}: {exc}"

                print(

                    f"[ERR ] {model_name:<11} {fold:<6} {row['detail']}"

                )

            rows.append(

                row

            )

        model_cache.pop(

            model_name,

            None

        )

    ########################################################
    # 报告
    ########################################################

    os.makedirs(

        eval_dir,

        exist_ok=True

    )

    csv_path = os.path.join(

        eval_dir,

        "preflight_report.csv"

    )

    fields = [

        "model",

        "fold",

        "data_ok",

        "ckpt_ok",

        "compatible",

        "ckpt_dir",

        "test_path",

        "ckpt_path",

        "detail"

    ]

    with open(

            csv_path,

            "w",

            newline="",

            encoding="utf-8-sig"

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=fields,

            extrasaction="ignore"

        )

        writer.writeheader()

        for row in rows:

            writer.writerow(

                row

            )

    with open(

            os.path.join(

                eval_dir,

                "preflight_report.json"

            ),

            "w",

            encoding="utf-8"

    ) as f:

        json.dump(

            to_builtin(

                rows

            ),

            f,

            indent=4,

            ensure_ascii=False

        )

    ready = [

        f"{r['model']}/{r['fold']}"

        for r in rows

        if r["data_ok"] and r["ckpt_ok"] and r["compatible"]

    ]

    blocked = [

        f"{r['model']}/{r['fold']} ({r['detail']})"

        for r in rows

        if not (r["data_ok"] and r["ckpt_ok"] and r["compatible"])

    ]

    print("")

    print("=" * 78)

    print(

        f"可直接评估: {len(ready)} 组"

    )

    for item in ready:

        print(

            "   ",

            item

        )

    print(

        f"不可评估:   {len(blocked)} 组"

    )

    for item in blocked:

        print(

            "   ",

            item

        )

    print("=" * 78)

    print(

        "报告已写入:",

        csv_path

    )

    return rows


############################################################
# Main
############################################################


def main():
    parser = argparse.ArgumentParser(

        description=(

            "Trigait Angle+Gait 5-fold Evaluation (v2)"

        )

    )

    parser.add_argument(

        "--model",

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

            "一次评估多个模型，例如 --models TCN GPT4TS；"

            "传 all 表示除 Persistence 外的全部模型"

        )

    )

    parser.add_argument(

        "--folds",

        nargs="+",

        default=None,

        help="折名，例如 fold1 fold2；默认自动发现"

    )

    parser.add_argument(

        "--fold_root",

        default=DEFAULT_FOLD_ROOT,

        help="test.pkl 的上一层目录名，默认 subject_5fold_split"

    )

    parser.add_argument(

        "--ckpt_style",

        default="auto",

        choices=[

            "auto",

            "new",

            "legacy"

        ],

        help=(

            "权重目录命名方式: new=<fold>_angle_gait_<model>；"

            "legacy=foldN；auto=依次尝试"

        )

    )

    parser.add_argument(

        "--ckpt_template",

        default=DEFAULT_CKPT_TEMPLATE,

        help=(

            "权重目录名模板，可用 {fold} {model} {fold_root}，"

            "默认 {fold}_angle_gait_{model}"

        )

    )

    parser.add_argument(

        "--base_dir",

        default=None,

        help="工程根目录，默认自动定位到脚本所在工程"

    )

    parser.add_argument(

        "--checkpoint_dir",

        default=None,

        help="权重根目录，默认 <base_dir>/train/checkpoints"

    )

    parser.add_argument(

        "--eval_dir",

        default=None,

        help="结果根目录，默认 <base_dir>/evaluation_results"

    )

    parser.add_argument(

        "--test",

        default=None,

        help=(

            "只评估指定的单个 test.pkl（所有折共用同一个文件，"

            "一般用于调试）"

        )

    )

    parser.add_argument(

        "--batch_size",

        type=int,

        default=256

    )

    parser.add_argument(

        "--device",

        default="auto",

        choices=[

            "auto",

            "cuda",

            "cpu"

        ]

    )

    parser.add_argument(

        "--max_batches",

        type=int,

        default=0,

        help="只跑前 N 个 batch（调试用，0 表示全部）"

    )

    parser.add_argument(

        "--preflight",

        action="store_true",

        help="只做路径与结构体检，不运行推理"

    )

    parser.add_argument(

        "--strict",

        action="store_true",

        help="缺失或结构不匹配时直接报错，而不是跳过"

    )

    parser.add_argument(

        "--skip_existing",

        action="store_true",

        help="已存在 summary.json 的折跳过，便于断点续跑"

    )

    args = parser.parse_args()

    ########################################################
    # Paths
    ########################################################

    base_dir = args.base_dir or _default_base_dir()

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

    if args.device == "auto":

        device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

    else:

        device = torch.device(

            args.device

        )

    ########################################################
    # Models
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

    print(

        "Device:",

        device

    )

    print(

        "Base dir:",

        base_dir

    )

    print(

        "Checkpoint dir:",

        checkpoint_dir

    )

    ########################################################
    # Folds
    ########################################################

    if args.folds:

        folds_override = list(

            args.folds

        )

    else:

        folds_override = None

    ########################################################
    # Preflight
    ########################################################

    if args.preflight:

        folds = folds_override or discover_folds(

            base_dir,

            checkpoint_dir,

            args.fold_root,

            models_to_eval[0] if len(models_to_eval) == 1 else None

        )

        if not folds:

            print(

                "[ERROR] 没有发现任何折，请用 --folds 指定"

            )

            return

        print(

            "Folds:",

            " ".join(

                folds

            )

        )

        run_preflight(

            args,

            models_to_eval,

            folds,

            base_dir,

            checkpoint_dir,

            eval_dir,

            device

        )

        return

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

        if folds_override:

            folds_to_eval = folds_override

        else:

            folds_to_eval = discover_folds(

                base_dir,

                checkpoint_dir,

                args.fold_root,

                model_name

            )

        if not folds_to_eval:

            print(

                f"[WARN] {model_name}: 未发现任何折，跳过"

            )

            continue

        print("=" * 60)

        print(

            f"Model: {model_name}   Folds: {' '.join(folds_to_eval)}"

        )

        print("=" * 60)

        fold_rows = []

        skipped = []

        for fold in folds_to_eval:

            print("-" * 60)

            print(

                "Evaluate:",

                model_name,

                fold

            )

            output_dir = os.path.join(

                summary_dir,

                fold

            )

            ####################################################
            # 断点续跑
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
            # 数据
            ####################################################

            test_path = (

                args.test

                if args.test

                else resolve_test_path(

                    base_dir,

                    args.fold_root,

                    fold

                )

            )

            if not test_path or not os.path.isfile(

                    test_path

            ):

                message = (

                    f"{model_name}/{fold}: test.pkl 未找到, 候选: "

                    + "; ".join(

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
            # 权重
            ####################################################

            checkpoint_path, ckpt_name, tried = resolve_checkpoint(

                checkpoint_dir,

                model_name,

                args.fold_root,

                fold,

                args.ckpt_style,

                args.ckpt_template

            )

            if checkpoint_path is None:

                message = (

                    f"{model_name}/{fold}: 权重未找到, 候选: "

                    + "; ".join(

                        tried

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
            # 评估
            ####################################################

            try:

                row = evaluate_one_fold(

                    model_name,

                    fold,

                    test_path,

                    checkpoint_path,

                    ckpt_name,

                    output_dir,

                    base_dir,

                    device,

                    args.batch_size,

                    max_batches=args.max_batches

                )

            except StateDictMismatch as exc:

                if args.strict:

                    raise

                print(

                    "[SKIP]",

                    str(exc)

                )

                print(

                    "      ",

                    ARCH_HINT

                )

                skipped.append(

                    {

                        "fold":

                            fold,

                        "reason":

                            "state_dict incompatible",

                        "detail":

                            str(exc)

                    }

                )

                continue

            fold_rows.append(

                row

            )

        ########################################################
        # 跨折汇总
        ########################################################

        if fold_rows:

            model_summary = save_fold_summary(

                summary_dir,

                model_name,

                args.fold_root,

                fold_rows,

                skipped,

                args.ckpt_style,

                args.ckpt_template

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

                f"[WARN] {model_name}: 没有任何一折评估成功"

            )

        ########################################################
        # 运行状态落盘（便于断点续跑与排查）
        ########################################################

        os.makedirs(

            summary_dir,

            exist_ok=True

        )

        with open(

                os.path.join(

                    summary_dir,

                    "run_state.json"

                ),

                "w",

                encoding="utf-8"

        ) as f:

            json.dump(

                to_builtin(

                    {

                        "model":

                            model_name,

                        "fold_root":

                            args.fold_root,

                        "folds_requested":

                            folds_to_eval,

                        "folds_done":

                            [

                                r["fold"]

                                for r in fold_rows

                            ],

                        "skipped":

                            skipped

                    }

                ),

                f,

                indent=4,

                ensure_ascii=False

            )

    ########################################################
    # 全部模型汇总
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
