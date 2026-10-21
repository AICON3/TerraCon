# -*- coding: utf-8 -*-
################################################################################
# Trigait Terrain Regression Comprehensive Master Evaluation Framework
#
# Complete Production & Publication Grade Benchmark Suite (~1500 lines)
#
# Core Capabilities:
#   1. Universal Model Engine & Multi-Architecture Loader
#   2. Batch Model Sweep & Cross-Model Master Leaderboard Generation
#   3. Hardware Profiling (Latency, FPS, Flops, Peak VRAM, Params)
#   4. Multi-Protocol Data Partition Evaluation (Seen/Unseen, LOSO)
#   5. Dynamic Physical Calibration & Scaler Inversion (Denormalization)
#   6. 3D Spatial Geometry & Bilateral Euclidean Drift Decomposition
#   7. Gait Cycle Phase Segmentation (Stance vs Swing Dynamic Errors)
#   8. Discrete Locomotion Mode Classification Inversion & Confusion Matrix
#   9. Granular Horizon Progression (Frames 1-30 & Segment Bins)
#  10. Cross-Subject & Cross-Task Generalization Profiling
#  11. Statistical Hypothesis Testing (Paired t-test, Wilcoxon, p-value Matrix)
#  12. Multi-Sheet Professional Excel & Automatic LaTeX Table Compilation
################################################################################

import os
import sys
import time
import math
import glob
import json
import shutil
import logging
import argparse
import importlib
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.multiprocessing
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")

# ==============================================================================
# 1. Global Path Definitions & Environment Setup
# ==============================================================================

BASE_DIR = "/mnt/newdisk/dateset_0909/code/model_train_terrain"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "train", "checkpoints")
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation_results")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for p in [BASE_DIR, DATASETS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dataset_terrain import TrigaitTerrainDataset
except ImportError:
    TrigaitTerrainDataset = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TerrainMasterEval")

# ==============================================================================
# 2. Domain Knowledge Constants & Channel Topology
# ==============================================================================

CHANNEL_NAMES = [
    "ipsi_x",
    "ipsi_y",
    "ipsi_z",
    "contra_x",
    "contra_y",
    "contra_z",
]

AXIS_MAPPING = {
    "X_axis": [0, 3],  # Fore-aft displacement (Longitudinal)
    "Y_axis": [1, 4],  # Lateral step width (Mediolateral)
    "Z_axis": [2, 5],  # Vertical height / step obstacle (Elevation)
}

LIMB_MAPPING = {
    "Ipsilateral": [0, 1, 2],
    "Contralateral": [3, 4, 5],
}

TIME_HORIZON_BINS = {
    "Bin1_VeryShort (1-5 frames)": (0, 5),
    "Bin2_Short (6-10 frames)": (5, 10),
    "Bin3_Medium (11-20 frames)": (10, 20),
    "Bin4_Long (21-30 frames)": (20, 30),
}

DISCRETE_TERRAIN_CLASSES = [
    "LevelGround",
    "RampAscent",
    "RampDescent",
    "StairAscent",
    "StairDescent",
    "DiscreteFoothold"
]

MODEL_REGISTRY = {
    "TCN": ("tcn_terrain", "TCNTerrainRegression"),
    "LSTM": ("lstm_terrain", "LSTMTerrainRegression"),
    "Transformer": ("transformer_terrain", "TransformerTerrainRegression"),
    "GPT4TS": ("gpt4ts_terrain", "GPT4TSTerrainRegression"),
    "TimesNet": ("timesnet_terrain", "TimesNetTerrainRegression"),
    "PatchTST": ("patchtst_terrain", "PatchTSTTerrainRegression"),
    "TimeMixer": ("timemixer_terrain", "TimeMixerTerrainRegression"),
    "Mamba": ("mamba_terrain", "MambaTerrainRegression"),
    "Persistence": ("persistence_terrain", "PersistenceTerrainRegression"),
}


# ==============================================================================
# 3. Universal Model Instantiation & Parameter Normalization
# ==============================================================================

def build_model_instance(model_name, device, seq_len=200, pred_len=30, input_dim=42):
    """
    Instantiate model with strictly verified signatures aligned with train.py.
    """
    model_key = model_name.upper()
    found_key = None
    for k in MODEL_REGISTRY:
        if k.upper() == model_key:
            found_key = k
            break

    if found_key is None:
        raise ValueError(f"Model architecture '{model_name}' not recognized in MODEL_REGISTRY.")

    module_name, class_name = MODEL_REGISTRY[found_key]
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        logger.error(f"Failed importing module {module_name}: {e}")
        raise

    if not hasattr(mod, class_name):
        raise AttributeError(f"Module '{module_name}' lacks class '{class_name}'")

    cls = getattr(mod, class_name)

    if found_key == "TCN":
        model = cls(
            input_dim=input_dim,
            hidden_channels=[64, 128, 128],
            output_len=pred_len,
            dropout=0.2,
        )
    elif found_key == "LSTM":
        model = cls(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            pred_len=pred_len,
            dropout=0.2,
        )
    elif found_key == "Transformer":
        model = cls(
            input_dim=input_dim,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            output_len=pred_len,
            dropout=0.2,
        )
    elif found_key == "GPT4TS":
        model = cls(
            input_dim=input_dim,
            pred_len=pred_len,
            freeze=True,
        )
    elif found_key == "TimesNet":
        model = cls(
            input_len=seq_len,
            pred_len=pred_len,
            input_dim=input_dim,
            d_model=64,
            d_ff=128,
            num_layers=2,
            dropout=0.1,
        )
    elif found_key == "PatchTST":
        model = cls(
            input_len=seq_len,
            pred_len=pred_len,
            input_dim=input_dim,
            patch_len=16,
            stride=8,
        )
    elif found_key == "TimeMixer":
        model = cls(
            input_dim=input_dim,
            pred_len=pred_len,
            d_model=64,
        )
    elif found_key == "Mamba":
        model = cls(
            input_dim=input_dim,
            d_model=64,
            num_layers=2,
            pred_len=pred_len,
        )
    elif found_key == "Persistence":
        model = cls()
    else:
        model = cls()

    return model.to(device)


def load_checkpoint_safely(model, checkpoint_path, device):
    """
    Load PyTorch checkpoint weights, handling DataParallel/DDP module prefixes.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing weights file at: {checkpoint_path}")

    logger.info(f"Loading checkpoint weights: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    state = (
            ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt
    )

    clean_state = {}
    for k, v in state.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        clean_state[name] = v

    try:
        model.load_state_dict(clean_state, strict=True)
    except Exception as e:
        logger.warning(f"Strict load failed ({e}), attempting non-strict match...")
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        logger.warning(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.eval()
    return model


# ==============================================================================
# 4. Hardware Profiler (Latency, FLOPs, Memory, Throughput)
# ==============================================================================

class HardwareProfiler:
    """
    Evaluates execution efficiency on edge and workstation devices.
    """

    @staticmethod
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"total_params": total, "trainable_params": trainable}

    @staticmethod
    def measure_latency_and_memory(model, device, input_shape=(1, 200, 42), warmup=30, runs=100):
        model.eval()
        dummy_input = torch.randn(*input_shape, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            timings = []
            with torch.no_grad():
                for _ in range(runs):
                    start_event.record()
                    _ = model(dummy_input)
                    end_event.record()
                    torch.cuda.synchronize()
                    timings.append(start_event.elapsed_time(end_event))  # in ms

            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
            mean_latency = float(np.mean(timings))
            std_latency = float(np.std(timings))
            fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
        else:
            timings = []
            with torch.no_grad():
                for _ in range(runs):
                    t0 = time.perf_counter()
                    _ = model(dummy_input)
                    t1 = time.perf_counter()
                    timings.append((t1 - t0) * 1000.0)  # in ms
            peak_vram = 0.0
            mean_latency = float(np.mean(timings))
            std_latency = float(np.std(timings))
            fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

        # Rough FLOPs estimation
        flops = HardwareProfiler.estimate_flops(model, dummy_input)

        return {
            "latency_mean_ms": round(mean_latency, 3),
            "latency_std_ms": round(std_latency, 3),
            "throughput_fps": round(fps, 1),
            "peak_memory_mb": round(peak_vram, 2),
            "macs_mflops": round(flops / 1e6, 2)
        }

    @staticmethod
    def estimate_flops(model, sample_input):
        # Fallback simplistic FLOP estimator
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(model, display=False) as fcm:
                _ = model(sample_input)
            return float(fcm.get_total_flops())
        except Exception:
            params = sum(p.numel() for p in model.parameters())
            # Rule of thumb: ~2 FLOPs per parameter per forward pass
            return float(params * 2 * sample_input.shape[1])


# ==============================================================================
# 5. Core Mathematical & Regression Metrics Engine
# ==============================================================================

def compute_1d_metrics(y_true, y_pred):
    """
    Computes regression statistics for 1D paired observations.
    """
    t = np.asarray(y_true, dtype=np.float64).flatten()
    p = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(t) == 0:
        return {
            "MAE": np.nan, "RMSE": np.nan, "NRMSE": np.nan, "MaxError": np.nan,
            "Std_AE": np.nan, "R2": np.nan, "Pearson": np.nan, "Spearman": np.nan,
            "MAPE": np.nan, "Sample_Count": 0
        }

    diff = t - p
    abs_err = np.abs(diff)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(abs_err))
    std_ae = float(np.std(abs_err))

    # Normalized RMSE by range
    val_range = np.max(t) - np.min(t)
    nrmse = float(rmse / val_range) if val_range > 1e-6 else np.nan

    # R2 Determination Coefficient
    try:
        r2 = float(r2_score(t, p))
    except Exception:
        r2 = np.nan

    # Pearson Correlation Coefficient
    if np.std(t) < 1e-7 or np.std(p) < 1e-7 or len(t) < 2:
        pearson_val = np.nan
        spearman_val = np.nan
    else:
        try:
            pearson_val = float(pearsonr(t, p)[0])
        except Exception:
            pearson_val = np.nan
        try:
            spearman_val = float(spearmanr(t, p)[0])
        except Exception:
            spearman_val = np.nan

    # Mean Absolute Percentage Error (clamped denominator)
    denom = np.abs(t)
    denom = np.where(denom < 1e-4, 1e-4, denom)
    mape = float(np.mean(abs_err / denom) * 100.0)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "MaxError": max_err,
        "Std_AE": std_ae,
        "R2": r2,
        "Pearson": pearson_val,
        "Spearman": spearman_val,
        "MAPE": mape,
        "Sample_Count": int(len(t))
    }


def compute_3d_euclidean_drift(true_xyz, pred_xyz):
    """
    Computes 3D Euclidean spatial foot placement drift.
    true_xyz, pred_xyz: [N, 3]
    """
    if len(true_xyz) == 0:
        return {
            "Euclidean_Mean": np.nan, "Euclidean_RMSE": np.nan, "Euclidean_Std": np.nan,
            "Euclidean_Max": np.nan, "Euclidean_Median": np.nan, "Euclidean_P95": np.nan
        }

    deltas = true_xyz - pred_xyz
    eucl_distances = np.linalg.norm(deltas, axis=-1)

    return {
        "Euclidean_Mean": float(np.mean(eucl_distances)),
        "Euclidean_RMSE": float(np.sqrt(np.mean(eucl_distances ** 2))),
        "Euclidean_Std": float(np.std(eucl_distances)),
        "Euclidean_Max": float(np.max(eucl_distances)),
        "Euclidean_Median": float(np.median(eucl_distances)),
        "Euclidean_P95": float(np.percentile(eucl_distances, 95))
    }


# ==============================================================================
# 6. Physical Calibrator & Gait Phase Segmentation Engine
# ==============================================================================

class PhysicalScalerEngine:
    """
    Restores normalized arrays back into real-world physical coordinates (cm, degrees).
    """

    def __init__(self, dataset_handle):
        self.ds = dataset_handle

    def denormalize(self, data_array):
        """
        data_array: numpy array [..., 6]
        """
        if self.ds is None:
            return data_array

        if hasattr(self.ds, "denormalize_terrain"):
            return self.ds.denormalize_terrain(data_array)

        if hasattr(self.ds, "terrain_mean") and hasattr(self.ds, "terrain_std"):
            mean = np.array(self.ds.terrain_mean).reshape(1, 1, -1)
            std = np.array(self.ds.terrain_std).reshape(1, 1, -1)
            return data_array * (std + 1e-6) + mean

        return data_array


class LocomotionContextClassifier:
    """
    Derives discrete terrain classes from continuous spatial trajectories.
    """

    @staticmethod
    def infer_discrete_class(terrain_6d):
        """
        Classifies sample into discrete categories based on z-slope and variation.
        terrain_6d: [30, 6]
        """
        z_ipsi = terrain_6d[:, 2]
        z_contra = terrain_6d[:, 5]
        z_all = np.concatenate([z_ipsi, z_contra])

        delta_z = np.max(z_all) - np.min(z_all)
        mean_slope = (z_all[-1] - z_all[0])

        if delta_z < 0.03:  # Threshold 3 cm
            return "LevelGround"
        elif mean_slope > 0.05:
            return "StairAscent" if delta_z > 0.12 else "RampAscent"
        elif mean_slope < -0.05:
            return "StairDescent" if delta_z > 0.12 else "RampDescent"
        else:
            return "DiscreteFoothold"


# ==============================================================================
# 7. Comprehensive Single-Model Evaluation Pipeline
# ==============================================================================

def run_evaluation_on_loader(model, dataloader, scaler, device):
    """
    Executes full multi-dimensional evaluation sweep on a given dataset partition.
    """
    model.eval()

    sample_records = []
    channel_records = []
    axis_records = []
    limb_records = []
    euclidean_records = []
    phase_records = []
    classification_true = []
    classification_pred = []

    all_true_arr = []
    all_pred_arr = []
    all_mask_arr = []
    all_subject_tags = []
    all_task_tags = []

    logger.info("Executing forward inference across batches...")
    start_time = time.time()

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            imu = batch["imu"].to(device)
            out = model(imu)

            p_raw = out["terrain"].detach().cpu().numpy()
            t_raw = batch["terrain"].numpy()
            m_raw = batch["terrain_mask"].numpy()

            # Physical units denormalization
            p_denorm = scaler.denormalize(p_raw)
            t_denorm = scaler.denormalize(t_raw)

            all_true_arr.append(t_denorm)
            all_pred_arr.append(p_denorm)
            all_mask_arr.append(m_raw)

            subjects = batch.get("subject", ["S_Unknown"] * len(p_raw))
            tasks = batch.get("task", ["T_Unknown"] * len(p_raw))

            all_subject_tags.extend(subjects)
            all_task_tags.extend(tasks)

            batch_sz = len(subjects)

            for i in range(batch_sz):
                sub = str(subjects[i])
                tsk = str(tasks[i])
                t_i = t_denorm[i]  # [30, 6]
                p_i = p_denorm[i]  # [30, 6]
                m_i = m_raw[i]  # [30, 6]

                # 1. Global Sample Level
                v_all = m_i > 0
                if v_all.sum() > 0:
                    m_stat = compute_1d_metrics(t_i[v_all], p_i[v_all])
                    m_stat.update({"subject": sub, "task": tsk})
                    sample_records.append(m_stat)

                # 2. Channel Level (6 Channels)
                for c_idx, c_name in enumerate(CHANNEL_NAMES):
                    v_c = m_i[:, c_idx] > 0
                    if v_c.sum() > 0:
                        c_stat = compute_1d_metrics(t_i[:, c_idx][v_c], p_i[:, c_idx][v_c])
                        c_stat.update({"subject": sub, "task": tsk, "channel": c_name})
                        channel_records.append(c_stat)

                # 3. Axis Level (X, Y, Z grouped)
                for ax_name, ax_cols in AXIS_MAPPING.items():
                    v_ax = m_i[:, ax_cols] > 0
                    if v_ax.sum() > 0:
                        ax_stat = compute_1d_metrics(t_i[:, ax_cols][v_ax], p_i[:, ax_cols][v_ax])
                        ax_stat.update({"subject": sub, "task": tsk, "axis": ax_name})
                        axis_records.append(ax_stat)

                # 4. Bilateral Limb Symmetry (Ipsilateral vs Contralateral)
                for limb_name, limb_cols in LIMB_MAPPING.items():
                    v_limb = m_i[:, limb_cols] > 0
                    if v_limb.sum() > 0:
                        limb_stat = compute_1d_metrics(t_i[:, limb_cols][v_limb], p_i[:, limb_cols][v_limb])
                        limb_stat.update({"subject": sub, "task": tsk, "limb": limb_name})
                        limb_records.append(limb_stat)

                # 5. 3D Spatial Euclidean Drift (Point-to-Point)
                # Ipsilateral Foot [0, 1, 2]
                v_ipsi = (m_i[:, [0, 1, 2]] > 0).all(axis=-1)
                if v_ipsi.sum() > 0:
                    ipsi_eucl = compute_3d_euclidean_drift(t_i[:, [0, 1, 2]][v_ipsi], p_i[:, [0, 1, 2]][v_ipsi])
                    ipsi_eucl.update({"subject": sub, "task": tsk, "side": "ipsilateral"})
                    euclidean_records.append(ipsi_eucl)

                # Contralateral Foot [3, 4, 5]
                v_contra = (m_i[:, [3, 4, 5]] > 0).all(axis=-1)
                if v_contra.sum() > 0:
                    contra_eucl = compute_3d_euclidean_drift(t_i[:, [3, 4, 5]][v_contra], p_i[:, [3, 4, 5]][v_contra])
                    contra_eucl.update({"subject": sub, "task": tsk, "side": "contralateral"})
                    euclidean_records.append(contra_eucl)

                # 6. Gait Phase Segmentation (Stance: early 60%, Swing: late 40%)
                stance_mask = m_i[:18, :] > 0
                swing_mask = m_i[18:, :] > 0
                if stance_mask.sum() > 0:
                    st_m = compute_1d_metrics(t_i[:18, :][stance_mask], p_i[:18, :][stance_mask])
                    st_m.update({"subject": sub, "task": tsk, "gait_phase": "Stance_Phase (0-60%)"})
                    phase_records.append(st_m)
                if swing_mask.sum() > 0:
                    sw_m = compute_1d_metrics(t_i[18:, :][swing_mask], p_i[18:, :][swing_mask])
                    sw_m.update({"subject": sub, "task": tsk, "gait_phase": "Swing_Phase (60-100%)"})
                    phase_records.append(sw_m)

                # 7. Discrete Locomotion Mode Classification Inversion
                cls_t = LocomotionContextClassifier.infer_discrete_class(t_i)
                cls_p = LocomotionContextClassifier.infer_discrete_class(p_i)
                classification_true.append(cls_t)
                classification_pred.append(cls_p)

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(dataloader):
                elapsed = time.time() - start_time
                logger.info(f"  Batch [{b_idx + 1}/{len(dataloader)}] finished | Elapsed: {elapsed:.1f}s")

    all_true_arr = np.concatenate(all_true_arr, axis=0)  # [N, 30, 6]
    all_pred_arr = np.concatenate(all_pred_arr, axis=0)  # [N, 30, 6]
    all_mask_arr = np.concatenate(all_mask_arr, axis=0)  # [N, 30, 6]

    # 8. Step-by-Step Horizon Decay (1 to 30)
    horizon_frame_records = []
    pred_steps = all_true_arr.shape[1]
    for st in range(pred_steps):
        h_rec = {"frame": st + 1}
        st_v = all_mask_arr[:, st, :] > 0
        if st_v.sum() > 0:
            m_step = compute_1d_metrics(all_true_arr[:, st, :][st_v], all_pred_arr[:, st, :][st_v])
            h_rec["Overall_MAE"] = m_step["MAE"]
            h_rec["Overall_RMSE"] = m_step["RMSE"]
            h_rec["Overall_R2"] = m_step["R2"]
            h_rec["Overall_Pearson"] = m_step["Pearson"]
        else:
            h_rec["Overall_MAE"] = np.nan
            h_rec["Overall_RMSE"] = np.nan
            h_rec["Overall_R2"] = np.nan
            h_rec["Overall_Pearson"] = np.nan

        # Per-axis step breakdown
        for ax_name, ax_cols in AXIS_MAPPING.items():
            ax_st_v = all_mask_arr[:, st, ax_cols] > 0
            if ax_st_v.sum() > 0:
                ax_st_m = compute_1d_metrics(all_true_arr[:, st, ax_cols][ax_st_v],
                                             all_pred_arr[:, st, ax_cols][ax_st_v])
                h_rec[f"{ax_name}_MAE"] = ax_st_m["MAE"]
                h_rec[f"{ax_name}_RMSE"] = ax_st_m["RMSE"]
            else:
                h_rec[f"{ax_name}_MAE"] = np.nan
                h_rec[f"{ax_name}_RMSE"] = np.nan

        horizon_frame_records.append(h_rec)

    # 9. Horizon Binned Progression
    horizon_binned_records = []
    for b_name, (b_s, b_e) in TIME_HORIZON_BINS.items():
        bin_v = all_mask_arr[:, b_s:b_e, :] > 0
        bin_rec = {"horizon_bin": b_name}
        if bin_v.sum() > 0:
            b_metrics = compute_1d_metrics(all_true_arr[:, b_s:b_e, :][bin_v], all_pred_arr[:, b_s:b_e, :][bin_v])
            bin_rec.update(b_metrics)
        horizon_binned_records.append(bin_rec)

    # 10. Classification Metrics (Accuracy, Macro-F1, Confusion Matrix)
    cls_acc = accuracy_score(classification_true, classification_pred)
    cls_f1 = f1_score(classification_true, classification_pred, average="macro", zero_division=0)
    cls_prec = precision_score(classification_true, classification_pred, average="macro", zero_division=0)
    cls_rec = recall_score(classification_true, classification_pred, average="macro", zero_division=0)

    unique_labels = sorted(list(set(classification_true + classification_pred)))
    cm = confusion_matrix(classification_true, classification_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{x}" for x in unique_labels], columns=[f"Pred_{x}" for x in unique_labels])

    cls_summary = {
        "Accuracy": float(cls_acc),
        "Macro_F1": float(cls_f1),
        "Macro_Precision": float(cls_prec),
        "Macro_Recall": float(cls_rec)
    }

    return {
        "sample_records": sample_records,
        "channel_records": channel_records,
        "axis_records": axis_records,
        "limb_records": limb_records,
        "euclidean_records": euclidean_records,
        "phase_records": phase_records,
        "horizon_frame_records": horizon_frame_records,
        "horizon_binned_records": horizon_binned_records,
        "classification_summary": cls_summary,
        "confusion_matrix_df": cm_df,
        "raw_pair_cache": (all_true_arr, all_pred_arr, all_mask_arr)
    }


# ==============================================================================
# 8. Statistical Significance & Hypothesis Testing Module
# ==============================================================================

class StatisticalHypothesisTester:
    """
    Computes paired t-tests and non-parametric Wilcoxon tests across models.
    """

    @staticmethod
    def compare_two_models(errors_model_a, errors_model_b):
        """
        errors_model_a, errors_model_b: 1D numpy arrays of sample-level absolute errors.
        """
        a = np.asarray(errors_model_a, dtype=np.float64)
        b = np.asarray(errors_model_b, dtype=np.float64)

        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

        if min_len < 5:
            return {"t_stat": np.nan, "p_value_ttest": np.nan, "wilcoxon_stat": np.nan, "p_value_wilcoxon": np.nan}

        # Paired t-test
        t_stat, p_t = ttest_rel(a, b, nan_policy="omit")

        # Wilcoxon signed-rank test
        try:
            diff = a - b
            diff = diff[diff != 0]
            if len(diff) > 5:
                w_stat, p_w = wilcoxon(diff)
            else:
                w_stat, p_w = np.nan, np.nan
        except Exception:
            w_stat, p_w = np.nan, np.nan

        return {
            "t_stat": float(t_stat),
            "p_value_ttest": float(p_t),
            "significant_ttest": bool(p_t < 0.05),
            "wilcoxon_stat": float(w_stat),
            "p_value_wilcoxon": float(p_w),
            "significant_wilcoxon": bool(p_w < 0.05)
        }


# ==============================================================================
# 9. LaTeX Table & Academic Report Generator
# ==============================================================================

class AcademicReportFormatter:
    """
    Converts metrics dataframes into ready-to-paste IEEE/ASME LaTeX publication tables.
    """

    @staticmethod
    def generate_latex_table(df, caption="Terrain Regression Performance", label="tab:terrain_perf"):
        latex_str = "\\begin{table*}[t]\n\\centering\n"
        latex_str += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        latex_str += "\\begin{tabular}{l" + "c" * (len(df.columns) - 1) + "}\n\\toprule\n"

        # Headers
        headers = [f"\\textbf{{{c}}}" for c in df.columns]
        latex_str += " & ".join(headers) + " \\\\\n\\midrule\n"

        # Rows
        for _, row in df.iterrows():
            row_items = []
            for val in row:
                if isinstance(val, (int, np.integer)):
                    row_items.append(f"{val}")
                elif isinstance(val, (float, np.floating)):
                    row_items.append(f"{val:.3f}")
                else:
                    row_items.append(str(val))
            latex_str += " & ".join(row_items) + " \\\\\n"

        latex_str += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
        return latex_str


# ==============================================================================
# 10. Multi-Sheet High-Density Excel Workbook Packaging
# ==============================================================================

def export_single_model_excel(eval_bundle, hw_profile, out_dir):
    """
    Compiles 12 detailed sheets into an exhaustive evaluation workbook.
    """
    os.makedirs(out_dir, exist_ok=True)
    target_excel = os.path.join(out_dir, "Full_Terrain_Evaluation_Book.xlsx")
    logger.info(f"Writing high-density analysis workbook to: {target_excel}")

    df_sample = pd.DataFrame(eval_bundle["sample_records"])
    df_chan = pd.DataFrame(eval_bundle["channel_records"])
    df_axis = pd.DataFrame(eval_bundle["axis_records"])
    df_limb = pd.DataFrame(eval_bundle["limb_records"])
    df_eucl = pd.DataFrame(eval_bundle["euclidean_records"])
    df_phase = pd.DataFrame(eval_bundle["phase_records"])
    df_h_frame = pd.DataFrame(eval_bundle["horizon_frame_records"])
    df_h_bin = pd.DataFrame(eval_bundle["horizon_binned_records"])

    # Aggregations
    axis_summary = df_axis.groupby("axis").mean(numeric_only=True).reset_index()
    chan_summary = df_chan.groupby("channel").mean(numeric_only=True).reset_index()
    limb_summary = df_limb.groupby("limb").mean(numeric_only=True).reset_index()
    if len(df_eucl) > 0 and "side" in df_eucl.columns:

        eucl_summary = (
            df_eucl
            .groupby("side")
            .mean(numeric_only=True)
            .reset_index()
        )

    else:

        eucl_summary = pd.DataFrame(
            {
                "side": ["unknown"],
                "Euclidean_Mean": [np.nan],
                "Euclidean_RMSE": [np.nan],
                "Euclidean_Std": [np.nan],
                "Euclidean_Max": [np.nan],
                "Euclidean_Median": [np.nan],
                "Euclidean_P95": [np.nan],
            }
        )
        
    phase_summary = df_phase.groupby("gait_phase").mean(numeric_only=True).reset_index()
    sub_summary = df_sample.groupby("subject").mean(numeric_only=True).reset_index()
    task_summary = df_sample.groupby("task").mean(numeric_only=True).reset_index()

    # Hardware & Overall Overview
    global_dict = df_sample.mean(numeric_only=True).to_dict()
    global_dict.update(hw_profile)
    global_dict.update(eval_bundle["classification_summary"])
    df_overview = pd.DataFrame(list(global_dict.items()), columns=["Metric_Item", "Value"])

    with pd.ExcelWriter(target_excel, engine="openpyxl") as writer:
        df_overview.to_excel(writer, sheet_name="00_Executive_Overview", index=False)
        axis_summary.to_excel(writer, sheet_name="01_Spatial_Axis_XYZ", index=False)
        eucl_summary.to_excel(writer, sheet_name="02_3D_Euclidean_Drift", index=False)
        phase_summary.to_excel(writer, sheet_name="03_Gait_Phase_Analysis", index=False)
        limb_summary.to_excel(writer, sheet_name="04_Bilateral_Limb_Symm", index=False)
        chan_summary.to_excel(writer, sheet_name="05_Channels_6Dim", index=False)
        df_h_bin.to_excel(writer, sheet_name="06_Horizon_Bins", index=False)
        df_h_frame.to_excel(writer, sheet_name="07_Horizon_30Steps", index=False)
        task_summary.to_excel(writer, sheet_name="08_Task_Contexts", index=False)
        sub_summary.to_excel(writer, sheet_name="09_Subject_Generalization", index=False)
        eval_bundle["confusion_matrix_df"].to_excel(writer, sheet_name="10_Locomotion_CM")
        df_sample.to_excel(writer, sheet_name="11_Sample_Level_Dump", index=False)

    # Master JSON Summary for API hooks
    master_json = {
        "Executive_Overview": global_dict,
        "Axis_Summary": axis_summary.set_index("axis").to_dict(orient="index"),
        "Euclidean_Summary":
            eucl_summary.set_index("side").to_dict(orient="index")
            if "side" in eucl_summary.columns
            else {},
        "Phase_Summary": phase_summary.set_index("gait_phase").to_dict(orient="index"),
        "Task_Summary": task_summary.set_index("task").to_dict(orient="index"),
        "Subject_Summary": sub_summary.set_index("subject").to_dict(orient="index")
    }

    with open(os.path.join(out_dir, "master_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(master_json, f, indent=4, ensure_ascii=False)

    # Save LaTeX table code snippet
    latex_code = AcademicReportFormatter.generate_latex_table(axis_summary, caption="Spatial Axis Error Breakdown (cm)")
    with open(os.path.join(out_dir, "table_spatial_axis.tex"), "w") as f:
        f.write(latex_code)

    logger.info(f"Model single-run report completed: {out_dir}")


# ==============================================================================
# 11. Cross-Model Master Leaderboard Sweeper
# ==============================================================================

def execute_cross_model_benchmark_sweep(test_path, split_name, batch_size, device):
    """
    Automatically detects all trained models, runs benchmarks, and constructs master ranking.
    """
    logger.info("=" * 80)
    logger.info("INITIATING CROSS-MODEL BENCHMARK MASTER SWEEP")
    logger.info("=" * 80)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset path does not exist: {test_path}")

    ds_test = TrigaitTerrainDataset(test_path)
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)
    scaler = PhysicalScalerEngine(ds_test)

    sweep_dir = os.path.join(EVALUATION_DIR, "Master_Leaderboard_Sweep", split_name)
    os.makedirs(sweep_dir, exist_ok=True)

    master_summary_rows = []
    sample_error_cache = {}

    available_models = [m for m in MODEL_REGISTRY.keys()]

    for m_name in available_models:
        ckpt_p = os.path.join(CHECKPOINT_DIR, m_name, split_name, "best_model.pth")
        if not os.path.isfile(ckpt_p):
            # Also check lowercase directory
            ckpt_p_lower = os.path.join(CHECKPOINT_DIR, m_name.lower(), split_name, "best_model.pth")
            if os.path.isfile(ckpt_p_lower):
                ckpt_p = ckpt_p_lower
            else:
                logger.warning(f"[-] Checkpoint for model {m_name} not found, skipping.")
                continue

        logger.info(f"\n[+] Processing Benchmark candidate: {m_name}")
        try:
            model = build_model_instance(m_name, device)
            model = load_checkpoint_safely(model, ckpt_p, device)

            # Profiling
            hw_prof = HardwareProfiler.measure_latency_and_memory(model, device)
            params_dict = HardwareProfiler.count_parameters(model)
            hw_prof.update(params_dict)

            # Evaluation
            eval_bundle = run_evaluation_on_loader(model, loader, scaler, device)

            # Export individual bundle
            m_out_dir = os.path.join(EVALUATION_DIR, m_name, split_name)
            export_single_model_excel(eval_bundle, hw_prof, m_out_dir)

            # Cache error distributions for hypothesis testing
            sample_df = pd.DataFrame(eval_bundle["sample_records"])
            sample_error_cache[m_name] = sample_df["MAE"].to_numpy()

            # Record leader board metrics
            axis_df = pd.DataFrame(eval_bundle["axis_records"]).groupby("axis").mean(numeric_only=True)
            df_eucl = pd.DataFrame(eval_bundle["euclidean_records"])

            if len(df_eucl) > 0:
                eucl_df = df_eucl.mean(numeric_only=True)
            else:
                eucl_df = pd.Series()
                
            row = {
                "Model": m_name,
                "Overall_MAE": float(sample_df["MAE"].mean()),
                "Overall_RMSE": float(sample_df["RMSE"].mean()),
                "Overall_R2": float(sample_df["R2"].mean()),
                "Pearson_Corr": float(sample_df["Pearson"].mean()),
                "3D_Euclidean_Mean": float(eucl_df.get("Euclidean_Mean", np.nan)),
                "3D_Euclidean_RMSE": float(eucl_df.get("Euclidean_RMSE", np.nan)),
                "X_axis_MAE": float(axis_df.loc["X_axis", "MAE"]) if "X_axis" in axis_df.index else np.nan,
                "Y_axis_MAE": float(axis_df.loc["Y_axis", "MAE"]) if "Y_axis" in axis_df.index else np.nan,
                "Z_axis_MAE": float(axis_df.loc["Z_axis", "MAE"]) if "Z_axis" in axis_df.index else np.nan,
                "Latency_ms": hw_prof["latency_mean_ms"],
                "FPS": hw_prof["throughput_fps"],
                "Params_Total": hw_prof["total_params"],
                "Macro_F1": eval_bundle["classification_summary"]["Macro_F1"]
            }
            master_summary_rows.append(row)

        except Exception as e:
            logger.error(f"Failed evaluating candidate {m_name}: {e}")
            traceback.print_exc()

    if not master_summary_rows:
        logger.error("No models were successfully evaluated during master sweep.")
        return

    # Create Master Leaderboard DataFrame
    master_df = pd.DataFrame(master_summary_rows)
    master_df = master_df.sort_values(by="Overall_MAE", ascending=True).reset_index(drop=True)
    master_df["Rank"] = range(1, len(master_df) + 1)

    # Reorder columns
    cols = ["Rank", "Model", "Overall_MAE", "Overall_RMSE", "Overall_R2", "Pearson_Corr",
            "3D_Euclidean_Mean", "Z_axis_MAE", "X_axis_MAE", "Y_axis_MAE",
            "Latency_ms", "FPS", "Params_Total", "Macro_F1"]
    cols = [c for c in cols if c in master_df.columns]
    master_df = master_df[cols]

    # Statistical significance matrix calculation
    tested_models = list(sample_error_cache.keys())
    p_matrix = pd.DataFrame(np.nan, index=tested_models, columns=tested_models)
    for m1 in tested_models:
        for m2 in tested_models:
            if m1 == m2:
                p_matrix.loc[m1, m2] = 1.0
            else:
                res = StatisticalHypothesisTester.compare_two_models(sample_error_cache[m1], sample_error_cache[m2])
                p_matrix.loc[m1, m2] = res["p_value_ttest"]

    # Export Leaderboard Artifacts
    master_excel = os.path.join(sweep_dir, "Master_Model_Leaderboard.xlsx")
    with pd.ExcelWriter(master_excel, engine="openpyxl") as writer:
        master_df.to_excel(writer, sheet_name="Master_Rankings", index=False)
        p_matrix.to_excel(writer, sheet_name="Paired_TTest_PValues")

    master_df.to_csv(os.path.join(sweep_dir, "leaderboard.csv"), index=False)

    # Save Publication Ready LaTeX Master Table
    latex_table = AcademicReportFormatter.generate_latex_table(
        master_df.drop(columns=["Params_Total"]),
        caption=f"Comparative Benchmark on Gait Terrain Regression ({split_name})",
        label="tab:master_leaderboard"
    )
    with open(os.path.join(sweep_dir, "master_leaderboard.tex"), "w") as f:
        f.write(latex_table)

    logger.info("=" * 80)
    logger.info(f"MASTER SWEEP COMPLETED SUCCESSFULLY! Artifacts: {sweep_dir}")
    print("\n" + master_df.to_string(index=False) + "\n")
    logger.info("=" * 80)


# ==============================================================================
# 12. Main Command Line Interface
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Master Full-Scale Evaluation Pipeline for Gait Terrain Regression"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ALL",
        help="Model architecture name (TCN, LSTM, Transformer, GPT4TS, TimesNet, PatchTST, TimeMixer, Mamba, Persistence) or 'ALL' to sweep everything."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default",
        help="Experiment checkpoint split directory name (e.g. subject_scene_split, seen_unseen)"
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test partition file (e.g. test_data.pt or processed_test.pkl)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Inference batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Autodetected if None."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Initialized Terrain Master Evaluator on Compute Device: {device}")

    # Case A: Sweep all models and construct publication master leaderboard
    if args.model.upper() == "ALL":
        execute_cross_model_benchmark_sweep(
            test_path=args.test,
            split_name=args.name,
            batch_size=args.batch_size,
            device=device
        )
        return

    # Case B: Standard single model deep evaluation
    target_model = args.model
    ckpt_path = os.path.join(CHECKPOINT_DIR, target_model, args.name, "best_model.pth")
    if not os.path.isfile(ckpt_path):
        ckpt_path_alt = os.path.join(CHECKPOINT_DIR, target_model.lower(), args.name, "best_model.pth")
        if os.path.isfile(ckpt_path_alt):
            ckpt_path = ckpt_path_alt
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # 1. Dataset loading
    logger.info(f"Loading test dataset from: {args.test}")
    if not os.path.exists(args.test):
        raise FileNotFoundError(f"Missing dataset at {args.test}")

    ds_test = TrigaitTerrainDataset(args.test)
    loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    logger.info(f"Total evaluation samples: {len(ds_test)}")
    scaler = PhysicalScalerEngine(ds_test)

    # 2. Build model & profile hardware
    model = build_model_instance(target_model, device)
    model = load_checkpoint_safely(model, ckpt_path, device)

    logger.info("Profiling model latency and memory bounds...")
    hw_prof = HardwareProfiler.measure_latency_and_memory(model, device)
    hw_prof.update(HardwareProfiler.count_parameters(model))

    # 3. Comprehensive evaluation sweep
    eval_bundle = run_evaluation_on_loader(model, loader, scaler, device)

    # 4. Multi-sheet Excel and academic exports
    out_dir = os.path.join(EVALUATION_DIR, target_model, args.name)
    export_single_model_excel(eval_bundle, hw_prof, out_dir)

    logger.info("=" * 70)
    logger.info(f"ALL EVALUATIONS SUCCESSFUL FOR MODEL: {target_model}")
    logger.info(f"Output directory: {out_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()