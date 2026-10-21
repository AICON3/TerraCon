# -*- coding: utf-8 -*-
"""Scene/axis terrain evaluation. No cross-scene or cross-axis averaging.
Single split: metrics per subject, then mean and sample std across subjects.
Five folds: pooled test points per fold, then mean and sample std across folds.
Uses original dataset preprocessing and original model architectures.
"""

import os
import sys
import time
import json
import logging
import argparse
import importlib

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ==============================================================================
# 1. Environment & Path Setup
# ==============================================================================

BASE_DIR = "/mnt/newdisk/dateset_0909/code/model_train_terrain"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "train", "checkpoints")
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation_results_by_scene")

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for p in [BASE_DIR, DATASETS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dataset_terrain import TrigaitTerrainDataset
except ImportError:
    TrigaitTerrainDataset = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TerrainFinal")

# ==============================================================================
# 2. Domain Topology & Registry
# ==============================================================================

CHANNEL_NAMES = [
    "ipsi_x", "ipsi_y", "ipsi_z",
    "contra_x", "contra_y", "contra_z"
]

CHANNEL_INFO = {
    "ipsi_x":   ("Ipsilateral", "X"),
    "ipsi_y":   ("Ipsilateral", "Y"),
    "ipsi_z":   ("Ipsilateral", "Z"),
    "contra_x": ("Contralateral", "X"),
    "contra_y": ("Contralateral", "Y"),
    "contra_z": ("Contralateral", "Z"),
}

LIMB_CHANNELS = {
    "Ipsilateral": ["ipsi_x", "ipsi_y", "ipsi_z"],
    "Contralateral": ["contra_x", "contra_y", "contra_z"]
}

METRICS_MEAN = ["MAE", "RMSE", "NRMSE", "R2", "Pearson", "Spearman", "Std_AE"]
ALL_METRICS = ["MAE", "RMSE", "NRMSE", "R2", "Pearson", "Spearman", "MaxError", "Std_AE"]

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
# 3. Model Loader & Deterministic Scaler
# ==============================================================================

def build_model_instance(model_name, device, seq_len=200, pred_len=30, input_dim=42):
    model_key = model_name.upper()
    found_key = None
    for k in MODEL_REGISTRY:
        if k.upper() == model_key:
            found_key = k
            break

    if found_key is None:
        raise ValueError(f"Model architecture '{model_name}' not recognized in MODEL_REGISTRY.")

    module_name, class_name = MODEL_REGISTRY[found_key]
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)

    if found_key == "TCN":
        model = cls(input_dim=input_dim, hidden_channels=[64, 128, 128], output_len=pred_len, dropout=0.2)
    elif found_key == "LSTM":
        model = cls(input_dim=input_dim, hidden_dim=128, num_layers=2, pred_len=pred_len, dropout=0.2)
    elif found_key == "Transformer":
        model = cls(input_dim=input_dim, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, output_len=pred_len, dropout=0.2)
    elif found_key == "GPT4TS":
        model = cls(input_dim=input_dim, pred_len=pred_len, freeze=True)
    elif found_key == "TimesNet":
        model = cls(input_len=seq_len, pred_len=pred_len, input_dim=input_dim, d_model=64, d_ff=128, num_layers=2, dropout=0.1)
    elif found_key == "PatchTST":
        model = cls(input_len=seq_len, pred_len=pred_len, input_dim=input_dim, patch_len=16, stride=8)
    elif found_key == "TimeMixer":
        model = cls(input_dim=input_dim, pred_len=pred_len, d_model=64)
    elif found_key == "Mamba":
        model = cls(input_dim=input_dim, d_model=64, num_layers=2, pred_len=pred_len)
    else:
        model = cls()

    return model.to(device)


def load_checkpoint_strictly(model, checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing weights file at: {checkpoint_path}")

    logger.info(f"Strictly loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    clean_state = {}
    for k, v in state.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        clean_state[name] = v

    model.load_state_dict(clean_state, strict=True)
    model.eval()
    return model


class PhysicalScalerEngine:
    def __init__(self, dataset_handle, target_normalized=False):
        self.ds = dataset_handle
        self.target_normalized = target_normalized

    def denormalize(self, data_array):
        if not self.target_normalized:
            return data_array

        if self.ds is None:
            raise RuntimeError("target_normalized=True, but dataset handle is None.")

        if hasattr(self.ds, "denormalize_terrain"):
            return self.ds.denormalize_terrain(data_array)

        if hasattr(self.ds, "terrain_mean") and hasattr(self.ds, "terrain_std"):
            mean = np.asarray(self.ds.terrain_mean).reshape(1, 1, -1)
            std = np.asarray(self.ds.terrain_std).reshape(1, 1, -1)
            return data_array * (std + 1e-6) + mean

        raise RuntimeError(
            "target_normalized=True specified, but neither 'denormalize_terrain' "
            "nor 'terrain_mean'/'terrain_std' found in dataset."
        )


# ==============================================================================
# 4. Metric Computation Core
# ==============================================================================

def compute_1d_metrics(y_true_1d, y_pred_1d):
    """
    Computes regression statistics on 1D pooled valid arrays.
    """
    t = np.asarray(y_true_1d, dtype=np.float64)
    p = np.asarray(y_pred_1d, dtype=np.float64)

    if len(t) == 0:
        return {
            "MAE": np.nan, "RMSE": np.nan, "NRMSE": np.nan, "MaxError": np.nan,
            "Std_AE": np.nan, "R2": np.nan, "Pearson": np.nan, "Spearman": np.nan, "Count": 0
        }

    diff = t - p
    abs_err = np.abs(diff)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(abs_err))
    std_ae = float(np.std(abs_err))

    std_y = float(np.std(t))
    nrmse = float(rmse / std_y) if std_y > 1e-6 else np.nan

    try:
        r2 = float(r2_score(t, p)) if len(t) >= 2 and std_y > 1e-7 else np.nan
    except Exception:
        r2 = np.nan

    if std_y < 1e-7 or np.std(p) < 1e-7 or len(t) < 2:
        pearson_val, spearman_val = np.nan, np.nan
    else:
        try:
            pearson_val = float(pearsonr(t, p)[0])
        except Exception:
            pearson_val = np.nan
        try:
            spearman_val = float(spearmanr(t, p)[0])
        except Exception:
            spearman_val = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "MaxError": max_err,
        "Std_AE": std_ae,
        "R2": r2,
        "Pearson": pearson_val,
        "Spearman": spearman_val,
        "Count": int(len(t))
    }



SCENES = ['walk', 'stair', 'x', 'y', 'z1', 'z2', 'z3', 'z4',
          'xy1', 'xy2', 'xy3', 'xy4', 'xy5', 'xy6', 'xyz1', 'xyz2', 'xyz3']


def parse_task(task):
    parts = str(task).lower().split('_')
    if len(parts) < 4 or parts[2] not in SCENES or parts[-1] not in ('fwd', 'bwd'):
        raise ValueError('Cannot parse scene/direction from task: ' + str(task))
    return parts[2], parts[-1]


def metric_rows(true, pred, mask, metadata):
    rows = []
    for c, channel in enumerate(CHANNEL_NAMES):
        all_true = true[:, :, c]
        valid = mask[:, :, c] > 0
        t = all_true[valid]
        p = pred[:, :, c][valid]
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)):
            raise ValueError('Non-finite valid values: %s / %s' % (metadata, channel))
        metrics = compute_1d_metrics(t, p)
        count = metrics.pop('Count')
        limb, axis = CHANNEL_INFO[channel]

        t64 = np.asarray(t, dtype=np.float64)
        p64 = np.asarray(p, dtype=np.float64)
        true_std = float(np.std(t64)) if count else np.nan
        pred_std = float(np.std(p64)) if count else np.nan
        values, counts = np.unique(t64, return_counts=True)
        value_counts = (
            '; '.join(f'{value:.10g}: {int(n)}' for value, n in zip(values, counts))
            if len(values) <= 20 else f'{len(values)} distinct values'
        )
        reasons = []
        if count == 0:
            reasons.append('No valid samples')
        else:
            if count < 2:
                reasons.append('Fewer than 2 valid samples: R2/correlations undefined')
            if true_std <= 1e-6:
                reasons.append('Target std <= 1e-6: NRMSE undefined')
            if true_std <= 1e-7:
                reasons.append('Target std <= 1e-7: R2 undefined')
            if true_std < 1e-7 or pred_std < 1e-7:
                reasons.append('Target or prediction std < 1e-7: correlations undefined')
            undefined = [k for k, v in metrics.items() if not np.isfinite(v)]
            if undefined and not reasons:
                reasons.append('Undefined metrics: ' + ', '.join(undefined))

        diagnostics = dict(
            True_Min=float(t64.min()) if count else np.nan,
            True_Max=float(t64.max()) if count else np.nan,
            True_Std=true_std,
            Pred_Std=pred_std,
            True_Unique_Count=int(len(values)),
            True_Value_Counts=value_counts,
            Valid_Zero_Points=int(np.count_nonzero(valid & (all_true == 0))),
            Masked_Zero_Points=int(np.count_nonzero(~valid & (all_true == 0))),
            Diagnostic='; '.join(reasons) if reasons else 'All metrics finite'
        )
        rows.append(dict(metadata, Channel=channel, Limb=limb, Axis=axis,
                         Valid_Points=count, Window_Count=len(true),
                         **diagnostics, **metrics))
        if metadata.get('Scene') == 'xyz2' and channel == 'contra_y':
            logger.info('XYZ2 contra_y | %s | valid=%d | true_std=%s | values=%s | %s',
                        metadata, count, true_std, value_counts, diagnostics['Diagnostic'])
    return rows


def infer_partition(model, ds, args, device, fold):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # Group by subject and scene, so all direction records can be pooled correctly.
    buckets = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            for key in ('subject', 'task', 'terrain', 'terrain_mask'):
                if key not in batch:
                    raise KeyError('Dataset must return ' + key)
            pred = model(batch['imu'].to(device))['terrain'].detach().cpu().numpy()
            true = batch['terrain'].numpy()
            mask = batch['terrain_mask'].numpy()
            if pred.shape != true.shape or mask.shape != true.shape or true.shape[-1] != 6:
                raise ValueError('Prediction/target/mask shape mismatch')
            for i, (subject, task) in enumerate(zip(batch['subject'], batch['task'])):
                scene, direction = parse_task(task)
                if args.scenes and scene not in args.scenes:
                    continue
                key = (str(subject), scene)
                bucket = buckets.setdefault(key, {'true': [], 'pred': [], 'mask': [], 'direction': []})
                bucket['true'].append(true[i])
                bucket['pred'].append(pred[i])
                bucket['mask'].append(mask[i])
                bucket['direction'].append(direction)
    if not buckets:
        raise ValueError('No test windows match the requested scenes')
    subject_rows, direction_rows = [], []
    fold_buckets = {}
    for (subject, scene), b in sorted(buckets.items()):
        t, p, m = (np.stack(b[k]) for k in ('true', 'pred', 'mask'))
        meta = dict(Model=args.model, Setting=args.name, Fold=fold, Subject=subject, Scene=scene)
        subject_rows.extend(metric_rows(t, p, m, dict(meta, Direction='ALL')))
        dirs = np.asarray(b['direction'])
        for direction in ('fwd', 'bwd'):
            idx = dirs == direction
            if idx.any():
                direction_rows.extend(metric_rows(t[idx], p[idx], m[idx], dict(meta, Direction=direction)))
        if fold:
            fb = fold_buckets.setdefault(scene, {'true': [], 'pred': [], 'mask': []})
            for name, values in [('true', t), ('pred', p), ('mask', m)]:
                fb[name].append(values)
    fold_rows = []
    for scene, b in fold_buckets.items():
        t, p, m = (np.concatenate(b[k]) for k in ('true', 'pred', 'mask'))
        meta = dict(Model=args.model, Setting=args.name, Fold=fold, Scene=scene, Direction='ALL')
        fold_rows.extend(metric_rows(t, p, m, meta))
    logger.info('Evaluated %d subjects; scenes: %s', len({k[0] for k in buckets}),
                ', '.join(s for s in SCENES if any(k[1] == s for k in buckets)))
    return subject_rows, direction_rows, fold_rows


def summarize(detail, entity):
    rows = []
    for (scene, channel), group in detail.groupby(['Scene', 'Channel'], sort=False):
        limb, axis = CHANNEL_INFO[channel]
        row = dict(Model=group.iloc[0]['Model'], Setting=group.iloc[0]['Setting'],
                   Scene=scene, Channel=channel, Limb=limb, Axis=axis,
                   Aggregation=entity, Entity_Count=len(group),
                   Valid_Entity_Count=int((group.Valid_Points > 0).sum()),
                   Valid_Points=int(group.Valid_Points.sum()),
                   Window_Count=int(group.Window_Count.sum()))
        for metric in ALL_METRICS:
            vals = group[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[metric + '_mean'] = float(vals.mean()) if len(vals) else np.nan
            row[metric + '_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[metric + '_count'] = len(vals)
        rows.append(row)
    rows.sort(key=lambda r: (SCENES.index(r['Scene']), CHANNEL_NAMES.index(r['Channel'])))
    return pd.DataFrame(rows)


def display_table(summary):
    rows = []
    for scene in SCENES:
        group = summary[summary.Scene == scene]
        if group.empty:
            continue
        by_channel = {r['Channel']: r for r in group.to_dict('records')}
        for metric in ALL_METRICS:
            row = dict(Scene=scene, Metric=metric,
                       Unit='cm' if metric in ('MAE', 'RMSE', 'MaxError', 'Std_AE') else 'dimensionless')
            for channel in CHANNEL_NAMES:
                r = by_channel[channel]
                mean, std = r[metric + '_mean'], r[metric + '_std']
                row[channel] = ('—' if not np.isfinite(mean) else
                                ('%.3f ± %.3f' % (mean, std) if np.isfinite(std) else '%.3f' % mean))
            rows.append(row)
    return pd.DataFrame(rows)


def write_result(summary, subjects, directions, folds, run_info, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    tables = {'Scene_Display': display_table(summary), 'Scene_Axis': summary,
              'Subject_Detail': subjects, 'Direction_Detail': directions}
    if not folds.empty:
        tables['Fold_Detail'] = folds
    tables['Run_Info'] = pd.DataFrame(run_info, columns=['Key', 'Value'])
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet, frame in tables.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = 'A2'
            ws.auto_filter.ref = ws.dimensions
            for cells in ws.columns:
                width = min(65, max(14, max(len(str(c.value or '')) for c in cells) + 2))
                ws.column_dimensions[cells[0].column_letter].width = width
    logger.info('Saved: %s', output)
    print(tables['Scene_Display'].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description='Terrain evaluation by scene, limb and axis; all eight metrics.')
    parser.add_argument('--model', required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument('--name', default=None, help='Single-split checkpoint folder; required with --test.')
    parser.add_argument('--test', default=None, help='Test pkl; omit to evaluate all five test folds.')
    parser.add_argument('--output_name', default=None)
    parser.add_argument('--output_root', default=EVALUATION_DIR)
    parser.add_argument('--checkpoint_root', default=CHECKPOINT_DIR)
    parser.add_argument('--split_root', default=os.path.join(BASE_DIR, 'splits'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default=None)
    parser.add_argument('--scenes', nargs='+', choices=SCENES, default=None,
                        help='Optional test scene subset; default is all scenes present.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.test and not args.name:
        parser.error('--name is required with --test')
    if args.batch_size <= 0:
        parser.error('--batch_size must be positive')
    if not args.test:
        args.name = args.name or 'Subject_5Fold_Summary'
    return args


def main():
    args = parse_args()
    if TrigaitTerrainDataset is None:
        raise ImportError('Cannot import dataset_terrain; check BASE_DIR and datasets directory.')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    output_name = args.output_name or args.name
    output = os.path.join(args.output_root, args.model, output_name, 'evaluation.xlsx')
    if os.path.exists(output) and not args.overwrite:
        raise FileExistsError('Output already exists; use a new --output_name or explicit --overwrite: ' + output)
    jobs = []
    if args.test:
        jobs.append((0, args.test, os.path.join(args.checkpoint_root, args.model, args.name, 'best_model.pth')))
    else:
        for fold in range(1, 6):
            jobs.append((fold,
                         os.path.join(args.split_root, 'subject_5fold_split', 'fold%d' % fold, 'test.pkl'),
                         os.path.join(args.checkpoint_root, args.model,
                                      '%s_subject_5fold_fold%d' % (args.model, fold), 'best_model.pth')))
    for _, test, checkpoint in jobs:
        for path in (test, checkpoint):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
    info = [('Model', args.model), ('Setting', args.name), ('Device', str(device)),
            ('Timestamp', time.strftime('%Y-%m-%d %H:%M:%S')),
            ('Aggregation', 'fold mean and sample std' if not args.test else 'subject mean and sample std'),
            ('Scene filter', ', '.join(args.scenes) if args.scenes else 'all test scenes'),
            ('Targets', 'Raw terrain targets; no normalization. Units assumed cm as in source data.'),
            ('Input processing', 'Original dataset: nan_to_num, then IMU / 100'),
            ('Window / horizon / stride', '200 / 30 / 10; original dataset indexing retained'),
            ('Valid points', 'mask > 0; overlapping forecast occurrences are counted, not unique frames'),
            ('NRMSE', 'RMSE / population standard deviation of true values; undefined if std <= 1e-6'),
            ('R2', 'Undefined if fewer than 2 points or true std <= 1e-7'),
            ('Correlations', 'Undefined for constant targets/predictions or fewer than 2 points'),
            ('Missing values', 'Undefined metrics excluded individually; metric_count reports valid entities'),
            ('Diagnostics', 'Detail sheets include valid-target min/max/std, unique values and counts, prediction std, zero-point counts and undefined-metric reasons; counts include overlapping forecast occurrences.'),
            ('Direction', 'Main result pools FWD/BWD points before metrics; Direction_Detail keeps them separate'),
            ('MaxError', 'Maximum within each entity, then entity mean and sample std'),
            ('Std_AE', 'Population std of absolute errors within each entity'),
            ('Cross-axis / cross-scene average', 'None')]
    subject_rows, direction_rows, fold_rows = [], [], []
    seen_subjects = set()
    for fold, test, checkpoint in jobs:
        logger.info('Fold=%s | checkpoint=%s | test=%s', fold, checkpoint, test)
        ds = TrigaitTerrainDataset(test, window_size=200, pred_len=30, stride=10)
        if not len(ds):
            raise ValueError('Empty dataset: ' + test)
        if fold:
            current = {str(sample[0]) for sample in ds.samples}
            if seen_subjects & current:
                raise ValueError('Test subjects overlap across folds: ' + str(seen_subjects & current))
            seen_subjects.update(current)
        model = load_checkpoint_strictly(build_model_instance(args.model, device), checkpoint, device)
        sr, dr, fr = infer_partition(model, ds, args, device, fold)
        subject_rows.extend(sr)
        direction_rows.extend(dr)
        fold_rows.extend(fr)
        info.extend([('Fold%d test' % fold, test), ('Fold%d checkpoint' % fold, checkpoint),
                     ('Fold%d dataset windows' % fold, len(ds))])
        del model, ds
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    subjects = pd.DataFrame(subject_rows)
    directions = pd.DataFrame(direction_rows)
    folds = pd.DataFrame(fold_rows)
    if not args.test:
        for (scene, channel), g in folds.groupby(['Scene', 'Channel']):
            if set(g.Fold) != set(range(1, 6)):
                raise ValueError('Missing fold for %s / %s' % (scene, channel))
    summary = summarize(subjects if args.test else folds, 'Subject' if args.test else 'Fold')
    write_result(summary, subjects, directions, folds, info, output)


if __name__ == '__main__':
    main()

