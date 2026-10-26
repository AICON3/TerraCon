# TerraCon Dataset Guide

This guide describes the dataset layout, access point, and naming conventions used by the current preprocessing code.

## Dataset and code

- **Dataset:** <https://huggingface.co/datasets/AICON3/TerraCon>
- **Source code:** <https://github.com/AICON3/TerraCon>

The dataset repository hosts the released data files. The GitHub source repository does not contain the complete raw or processed dataset, generated `.pkl` datasets or splits, trained model checkpoints, training logs, or large evaluation artifacts.

## Dataset organization

| Path | Purpose |
|---|---|
| `raw data/` | Original device exports and raw recordings included in the dataset release. |
| `processed data/information/` | Dataset-level reference files and scene images supplied with the dataset release. |
| `processed data/sub_XXX/data/` | Processed data associated with a participant. |
| `processed data/sub_XXX/labels/` | Labels associated with a participant's processed data. |
| `code/preprocessing/` | Device-export conversion, IMU stream merging and alignment, scene segmentation, and processed-data construction. |
| `code/labeling/label_gait/` | Gait-labeling code. |
| `code/labeling/label_terrain/` | Terrain-labeling code. |
| `code/model_train_angle_gait/` | Gait-angle modeling, data loading, split logic, training, and evaluation. |
| `code/model_train_terrain/` | Terrain modeling, data loading, split logic, training, and evaluation. |

The working processed-data tree uses participant directory names from `sub_001` through `sub_048`. Directory names alone do not indicate that every participant has complete recordings for every task.

## Seven-IMU naming used by preprocessing code

| Sensor key | Body-location label in code |
|---|---|
| `G` | `pelvis` |
| `L1` | `L_thigh` |
| `L2` | `L_shank` |
| `L3` | `L_foot` |
| `R1` | `R_thigh` |
| `R2` | `R_shank` |
| `R3` | `R_foot` |

These names describe the mapping implemented in the current preprocessing source. This repository does not provide a separate sensor-placement guide specifying mounting positions, coordinate frames, axes, orientation conventions, units, or device specifications. Those details should only be reported when supported by the dataset documentation or device records.

## High-level processing stages

1. **Device-export conversion:** `code/preprocessing/convert_device_export.py` converts WIT binary exports into Excel workbooks.
2. **Seven-stream merging and alignment:** `code/preprocessing/merge_7imu.py` combines the seven sensor streams. The source describes alignment to a 100 Hz time axis.
3. **Scene segmentation:** `code/preprocessing/split_by_scene.py` extracts time windows associated with selected scenes from stitched sensor workbooks.
4. **Processed-data construction:** `code/preprocessing/build_processed.py` builds processed outputs.
5. **Label generation:** scripts under `code/labeling/` create gait- and terrain-related labels.

See the [Data Processing Guide](DATA_PROCESSING.md) for details.

## Data schema and interpretation

Use the dataset files and corresponding code to identify column names, units, timestamp conventions, sensor axes and orientation, missing or duplicate samples, gait and terrain label definitions, label time resolution, row alignment, and participant/session/scene identifiers. Do not infer unspecified units, class names, or preprocessing rules from filenames alone.

## Reproducibility and split files

The GitHub source repository includes some CSV split-list files under `code/model_train_terrain/splits/`. These are split metadata, not the underlying training data. Check their sample identifiers and compatibility with the dataset version before use. Generated `.pkl` split files and training datasets are not included in the source release.

For reproducible comparisons, record the dataset snapshot, split protocol, preprocessing version, configuration, random seed, dependency versions, and checkpoint-selection and evaluation procedure.
