# TerraCon Dataset Guide

This guide describes the high-level organization of TerraCon and how to access the dataset.

## Dataset access

- **Dataset repository:** <https://huggingface.co/datasets/AICON3/TerraCon>
- **Code repository:** <https://github.com/AICON3/TerraCon>

The Hugging Face repository is the dataset distribution location. The GitHub repository focuses on source code and documentation and does not contain the complete raw or processed dataset or generated `.pkl` files.

## Dataset organization

The project tree contains the following major areas. The exact files in a particular dataset snapshot may differ from the working directory.

| Path | Purpose |
|---|---|
| `raw data/` | Original device exports and raw recordings included in the dataset release. |
| `processed data/information/` | Dataset-level information workbooks, including `equipment_information.xlsx` and `subject_information.xlsx`. |
| `processed data/sub_XXX/data/` | Processed data associated with a participant. |
| `processed data/sub_XXX/labels/` | Labels associated with a participant's processed data. |
| `code/preprocessing/` | Device-export conversion, IMU stream merging/alignment, scene segmentation, and processed-data construction. |
| `code/labeling/label_gait/` | Gait-labeling code. |
| `code/labeling/label_terrain/` | Terrain-labeling code. |
| `code/model_train_angle_gait/` | Gait-angle modeling, data loading, split logic, training, and evaluation. |
| `code/model_train_terrain/` | Terrain modeling, data loading, split logic, training, and evaluation. |

The working processed-data tree contains participant directories named `sub_001` through `sub_048`. This describes the observed directory layout; it does not imply that every participant has complete recordings for every task.

The GitHub repository includes the equipment and participant information workbooks. Scene images have been excluded from this public repository to reduce the risk of exposing identifying or sensitive information. The full raw recordings and processed participant-level data are distributed separately.

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

These are the labels used by the current preprocessing source. Sensor placement, coordinate frames, axes, orientation conventions, units, and device details should be described consistently with the dataset's sensor-placement documentation.

## High-level processing stages

1. **Device-export conversion:** `code/preprocessing/convert_device_export.py` converts WIT binary exports into Excel workbooks.
2. **Seven-stream merging/alignment:** `code/preprocessing/merge_7imu.py` combines the seven sensor streams. The supplied source describes alignment to a 100 Hz time axis.
3. **Scene segmentation:** `code/preprocessing/split_by_scene.py` extracts time windows associated with selected scenes from stitched sensor workbooks.
4. **Processed-data construction:** `code/preprocessing/build_processed.py` builds processed outputs.
5. **Label generation:** scripts under `code/labeling/` create gait- and terrain-related labels.

For further detail, see the [Data Processing Guide](DATA_PROCESSING.md).

## Data schema and interpretation

Check field definitions and processing conventions relevant to the task, including column names, units, timestamp conventions, sensor axes/orientation, missing and duplicate samples, gait/terrain label definitions, label time resolution, row alignment, and participant/session/scene identifiers. Do not infer unspecified units, class names, or preprocessing rules from filenames alone.

## Reproducibility and split files

The GitHub source repository includes some CSV split-list files under `code/model_train_terrain/splits/`. These are split metadata, not the complete training data. Check their sample identifiers and compatibility with the dataset version before using them. Generated `.pkl` split files and training datasets are excluded from the GitHub source release.

For reproducible comparisons, record the dataset snapshot, split protocol, preprocessing version, configuration, random seed, dependency versions, and checkpoint-selection/evaluation procedure.
