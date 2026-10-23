# TerraCon Dataset Guide

This guide summarizes the current high-level organization of TerraCon and how to access the data. It is not a substitute for the dataset paper or a complete field-level data dictionary. Details that have not been verified are explicitly marked as pending.

## Dataset access and release status

- **Dataset repository:** <https://huggingface.co/datasets/AICON3/TerraCon>
- **Code repository:** <https://github.com/AICON3/TerraCon>
- **Dataset license:** pending maintainer confirmation
- **Paper citation:** pending publication

The Hugging Face repository is the data distribution location. The GitHub repository focuses on code and documentation and does not contain the complete raw or processed dataset or generated `.pkl` files. Check the Hugging Face page for current access instructions and terms. Repository visibility alone does not establish reuse or redistribution rights.

## Dataset organization

The working project tree provided by the maintainers includes the following areas. The exact files present in a published dataset snapshot may differ from the working directory.

| Path | Purpose |
|---|---|
| `raw data/` | Original device exports and raw recordings, where included in the dataset release. |
| `processed data/information/` | Dataset-level information files, including `equipment_information.xlsx`, `subject_information.xlsx`, and the directory currently spelled `scence_pics/`. |
| `processed data/sub_XXX/data/` | Processed data associated with a participant. |
| `processed data/sub_XXX/labels/` | Labels associated with a participant's processed data. |
| `code/preprocessing/` | Utilities for device-export conversion, IMU stream merging/alignment, scene segmentation, and processed-data construction. |
| `code/labeling/label_gait/` | Gait-labeling code. |
| `code/labeling/label_terrain/` | Terrain-labeling code. |
| `code/model_train_angle_gait/` | Gait-angle modeling, data loading, split logic, training, and evaluation. |
| `code/model_train_terrain/` | Terrain modeling, data loading, split logic, training, and evaluation. |

The working processed-data tree previously shown by the maintainers contains participant directories named `sub_001` through `sub_048`. This is a directory observation only; it does not imply that every participant has complete or usable recordings for every task. Final participant counts and sample statistics should be reported only after the dataset snapshot is checked and frozen.

## Seven-IMU naming used by preprocessing code

The preprocessing code uses the following sensor-key mapping:

| Sensor key | Body-location label in code |
|---|---|
| `G` | `pelvis` |
| `L1` | `L_thigh` |
| `L2` | `L_shank` |
| `L3` | `L_foot` |
| `R1` | `R_thigh` |
| `R2` | `R_shank` |
| `R3` | `R_foot` |

These are the labels used by the current preprocessing source. Device model, sensor placement, coordinate frames, axis conventions, orientation conventions, units, and any changes in the final hardware setup should be confirmed and documented in the dataset paper or a dedicated sensor-placement figure.

## High-level processing stages

The available scripts indicate this general workflow:

1. **Device-export conversion:** `code/preprocessing/convert_device_export.py` converts WIT binary exports into Excel workbooks.
2. **Seven-stream merging/alignment:** `code/preprocessing/merge_7imu.py` combines the seven sensor streams. The supplied source summary describes alignment to a 100 Hz time axis; exact matching, interpolation, and missing-data behavior must be verified against the released code.
3. **Scene segmentation:** `code/preprocessing/split_by_scene.py` extracts time windows associated with selected scenes from stitched sensor workbooks.
4. **Processed-data construction:** `code/preprocessing/build_processed.py` builds standardized processed outputs.
5. **Label generation:** scripts under `code/labeling/` create gait- and terrain-related labels.

For further detail and validation cautions, see [Data Processing Guide](DATA_PROCESSING.md).

## Data schema and interpretation

A definitive field-level data dictionary is still needed. Before relying on the dataset for scientific comparisons, confirm at least:

- column names, units, and data types;
- timestamp units, reference, and sampling conventions;
- sensor-axis and orientation definitions;
- treatment of missing, duplicate, and out-of-order samples;
- gait and terrain label definitions and allowed values;
- label time resolution and scene-boundary handling;
- correspondence between data rows and label rows;
- participant/session/scene identifiers and their interpretation.

Do not infer unspecified units, class names, or preprocessing rules from filenames alone.

## Reproducibility and split files

The GitHub source repository includes some CSV split-list files under `code/model_train_terrain/splits/`. These are split metadata, not the complete training data. Their meaning, referenced sample identifiers, and compatibility with a particular dataset version should be checked before use. Generated `.pkl` split files and training datasets are excluded from the GitHub source release.

For reproducible comparisons, record the dataset version or snapshot, split protocol, preprocessing version, configuration, random seed, dependency versions, and checkpoint-selection/evaluation procedure.

## Items to confirm before the final dataset release

- [ ] Confirm dataset license, reuse terms, and redistribution permissions.
- [ ] Verify public or authenticated download behavior on Hugging Face.
- [ ] Add the paper citation when available.
- [ ] Confirm participant/session/task statistics from the frozen release snapshot.
- [ ] Publish sensor-placement and coordinate-frame definitions.
- [ ] Complete the field-level data dictionary, including units and timestamp conventions.
- [ ] Document gait/terrain label definitions and boundary rules.
- [ ] Check CSV split lists for unintended personal information or machine-specific paths.
- [ ] Audit data and metadata for information that should not be public.
- [ ] Version the dataset snapshot and document known limitations.
