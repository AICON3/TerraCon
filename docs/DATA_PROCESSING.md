# TerraCon Data Processing Guide

This guide summarizes the preprocessing, labeling, and modeling components in the TerraCon source tree and describes practical checks for using them.

## Code map

| Script or directory | Role |
|---|---|
| `code/preprocessing/convert_device_export.py` | Convert WIT device exports into Excel workbooks. |
| `code/preprocessing/merge_7imu.py` | Merge and align seven IMU streams. |
| `code/preprocessing/split_by_scene.py` | Extract time windows for selected scenes from stitched sensor workbooks. |
| `code/preprocessing/build_processed.py` | Build processed-data outputs. |
| `code/labeling/label_gait/label_gait.py` | Gait-related labeling. |
| `code/labeling/label_terrain/` | Terrain-labeling scripts. |
| `code/model_train_angle_gait/` | Gait-angle data loading, models, split logic, training, and evaluation. |
| `code/model_train_terrain/` | Terrain data loading, models, split logic, training, and evaluation. |

The dataset files are hosted separately at <https://huggingface.co/datasets/AICON3/TerraCon>. The GitHub repository does not include the complete dataset, generated `.pkl` files, or trained checkpoint weights.

## Before running scripts

Some source files contain machine-specific default paths from an earlier local directory layout, including paths under `/mnt/newdisk/Trigait/`. Update these paths for your environment before running the scripts.

1. Clone or download the GitHub code repository.
2. Download the dataset files required by the selected workflow.
3. Inspect the script's argument parser, constants, and configuration files.
4. Replace absolute paths with valid local paths.
5. Install the Python packages required by the selected script.
6. Keep an unchanged backup of raw data before preprocessing and test on a small copy first.

From the repository root, inspect script options with commands such as:

```bash
python code/preprocessing/convert_device_export.py --help
python code/preprocessing/merge_7imu.py --help
python code/preprocessing/split_by_scene.py --help
python code/preprocessing/build_processed.py --help
```

If a script does not implement `--help`, inspect its argument parser and source code.

## Stage 1 — Convert device exports

**Script:** `code/preprocessing/convert_device_export.py`

This stage converts WIT binary files for sensor keys `G`, `L1`, `L2`, `L3`, `R1`, `R2`, and `R3` into Excel workbooks. The described workflow includes concatenating numbered files in order, preserving source files, retaining raw hexadecimal data for traceability, and storing unparsed or trailing incomplete bytes in a separate worksheet.

Before processing a full recording set, check expected folder/file naming, workbook sheet names, output columns, and overwrite behavior. Keep an unchanged copy of source exports.

## Stage 2 — Merge and align seven IMU streams

**Script:** `code/preprocessing/merge_7imu.py`

| Key | Body-location label |
|---|---|
| `G` | `pelvis` |
| `L1` | `L_thigh` |
| `L2` | `L_shank` |
| `L3` | `L_foot` |
| `R1` | `R_thigh` |
| `R2` | `R_shank` |
| `R3` | `R_foot` |

The provided source description specifies alignment to a 100 Hz time axis. Inspect the timestamp reference, matching of different sampling times, duplicate timestamps and gaps, missing samples, and any synchronization or quality-control reports.

## Stage 3 — Split recordings by scene

**Script:** `code/preprocessing/split_by_scene.py`

This stage extracts scene windows from stitched sensor workbooks. The supplied script description specifies a default margin of 0.5 seconds before and after the scene window, adjustable with `--before` and `--after`. It describes preserving row order and selecting rows by time rather than re-merging, filtering, interpolating, or re-aligning streams.

Check required arguments, scene-time metadata, output directory, and overwrite behavior before processing a large batch.

## Stage 4 — Build processed outputs

**Script:** `code/preprocessing/build_processed.py`

Check the input/output directory structure, expected and generated columns, data/label pairing, row ordering, timestamp matching, and quality-control reports. Processed samples should remain traceable to their participant, recording or scene, source stream, and associated labels.

## Stage 5 — Generate and inspect labels

Gait-labeling code is under `code/labeling/label_gait/`; terrain-labeling scripts are under `code/labeling/label_terrain/`. The working data tree also includes `processed data/generate_terrain_cls_all_subject.py`, which may be distributed with the data rather than the code-only repository snapshot.

For each label type, identify its definition and allowed values, how it is generated, temporal resolution and scene-boundary handling, treatment of ambiguous/missing/transition samples, and whether timestamps and row counts align with the corresponding data. Do not assume similarly named terrain scripts produce interchangeable labels.

## Training and evaluation

### Gait-angle modeling

`code/model_train_angle_gait/` contains configuration files, dataset-loading code, model definitions, split logic, a training entry point, and evaluation scripts. Generated `.pkl` datasets and splits, trained weights, logs, and large generated artifacts are not included in the GitHub source release.

### Terrain modeling

`code/model_train_terrain/` contains configuration files, dataset-loading code, model definitions, split logic, a training entry point, and evaluation scripts. Some CSV split-list files are included under `splits/`; they do not replace the underlying data or generated split files.

For reproducible experiments, record the dataset snapshot, Python/framework versions, dependencies, configuration, split protocol, random seeds, preprocessing version, checkpoint-selection rule, and evaluation command. Replace machine-specific absolute paths before running on another machine.

## Validation checklist

After each processing stage, verify that:

- [ ] Inputs remain unchanged when the stage is intended to be non-destructive.
- [ ] Expected outputs exist and can be opened.
- [ ] Participant, sensor, recording, and scene identifiers are preserved.
- [ ] Timestamp units, ordering, and sampling conventions are consistent.
- [ ] Data rows and corresponding labels align correctly.
- [ ] Missing values, duplicate samples, and unexpected gaps are inspected.
- [ ] Outputs match the expected schema and directory structure.
- [ ] Configuration, software versions, and processing parameters are recorded.
