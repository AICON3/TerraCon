# TerraCon Data Processing Guide

This guide maps the preprocessing, labeling, and modeling components in the TerraCon source tree.

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

The dataset is hosted separately at <https://huggingface.co/datasets/AICON3/TerraCon>. The GitHub source repository does not include the complete raw or processed dataset, generated `.pkl` datasets or splits, trained model checkpoints, training logs, or large evaluation artifacts. Some CSV split lists are included under the terrain-modeling workflow.

## Environment and paths

Some source files contain machine-specific default paths from an earlier local directory layout, including paths under `/mnt/newdisk/Trigait/`. Set these paths for your environment before running the affected scripts.

From the repository root, inspect the available options with:

```bash
python code/preprocessing/convert_device_export.py --help
python code/preprocessing/merge_7imu.py --help
python code/preprocessing/split_by_scene.py --help
python code/preprocessing/build_processed.py --help
```

If a script does not implement `--help`, check its argument parser and configuration in the source.

## Stage 1 — Convert device exports

**Script:** `code/preprocessing/convert_device_export.py`

This stage converts WIT binary files for sensor keys `G`, `L1`, `L2`, `L3`, `R1`, `R2`, and `R3` into Excel workbooks. The described workflow concatenates numbered files in order, preserves source files, retains raw hexadecimal data for traceability, and stores unparsed or trailing incomplete bytes in a separate worksheet.

Confirm the input naming pattern, workbook sheet names, output columns, and overwrite behavior for the selected script.

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

The source description specifies alignment to a 100 Hz time axis. The resulting data should be interpreted with the script's timestamp matching and missing-sample behavior in mind.

## Stage 3 — Split recordings by scene

**Script:** `code/preprocessing/split_by_scene.py`

This stage extracts scene windows from stitched sensor workbooks. The script description specifies a default margin of 0.5 seconds before and after each scene window, adjustable with `--before` and `--after`. It describes preserving row order and selecting rows by time rather than re-merging, filtering, interpolating, or re-aligning streams.

## Stage 4 — Build processed outputs

**Script:** `code/preprocessing/build_processed.py`

This stage builds processed outputs from the inputs available to the script. Expected input files, generated columns, data/label pairing, row ordering, timestamp matching, and output structure are defined by the implementation and its configuration.

## Stage 5 — Generate and inspect labels

Gait-labeling code is under `code/labeling/label_gait/`; terrain-labeling scripts are under `code/labeling/label_terrain/`. The script `processed data/generate_terrain_cls_all_subject.py` is not present in this GitHub source snapshot, so it is not listed as an executable component here.

For each label type, use the relevant labeling script to establish its definition and allowed values, temporal resolution, scene-boundary handling, treatment of ambiguous or missing samples, and alignment with the corresponding data. Similarly named terrain scripts may implement different label schemes.

## Training and evaluation

### Gait-angle modeling

`code/model_train_angle_gait/` contains configuration files, dataset-loading code, model definitions, split logic, a training entry point, and evaluation scripts. Generated `.pkl` datasets and splits, trained weights, logs, and large generated artifacts are not included in the GitHub source release.

### Terrain modeling

`code/model_train_terrain/` contains configuration files, dataset-loading code, model definitions, split logic, a training entry point, and evaluation scripts. Some CSV split-list files are included under `splits/`; they do not replace the underlying data or generated split files.

For reproducible experiments, record the dataset snapshot, Python and framework versions, dependencies, configuration, split protocol, random seed, preprocessing version, checkpoint-selection rule, and evaluation command.

## Output checks

After a processing stage, confirm that the expected outputs were produced, participant and sensor identifiers remain traceable, timestamps and sampling conventions are consistent, labels align with data rows, and the output schema matches the next stage's input requirements.
