# TerraCon Data Processing Guide

This guide describes the preprocessing, labeling, and modeling components visible in the current TerraCon source tree. It is a workflow overview, not a guarantee that every script runs end-to-end without configuration. Input/output schemas and commands should be verified against the released source and the intended dataset snapshot.

## Code map

| Script or directory | Role indicated by the current project/source |
|---|---|
| `code/preprocessing/convert_device_export.py` | Convert WIT device exports into Excel workbooks while retaining traceability information. |
| `code/preprocessing/merge_7imu.py` | Merge/align seven IMU streams. |
| `code/preprocessing/split_by_scene.py` | Extract time windows for selected scenes from stitched sensor workbooks. |
| `code/preprocessing/build_processed.py` | Build standardized processed-data outputs. |
| `code/labeling/label_gait/label_gait.py` | Gait-related labeling. |
| `code/labeling/label_terrain/` | A set of terrain-labeling scripts. Do not assume similarly named scripts define interchangeable labels. |
| `code/model_train_angle_gait/` | Gait-angle dataset loading, models, split logic, training, and evaluation. |
| `code/model_train_terrain/` | Terrain dataset loading, models, split logic, training, and evaluation. |

The dataset files are hosted separately at <https://huggingface.co/datasets/AICON3/TerraCon>. The GitHub repository does not include the full dataset, generated `.pkl` files, or trained checkpoint weights. Obtain the required inputs separately and ensure that your access and use comply with the dataset's published terms.

## Before running scripts

Some source files contain machine-specific default paths from an earlier local directory layout, including paths under `/mnt/newdisk/Trigait/`. Do not assume these paths exist on your system.

1. Clone or download the GitHub code repository.
2. Download the required dataset files from Hugging Face, if permitted by the current terms.
3. Inspect the script's argument parser, constants, and configuration files.
4. Replace absolute paths with valid local paths.
5. Check required Python packages and versions. No single tested environment specification has yet been confirmed for this release.
6. Keep an immutable backup of raw data before preprocessing, and test on a small copy first.

From the repository root, you can inspect the scripts with commands such as:

```bash
python code/preprocessing/convert_device_export.py --help
python code/preprocessing/merge_7imu.py --help
python code/preprocessing/split_by_scene.py --help
python code/preprocessing/build_processed.py --help
```

These are inspection attempts only. A script may not implement `--help`; if it does not, inspect its parser and source instead. Do not treat the commands as a verified end-to-end run recipe.

## Stage 1 — Convert device exports

**Script:** `code/preprocessing/convert_device_export.py`

The supplied script documentation describes conversion of WIT binary files for sensor keys `G`, `L1`, `L2`, `L3`, `R1`, `R2`, and `R3` into Excel workbooks. The documented design includes concatenating numbered files in order, leaving source files unchanged, retaining raw hexadecimal data for traceability, and storing unparsed or trailing incomplete bytes in a separate worksheet.

Before using a released version, verify its exact arguments, expected folder/file naming, workbook sheet names, output columns, and overwrite behavior. Preserve an unchanged copy of the source exports.

## Stage 2 — Merge and align seven IMU streams

**Script:** `code/preprocessing/merge_7imu.py`

The current code uses the following sensor-key mapping:

| Key | Body-location label |
|---|---|
| `G` | `pelvis` |
| `L1` | `L_thigh` |
| `L2` | `L_shank` |
| `L3` | `L_foot` |
| `R1` | `R_thigh` |
| `R2` | `R_shank` |
| `R3` | `R_foot` |

The provided source summary describes alignment to a 100 Hz time axis. Before documenting this as a definitive data property, verify the released implementation and record:

- which timestamp is used as the reference;
- how different sampling times are matched;
- how duplicate timestamps, gaps, and missing samples are handled;
- whether values are interpolated, forward-filled, or left missing;
- what synchronization and quality-control reports are generated.

The target sampling rate alone does not establish the interpolation or missing-data policy.

## Stage 3 — Split recordings by scene

**Script:** `code/preprocessing/split_by_scene.py`

The supplied script description indicates that this stage extracts scene windows from already-stitched sensor workbooks. It describes a default margin of 0.5 seconds before and after the scene window, adjustable with `--before` and `--after`. It also describes preserving row order and selecting rows by time rather than re-merging, filtering, interpolating, or re-aligning the streams.

Verify the current script's required arguments, scene-time metadata, output directory, and overwrite behavior before running it. Back up existing outputs if the script may overwrite them.

## Stage 4 — Build processed outputs

**Script:** `code/preprocessing/build_processed.py`

This script belongs to the processed-data construction workflow. Confirm the following from the exact released implementation before treating the output as standardized:

- input and output directory structure;
- expected input columns and generated columns;
- how data rows are paired with labels;
- whether rows are dropped or reordered;
- timestamp matching and validation rules;
- quality-control checks and summary reports.

Processed samples should remain traceable to their participant, recording/scene, source stream, and associated labels.

## Stage 5 — Generate and inspect labels

Gait-labeling code is under `code/labeling/label_gait/`; terrain-labeling scripts are under `code/labeling/label_terrain/`. The working data tree also includes `processed data/generate_terrain_cls_all_subject.py`, which may be distributed with the data rather than the current code-only repository snapshot.

For each label type, document:

- label name and definition;
- allowed values and meanings;
- whether labels are manually assigned, rule-generated, or derived;
- temporal resolution and scene-boundary handling;
- treatment of ambiguous, missing, and transition samples;
- whether timestamps and row counts align with the corresponding data.

Do not assume that similarly named terrain scripts produce interchangeable label definitions. Inspect each script's actual inputs and outputs.

## Training and evaluation

### Gait-angle modeling

`code/model_train_angle_gait/` contains configuration files, dataset-loading code, model definitions, split logic, a training entry point, and evaluation scripts. Generated `.pkl` datasets/splits, trained weights, logs, and large generated artifacts are not part of the GitHub source release.

### Terrain modeling

`code/model_train_terrain/` contains configuration files, dataset-loading code, model definitions, split logic, a training entry point, and evaluation scripts. Some CSV split-list files are included under `splits/`; they do not replace the underlying data or generated split files.

For a reproducible experiment, record the dataset snapshot, Python and framework versions, dependencies, exact configuration, split protocol, random seeds, preprocessing version, checkpoint-selection rule, and evaluation command. Replace machine-specific absolute paths before running on another machine.

## Validation checklist

After each processing stage, verify at least:

- [ ] Inputs are unchanged when a stage is intended to be non-destructive.
- [ ] Expected outputs exist and can be opened.
- [ ] Participant, sensor, recording, and scene identifiers are preserved.
- [ ] Timestamp units and ordering match the documented convention.
- [ ] Sample counts and missing values are reported.
- [ ] Data/label row alignment is checked explicitly.
- [ ] Outputs can be traced to source files and processing configuration.
- [ ] No unintended identifiers, credentials, or local paths are exposed in outputs intended for public release.

## Known limitations and pending documentation

The current documentation does not provide a copy-paste end-to-end command because the supplied source uses machine-specific paths and not all runtime requirements and input/output contracts have been verified. Before claiming full reproducibility, test the workflow from a clean checkout and publish tested commands, dependency versions, expected output paths, and a small example.

Items still to confirm include dependency installation, exact command-line arguments, scene-time metadata, data schemas and units, timestamp and missing-data policies, label definitions/class mappings, and the intended scope of generated artifacts. Dataset license and paper citation are also pending maintainer confirmation.
