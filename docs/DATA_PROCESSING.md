# TerraCon Data Processing Guide

This guide outlines the processing code organization. It is a first draft; it does not claim that the scaffold package contains the actual implementation files or that untested commands will run as-is.

## Code organization

The intended preprocessing scripts are:

| Script | Intended role |
|---|---|
| `code/preprocessing/convert_device_export.py` | Convert device-export files to a more accessible tabular format. |
| `code/preprocessing/merge_7imu.py` | Merge/align seven IMU streams. |
| `code/preprocessing/split_by_scene.py` | Extract time windows for named scenes. |
| `code/preprocessing/build_processed.py` | Build standardized processed outputs. |

Related code is organized under:

- `code/labeling/label_gait/`
- `code/labeling/label_terrain/`
- `code/model_train_angle_gait/`
- `code/model_train_terrain/`

The actual `.py` source files must be copied from the existing TerraCon project into these folders. This GitHub scaffold does not include the implementation source.

## Before running scripts

The supplied project notes indicate that some preprocessing scripts use absolute paths from an earlier local directory. Inspect path constants and command-line arguments before execution; replace machine-specific paths with paths valid for your checkout.

From the repository root, inspect the available options:

```bash
cd /path/to/TerraCon
python code/preprocessing/convert_device_export.py --help
python code/preprocessing/merge_7imu.py --help
python code/preprocessing/split_by_scene.py --help
python code/preprocessing/build_processed.py --help
```

These are suggested inspection commands. If a script does not implement `--help`, inspect its argument parser and path constants instead.

## Intended processing stages

### 1. Convert device exports

`convert_device_export.py` is intended to convert raw device-export files into tabular workbooks. Verify its input file pattern, frame parsing, output worksheets, column schema, and handling of incomplete or unparsed bytes from the actual implementation.

Keep raw data unchanged and retain a backup before processing.

### 2. Merge and align seven IMU streams

`merge_7imu.py` is intended to merge data from the seven sensor keys:

| Key | Body location label |
|---|---|
| `G` | `pelvis` |
| `L1` | `L_thigh` |
| `L2` | `L_shank` |
| `L3` | `L_foot` |
| `R1` | `R_thigh` |
| `R2` | `R_shank` |
| `R3` | `R_foot` |

Verify the reference clock, matching/interpolation policy, duplicate timestamp handling, missing-sample policy, target sampling rate, and quality-control outputs before publishing the workflow as reproducible.

### 3. Split by scene

`split_by_scene.py` is intended to extract windows corresponding to scene annotations from stitched recordings. Confirm the source of scene boundaries, default time margins, required arguments, output paths, and overwrite behavior from the actual source code.

### 4. Build processed outputs

`build_processed.py` is intended to construct processed data and associated outputs. Document its exact input schema, output schema, row filtering/reordering behavior, label matching rules, and validation checks after inspecting and testing the implementation.

### 5. Generate labels

Gait and terrain labeling code is organized separately. For each label, document its definition, valid values, labeling method, temporal resolution, boundary handling, and behavior for ambiguous or missing samples. Verify that data and label rows/timestamps align.

## Training and evaluation

- `code/model_train_angle_gait/` — gait-angle modeling and evaluation workflow.
- `code/model_train_terrain/` — terrain-modeling and evaluation workflow.

For reproducibility, document dependency versions, configuration, dataset version, split protocol, random seeds, checkpoint-selection rules, and exact training/evaluation commands after verifying them from the source.

## Validation checklist

- [ ] Confirm raw input files remain unchanged.
- [ ] Confirm outputs exist and can be read.
- [ ] Preserve participant, sensor, recording, and scene identifiers.
- [ ] Verify timestamp units, ordering, and synchronization.
- [ ] Record sample counts and missing values.
- [ ] Verify data/label pairing and row alignment.
- [ ] Record source file, script version, and processing configuration.

## Known documentation gaps

The following must be completed before claiming full reproducibility:

- tested environment and dependency installation;
- verified commands and required arguments;
- canonical scene names and boundary metadata;
- data and label schemas, units, and valid values;
- synchronization and missing-data policies;
- exact training/evaluation configurations;
- final dataset license and paper citation.
