# TerraCon Dataset Guide

## Dataset access

- **Hugging Face:** [https://huggingface.co/datasets/NJUPTATL/TerraCon](https://huggingface.co/datasets/NJUPTATL/TerraCon)
- **Dataset license:** pending maintainer confirmation
- **Paper citation:** pending publication

This GitHub repository does not contain the large raw or processed dataset files. Download them from the Hugging Face dataset repository, subject to its visibility and access settings.

## Top-level organization

| Path | Purpose |
|---|---|
| `raw data/` | Placeholder directory. Raw data files are not included in this GitHub package. |
| `processed data/information/` | Intended location for dataset-level information files, including equipment and subject metadata approved for release. |
| `processed data/sub_XXX/data/` | Intended location for processed data for each participant. |
| `processed data/sub_XXX/labels/` | Intended location for labels associated with processed participant data. |
| `code/preprocessing/` | Preprocessing utilities. |
| `code/labeling/label_gait/` | Gait-labeling code. |
| `code/labeling/label_terrain/` | Terrain-labeling code. |
| `code/model_train_angle_gait/` | Gait-angle modeling and evaluation. |
| `code/model_train_terrain/` | Terrain modeling and evaluation. |

The project tree previously provided by the maintainers showed participant directories `sub_001` through `sub_048`. This describes the observed directory naming, not a guarantee that every participant has complete recordings for every task.

## Seven-IMU naming convention

The preprocessing notes identify the following mapping:

| Sensor key | Body location label |
|---|---|
| `G` | `pelvis` |
| `L1` | `L_thigh` |
| `L2` | `L_shank` |
| `L3` | `L_foot` |
| `R1` | `R_thigh` |
| `R2` | `R_shank` |
| `R3` | `R_foot` |

The final release should include the sensor-placement diagram and confirm device model, coordinate frames, axes, orientation conventions, units, and sampling rate.

## High-level data workflow

1. `code/preprocessing/convert_device_export.py` — device-export conversion.
2. `code/preprocessing/merge_7imu.py` — merging/alignment of seven IMU streams.
3. `code/preprocessing/split_by_scene.py` — extracting time windows for scenes.
4. `code/preprocessing/build_processed.py` — constructing processed outputs.
5. `code/labeling/` and `processed data/generate_terrain_cls_all_subject.py` — labeling-related workflows.

This is a high-level description of the intended workflow. Confirm exact ordering, inputs, outputs, and prerequisites against the released source code before reproducing the pipeline.

## Data dictionary status

**A complete field-level data dictionary remains to be added.** For every released file type, document:

- file type and worksheet names;
- column names, data types, meanings, and units;
- timestamp unit, origin, sampling rate, and synchronization behavior;
- coordinate frame and axis convention for vector measurements;
- missing-value and invalid-sample conventions;
- label names, allowed values, and definitions;
- how data files are matched to label files.

Do not infer units or semantics from column names alone.

## Metadata and privacy

The original project tree includes `subject_information.xlsx`, `equipment_information.xlsx`, and a `scence_pics/` directory. Only include files that have been reviewed and approved for public release. Check for names, contact details, direct identifiers, or sensitive information before publishing.

## Release checklist

- [ ] Confirm the Hugging Face dataset is publicly downloadable without unintended access barriers.
- [ ] Select and publish the dataset license.
- [ ] Add the paper citation when available.
- [ ] Verify the sensor diagram, coordinate frames, units, and sampling rate.
- [ ] Document gait and terrain label definitions and class mappings.
- [ ] Verify data/label alignment and missing-data conventions.
- [ ] Audit metadata and files for privacy concerns.
- [ ] Version the dataset and document known limitations.
