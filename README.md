# TerraCon

TerraCon is a wearable-sensor dataset and code project for research on gait-related signals and terrain recognition. This GitHub repository provides preprocessing utilities, labeling scripts, and model code for gait-angle and terrain-model experiments. The dataset files and large generated artifacts are hosted separately.

> **Release notes:** The dataset license and paper citation have not yet been finalized. Some environment requirements, field-level definitions, and end-to-end commands also remain to be verified. Public visibility does not by itself grant permission to reuse or redistribute data. Please follow the license and terms published by the maintainers when available.

## Dataset access

- **Hugging Face dataset:** <https://huggingface.co/datasets/NJUPTATL/TerraCon>
- **GitHub code repository:** <https://github.com/NJUPTATL/TerraCon>

The Hugging Face repository is the dataset download location. This GitHub repository is intended primarily for source code and documentation; it does **not** include the full raw or processed dataset, generated `.pkl` datasets/splits, or trained model checkpoints. A small number of CSV split lists may be present under the terrain-modeling workflow.

For download instructions and a description of the dataset directory organization, see [Dataset Guide](docs/DATASET.md).

## What is included?

- **Preprocessing:** conversion of device exports, merging/alignment of seven IMU streams, scene-based segmentation, and construction of processed outputs.
- **Labeling:** gait-related and terrain-related labeling scripts.
- **Gait-angle modeling:** dataset loading, model definitions, training entry point, split logic, and evaluation scripts.
- **Terrain modeling:** dataset loading, model definitions, training entry point, split logic, and evaluation scripts.

See [Data Processing Guide](docs/DATA_PROCESSING.md) for the workflow overview and current reproducibility notes.

## Repository structure

```text
TerraCon/
├── README.md
├── docs/
│   ├── DATASET.md
│   └── DATA_PROCESSING.md
└── code/
    ├── preprocessing/
    │   ├── build_processed.py
    │   ├── convert_device_export.py
    │   ├── merge_7imu.py
    │   └── split_by_scene.py
    ├── labeling/
    │   ├── label_gait/
    │   │   └── label_gait.py
    │   └── label_terrain/
    │       ├── labeled_terrain_x.py
    │       ├── labeled_terrain_xy.py
    │       ├── labeled_terrain_xyz1.py
    │       ├── labeled_terrain_xyz2.py
    │       ├── labeled_terrain_xyz3.py
    │       ├── labeled_terrain_y.py
    │       ├── labeled_terrain_z.py
    │       └── labeled_terrain_z1.py
    ├── model_train_angle_gait/
    │   ├── configs/
    │   ├── data_pkl/             # conversion script; generated PKL files excluded
    │   ├── datasets/
    │   ├── evaluation/
    │   ├── models/
    │   ├── splits/
    │   └── train/
    └── model_train_terrain/
        ├── configs/
        ├── data_pkl/             # conversion script; generated PKL files excluded
        ├── datasets/
        ├── evaluation/
        ├── models/
        ├── splits/               # selected CSV split lists may be included
        └── train/
```

Generated datasets, trained checkpoints, logs, and large evaluation artifacts are not included as part of this source-code release. Download data from Hugging Face where access and license terms permit.

## Downloading the dataset

Open the [TerraCon dataset page on Hugging Face](https://huggingface.co/datasets/NJUPTATL/TerraCon) and follow the repository's current access and license information. With a compatible version of `huggingface_hub`, the following command can download a dataset repository:

```bash
python -m pip install -U huggingface_hub
hf download NJUPTATL/TerraCon \
  --repo-type dataset \
  --local-dir ./TerraCon-data
```

If the repository requires authentication, sign in using the Hugging Face CLI. Never place access tokens in source code, documentation, shell history, or public issues.

## Running the code

The code has not yet been verified as a portable, end-to-end pipeline in a clean environment. Before running any script:

1. Download the required dataset files and identify the correct input/output directories.
2. Inspect the script's argument parser and configuration files.
3. Replace machine-specific absolute paths with paths valid on your system.
4. Confirm the required Python packages and versions.
5. Keep an unchanged backup of raw recordings before preprocessing.

Some scripts may not implement a `--help` option, and the commands or defaults can vary by script. Do not assume that all modules can be run directly without configuration. See [Data Processing Guide](docs/DATA_PROCESSING.md).

## Citation

**Citation details are pending.** Once the associated paper is published, this section should be updated with its complete citation and persistent identifier (such as a DOI). No publication details are asserted here until confirmed.

## License and terms of use

**Dataset license: pending maintainer confirmation.** The maintainers need to confirm data ownership, participant consent, institutional requirements, and redistribution restrictions before publishing definitive terms. Do not assume that the dataset can be reused or redistributed solely because its repository is visible.

The code and dataset may require different licenses. No license for one should be assumed to apply to the other. Add a code license only after the project maintainers have selected and approved it, and publish the dataset terms on the Hugging Face dataset page.

## Privacy and responsible use

Use the dataset only in accordance with the final license, participant-consent terms, and applicable institutional requirements and laws. Before any release update, review files and metadata for direct identifiers, credentials, local paths, and other information that should not be public.

## Contact

Maintainer contact information will be added by the project team.
