# TerraCon

TerraCon is a dataset-and-code project for studying gait-related signals and terrain recognition from wearable-sensor recordings. This repository is intended to provide project documentation, preprocessing and labeling code, and code for gait-angle and terrain-model experiments.

> **Release note:** This GitHub package contains the repository structure and documentation. The large raw and processed dataset files are intentionally not included. Download the dataset from the Hugging Face repository linked below.

## Dataset access

**Hugging Face dataset:** [https://huggingface.co/datasets/NJUPTATL/TerraCon](https://huggingface.co/datasets/NJUPTATL/TerraCon)

The Hugging Face repository may require access permission until the maintainers finish configuring public visibility. Dataset licensing is currently **not yet specified**. Do not assume that public visibility alone grants permission to use or redistribute the data.

## Repository structure

```text
TerraCon/
├── README.md
├── docs/
│   ├── DATASET.md
│   └── DATA_PROCESSING.md
├── code/
│   ├── labeling/
│   │   ├── label_gait/
│   │   └── label_terrain/
│   ├── model_train_angle_gait/
│   │   ├── configs/
│   │   ├── data_pkl/
│   │   ├── datasets/
│   │   ├── evaluation/
│   │   ├── evaluation_results/
│   │   ├── logs/
│   │   ├── models/
│   │   ├── splits/
│   │   └── train/
│   ├── model_train_terrain/
│   │   ├── configs/
│   │   ├── data_pkl/
│   │   ├── datasets/
│   │   ├── evaluation/
│   │   ├── evaluation_results/
│   │   ├── evaluation_results2/
│   │   ├── logs/
│   │   ├── models/
│   │   ├── splits/
│   │   └── train/
│   └── preprocessing/
│       ├── build_processed.py
│       ├── convert_device_export.py
│       ├── merge_7imu.py
│       └── split_by_scene.py
├── processed data/
│   ├── information/
│   └── sub_001/ ... sub_048/
│       ├── data/
│       └── labels/
└── raw data/
```

The folder structure is provided as a scaffold. **Source code files are not populated in this package** because the actual source files were not supplied in this conversation. Copy your existing project code into the corresponding `code/` folders before publishing the GitHub repository.

## Documentation

- [Dataset guide](docs/DATASET.md)
- [Data processing guide](docs/DATA_PROCESSING.md)

## Preprocessing and experiments

The planned preprocessing workflow includes device-export conversion, seven-IMU merging/alignment, scene-based splitting, and processed-data construction. Gait labeling and terrain labeling are organized separately. Training code is separated into gait-angle and terrain-model experiments.

Before treating commands as reproducible, check the scripts' actual command-line arguments, dependency versions, configuration files, and machine-specific paths. The documentation intentionally marks unverified details rather than inventing execution commands or field definitions.

## Citation

**Paper citation: pending.** Add the final paper citation and DOI when available. Do not use a placeholder citation as if it were a published reference.

## License

**License: pending maintainer confirmation.** Before public release, confirm data ownership, participant consent, institutional requirements, and permitted redistribution. Code and dataset may need separate licenses. Add a root `LICENSE` file for code only after the maintainers choose the appropriate terms, and state the dataset license separately on the Hugging Face page.

## Privacy and responsible use

Before publication, audit the raw data, processed files, and metadata spreadsheets for direct identifiers, sensitive information, credentials, and content not approved for public release. Follow the final dataset license, participant-consent terms, and applicable institutional requirements.

## Contact

Maintainer contact: **to be added**.
