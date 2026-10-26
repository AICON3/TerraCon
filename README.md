# TerraCon

TerraCon is a wearable-sensor dataset and code project for research on gait-related signals and terrain recognition. This GitHub repository provides preprocessing utilities, labeling scripts, and model code for gait-angle and terrain-model experiments. Dataset files and large generated artifacts are hosted separately.

## Dataset access

- **Hugging Face dataset:** <https://huggingface.co/datasets/AICON3/TerraCon>
- **GitHub code repository:** <https://github.com/AICON3/TerraCon>

The Hugging Face repository is the dataset download location. This GitHub repository primarily provides source code and documentation; it does not include the complete raw or processed dataset, generated `.pkl` datasets/splits, or trained model checkpoints. A small number of CSV split lists may be present under the terrain-modeling workflow.

For download instructions and dataset organization, see the [Dataset Guide](docs/DATASET.md).

## Information files

The repository includes `processed data/information/equipment_information.xlsx`, `processed data/information/subject_information.xlsx`, and 16 scene images under `processed data/information/scence_pics/`. The complete raw recordings and processed participant-level data are distributed separately through the [TerraCon dataset repository](https://huggingface.co/datasets/AICON3/TerraCon).

## What is included?

- **Preprocessing:** conversion of device exports, merging/alignment of seven IMU streams, scene-based segmentation, and construction of processed outputs.
- **Labeling:** gait-related and terrain-related labeling scripts.
- **Gait-angle modeling:** dataset loading, model definitions, training entry point, split logic, and evaluation scripts.
- **Terrain modeling:** dataset loading, model definitions, training entry point, split logic, and evaluation scripts.

See the [Data Processing Guide](docs/DATA_PROCESSING.md) for workflow details.

## Repository structure

```text
TerraCon/
├── README.md
├── docs/
│   ├── DATASET.md
│   └── DATA_PROCESSING.md
└── code/
    ├── preprocessing/
    ├── labeling/
    ├── model_train_angle_gait/
    └── model_train_terrain/
```

Generated datasets, trained checkpoints, logs, and large evaluation artifacts are not included in this source-code repository. Download required dataset files from the [TerraCon dataset page](https://huggingface.co/datasets/AICON3/TerraCon).

## Downloading the dataset

```bash
python -m pip install -U huggingface_hub
hf download AICON3/TerraCon --repo-type dataset --local-dir ./TerraCon-data
```

If the repository requires authentication, sign in using the Hugging Face CLI.

## Running the code

Before running a script:

1. Download the required dataset files and identify the expected input/output directories.
2. Inspect the script's argument parser and configuration files.
3. Replace machine-specific absolute paths with paths valid on your system.
4. Install the Python packages required by the selected script.
5. Keep an unchanged backup of raw recordings before preprocessing and test on a small copy first.

Some scripts use defaults from a previous local directory layout. Command-line arguments and configuration options may differ between scripts. See the [Data Processing Guide](docs/DATA_PROCESSING.md).
