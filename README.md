# TerraCon

TerraCon is a wearable-sensor dataset and code project for gait-related signal analysis and terrain recognition. The repository provides preprocessing utilities, labeling scripts, and model code for gait-angle and terrain-model experiments. Dataset files and large generated artifacts are distributed separately.

## Dataset and code

- **Dataset:** <https://huggingface.co/datasets/AICON3/TerraCon>
- **Source code:** <https://github.com/AICON3/TerraCon>

The dataset repository is the source for the released data files. This code repository does not contain the complete raw or processed dataset, generated `.pkl` datasets or splits, trained model checkpoints, training logs, or large evaluation artifacts. Some CSV split lists are included in the terrain-modeling workflow.

## Project components

- **Preprocessing:** conversion of device exports, merging and alignment of seven IMU streams, scene-based segmentation, and construction of processed outputs.
- **Labeling:** gait-related and terrain-related labeling scripts.
- **Gait-angle modeling:** dataset loading, model definitions, split logic, training, and evaluation.
- **Terrain modeling:** dataset loading, model definitions, split logic, training, and evaluation.

See the [Dataset Guide](docs/DATASET.md) for dataset organization and the [Data Processing Guide](docs/DATA_PROCESSING.md) for the code workflow.

## Repository structure

```text
TerraCon/
├── README.md
├── docs/
│   ├── DATASET.md
│   └── DATA_PROCESSING.md
├── code/
│   ├── preprocessing/
│   ├── labeling/
│   ├── model_train_angle_gait/
│   └── model_train_terrain/
├── raw data/
└── processed data/
    └── information/
```

The data directories in this repository are scaffolding for the dataset layout; the full recordings and processed participant data are hosted separately.

## Download the dataset

Install the Hugging Face CLI and download the dataset repository:

```bash
python -m pip install -U huggingface_hub
hf download AICON3/TerraCon --repo-type dataset --local-dir ./TerraCon-data
```

If access requires authentication, sign in with the Hugging Face CLI before downloading.

## Run the code

Start from the repository root and use the script that matches your task. Check its arguments and configuration, set input and output paths for your environment, and install the dependencies required by that script. The [Data Processing Guide](docs/DATA_PROCESSING.md) lists the main scripts and processing stages.
