# 3D Medical Image Classification

This repository provides an easy-to-use pipeline for 3D medical image classification with three key features: 
- support multi-label and multi-class classification
- fine-tune foundation models 
- saliency map analysis


# Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and package management.

```shell
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From inside the cloned repo
uv venv --python 3.11
source .venv/bin/activate

uv pip install torch torchvision --torch-backend=auto
# Verifying the install
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
# You should see your torch version, `cuda: True`, and a non-zero device count if you have a GPU available.

uv pip install -e .
```

# Dataset preprocessing

Two preprocessing scripts, one per modality: [`scripts/preprocess_ct.py`](scripts/preprocess_ct.py) for CT and [`scripts/preprocess_mri.py`](scripts/preprocess_mri.py) for MRI. Both take one or more directories of `.nii.gz` images, resample to a target spacing (defaults to the per-axis median across the input dataset), crop to the non-zero bounding box, normalize, and save as Blosc2 (`.b2nd`). Center 160³ patches are extracted at training time, not during preprocessing.

Each script writes its output as one sub-directory per subject:

```
<out-root>/
    preprocessing.json          <- modality, target spacing, (CT) intensity stats, has_masks
    preprocessed_b2nd/
        <image_id>/
            <image_id>.b2nd          <- image
            <image_id>_mask.b2nd     <- mask (only when --mask-dir is given)
```

The `preprocessing.json` sidecar records every knob `predict_external.py` needs to replay the same preprocessing on new NIfTI files at inference time. `cli.py` automatically copies it into each training run's `Configs/` directory so the run is self-describing.

## Optional: masks as a second input channel

Pass `--mask-dir` to either preprocess script to attach a co-registered segmentation/ROI mask to each image. For an image `<id>.nii.gz` the matching mask must be named `<id>_mask.nii.gz` in `--mask-dir`. The mask is resampled with nearest-neighbour, cropped with the image's bounding box, saved raw (no intensity normalization) as `<id>_mask.b2nd`, and recorded via `has_masks: true` in `preprocessing.json`. Every image must have a matching mask — a missing one is an error.

To train on image + mask, set `data.module.use_mask: True` and `model.input_channels: 2` (the datamodule concatenates the mask as channel 1). For external inference on such a run, pass `--mask-dir` to `predict_external.py`. Full details, including the co-registration requirement, are in [docs/preprocessing-ct.md](docs/preprocessing-ct.md) / [docs/preprocessing-mri.md](docs/preprocessing-mri.md).

For the per-modality pipeline details (resampling, intensity statistics, normalization differences), see [docs/preprocessing-ct.md](docs/preprocessing-ct.md) and [docs/preprocessing-mri.md](docs/preprocessing-mri.md).

## Suggested directory layout

The data module doesn't enforce a layout — it just needs `img_dir` (a folder of preprocessed `.b2nd` files) and `csv_file` (the splits/labels CSV). A tidy convention:

```
dataset/
└── <data_name>/
    ├── raw/                    <- original .nii.gz files (kept for re-preprocessing)
    ├── masks/                  <- optional co-registered <image_id>_mask.nii.gz files
    ├── preprocessing.json      <- written by scripts/preprocess_ct.py / scripts/preprocess_mri.py
    ├── preprocessed_b2nd/      <- .b2nd output from those same scripts
    │   └── <image_id>/
    │       ├── <image_id>.b2nd
    │       └── <image_id>_mask.b2nd   <- only when --mask-dir is given
    └── split_labels.csv        <- splits/labels/folds CSV
```

Point the preprocess scripts' `--out-root` at `dataset/<data_name>/`, and your training config's `data.module.img_dir` at `dataset/<data_name>/preprocessed_b2nd/`.

### CT

```bash
python scripts/preprocess_ct.py \
    --in-dir /path/to/raw/CT/imagesTr /path/to/raw/CT/imagesVal \
    --out-root /path/to/dataset/Dataset001_LiverROI \
    --num-workers 8
```

| Flag | Description |
|---|---|
| `--in-dir` | One or more directories of raw `.nii.gz` CT images. Stats and median spacing span all of them. |
| `--out-root` | Output directory for this dataset. The script writes `<out-root>/preprocessed_b2nd/<image_id>/<image_id>.b2nd` and `<out-root>/preprocessing.json`. |
| `--mask-dir` | **Optional.** Directory of co-registered masks named `<image_id>_mask.nii.gz`. When set, each mask is saved as `<image_id>_mask.b2nd` and every image must have a matching mask. See [docs/preprocessing-ct.md](docs/preprocessing-ct.md). |
| `--target-spacing Z Y X` | Target voxel spacing in mm. **Optional.** If omitted, defaults to the per-axis median spacing across all input images. |
| `--skip-resample` | Skip the resampling step entirely (use native spacing). |
| `--num-workers` | Parallel processes for the stats / spacing / per-case passes. Default `8`. |
| `--stats-mean / --stats-std / --stats-pct-00-5 / --stats-pct-99-5` | Optional pre-supplied stats; bypasses the dataset-wide stats pass. All four must be set together. |

### MRI

```bash
python scripts/preprocess_mri.py \
    --in-dir /path/to/raw/MRI/imagesTr /path/to/raw/MRI/imagesVal \
    --out-root /path/to/dataset/Dataset017_OpenNeuro \
    --num-workers 8
```

| Flag | Description |
|---|---|
| `--in-dir` | One or more directories of raw `.nii.gz` MR images. Median spacing spans all of them. |
| `--out-root` | Output directory for this dataset. The script writes `<out-root>/preprocessed_b2nd/<image_id>/<image_id>.b2nd` and `<out-root>/preprocessing.json`. |
| `--mask-dir` | **Optional.** Directory of co-registered masks named `<image_id>_mask.nii.gz`. When set, each mask is saved as `<image_id>_mask.b2nd` and every image must have a matching mask. See [docs/preprocessing-mri.md](docs/preprocessing-mri.md). |
| `--target-spacing Z Y X` | Target voxel spacing in mm. **Optional.** If omitted, defaults to the per-axis median spacing across all input images. |
| `--skip-resample` | Skip the resampling step entirely. |
| `--num-workers` | Parallel processes. Default `8`. |

# Training

Every training run is driven by one **self-contained** config file in `configs/train_*.yaml`. Each config has env + data + model + trainer settings inline — there is no `defaults:` composition. Launch with:

```bash
python scripts/train.py --config-name=<config_name>
```

(or `medclass-train --config-name=<config_name>` — the installed console script does the same thing).

If you forget the flag, `train.py` prints a friendly error listing the available configs.

## Provided configs

| Config | Task | Notes |
|---|---|---|
| `train_classification` | `Classification` | Template. Set `num_classes`, paths, `subtask`, `loss_fn`. Copy this for a new classification experiment. |

See [docs/training-configs.md](docs/training-configs.md) for the config layout and which knobs matter when.

## Examples

Fill in the placeholders in `train_classification.yaml` (or copy it first), then:

```bash
python scripts/train.py --config-name=train_classification
```

Override any key on the CLI without editing the file (single GPU, weighted CE for class imbalance, smaller LR):

```bash
python scripts/train.py --config-name=train_classification \
    trainer.devices=1 \
    model.lr=5e-4 \
    model.loss_fn=weighted_ce
```

Fine-tune from a checkpoint:

```bash
python scripts/train.py --config-name=train_classification \
    model.pretrained=True \
    model.chpt_path=/path/to/checkpoint.ckpt
```

## Adding a new experiment

Copy an existing `configs/train_*.yaml` to `configs/train_<your_name>.yaml`, edit the placeholders (`img_dir`, `csv_file`, `num_classes`, project, name), then `python scripts/train.py --config-name=train_<your_name>`. See [docs/custom-datasets.md](docs/custom-datasets.md) for the full workflow, and [docs/tasks-and-losses.md](docs/tasks-and-losses.md) for the `subtask` / `loss_fn` options. [docs/output-layout.md](docs/output-layout.md) explains where runs land on disk.

For running inference on a trained model, see [docs/inference.md](docs/inference.md).

---

## Acknowledgement

This codebase is adapted from [SSL3D_classification](https://github.com/constantinulrich/SSL3D_classification) and [nnSSL](https://github.com/MIC-DKFZ/nnssl). 

