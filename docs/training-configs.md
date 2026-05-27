# Training configs

Every training run is driven by **one self-contained YAML** in `configs/train_*.yaml`. Each config carries the full env + data + model + trainer + metrics settings inline — there is no `defaults:` composition and no `configs/{env,model,data}/` subdirectories. One file = one experiment, fully readable from top to bottom.

Launch with `--config-name=<basename>` (no `.yaml` extension):

```bash
python scripts/train.py --config-name=train_age_ord_reg
```

If you forget the flag, `scripts/train.py` prints a friendly error listing the configs it found.

## Provided configs

### `train_age_ord_reg.yaml` — Ordinal regression template

The starting point for any ordinal-regression experiment.

| Setting | Value |
|---|---|
| `model._target_` | `medregression3d.models.backbones.resenc.ResEncoder_OrdinalRegressor` |
| `model.task` | `'Ordinal_Regression'` |
| `model.loss_fn` | `null` → CORAL loss (default for ordinal regression) |
| `model.pretrained` | `True` |
| `data.module._target_` | `medregression3d.data.datamodules.AgeReg_DataModule` |
| `data.cv.k` | `5` (5-fold cross-validation) |
| `data.num_classes` | `100` (ordinal levels) |
| `trainer.max_epochs` | `200` |

Pick this when:
- Your label is an integer-valued ordered quantity (age in years, severity grade, etc.) and you want to exploit the ordering.
- You're starting from a pretrained `ResEncoder` checkpoint.

Things you must fill in before training:
- `data.module.img_dir` — path to your preprocessed `.b2nd` files
- `data.module.csv_file` — path to your splits/labels CSV
- `trainer.logger.project` / `trainer.logger.name` — W&B project and run name
- `output_dir` — either via the `EXPERIMENT_LOCATION` env var or by replacing the `<path_to_output>` fallback

The CORAL head emits `num_classes - 1` logits. To swap to a different ordinal loss without editing the file:

```bash
python scripts/train.py --config-name=train_age_ord_reg model.loss_fn=focal
```

(See [tasks-and-losses.md](tasks-and-losses.md) for the full list of `loss_fn` values.)

### `train_age_reg.yaml` — Plain regression template

The starting point for any plain (non-ordinal) regression experiment.

| Setting | Value |
|---|---|
| `model._target_` | `medregression3d.models.backbones.resenc.ResEncoder_Regressor` |
| `model.task` | `'Regression'` |
| `model.loss_fn` | `null` → `MSELoss` |
| `model.pretrained` | `False` |
| `data.module._target_` | `medregression3d.data.datamodules.AgeReg_DataModule` |
| `data.cv.k` | `5` |
| `trainer.max_epochs` | `200` |

Pick this when:
- Your label is a continuous quantity and you don't care about discrete ordinal levels.
- You want MSE-style training rather than CORAL's binary-decomposition.

Fill in the same placeholders as `train_age_ord_reg.yaml` (`img_dir`, `csv_file`, `logger.project`, `logger.name`, `output_dir`).

The regression head emits a single scalar per sample by default.

## MLP-head variant

Both the ordinal and regression templates pin `ResEncoder_OrdinalRegressor` / `ResEncoder_Regressor` as the model `_target_`. There's also a third class, `ResEncoder_OrdinalRegressor_MLP`, which swaps the single-linear CORAL projection for a small MLP. To use it, just change the `_target_` line in your config:

```yaml
model:
  _target_: medregression3d.models.backbones.resenc.ResEncoder_OrdinalRegressor_MLP
```

No new config file needed.

## Adding your own experiment

Copy an existing template to `configs/train_<your_name>.yaml`, edit the placeholders, and launch with `--config-name=train_<your_name>`. See [custom-datasets.md](custom-datasets.md) for the full step-by-step.

## Structure of a config file

Every `train_*.yaml` is organized as:

```yaml
# Top-level: output paths, hydra meta, run misc, metrics
output_dir: ...
hydra: { ... }
seed: False
val_only: False
metrics: [ ... ]

# Data: datamodule, transforms, CV, num_classes, patch_size
data:
  module: { _target_, name, img_dir, csv_file, label_column, batch_size, ... }
  cv: { k }
  num_classes: ...
  patch_size: ...

# Model: target class, task + loss, optimizer, scheduler, regularization
model:
  _target_: ...
  task: ...
  loss_fn: ...
  optimizer: ...
  # ...

# Trainer: Lightning trainer args, callbacks (checkpoint, LR monitor, progress bar), logger (W&B)
trainer:
  _target_: lightning.pytorch.Trainer
  devices: ...
  callbacks: { ... }
  logger: { ... }
```

Any key can be overridden on the CLI with dotted-path syntax (`model.lr=5e-4`, `trainer.devices=2`, `data.module.batch_size=8`).
