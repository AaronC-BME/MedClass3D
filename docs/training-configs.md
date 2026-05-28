# Training configs

Every training run is driven by **one self-contained YAML** in `configs/train_*.yaml`. Each config carries the full env + data + model + trainer + metrics settings inline — there is no `defaults:` composition and no `configs/{env,model,data}/` subdirectories. One file = one experiment, fully readable from top to bottom.

Launch with `--config-name=<basename>` (no `.yaml` extension):

```bash
python scripts/train.py --config-name=train_classification
```

If you forget the flag, `scripts/train.py` prints a friendly error listing the configs it found.

## Provided configs

### `train_classification.yaml` — Classification template

The starting point for any classification experiment.

| Setting | Value |
|---|---|
| `model._target_` | `medclass3d.models.backbones.resenc.ResEncoder_Classifier` |
| `model.task` | `'Classification'` |
| `model.subtask` | `'multiclass'` (set to `'multilabel'` for multi-label problems) |
| `model.loss_fn` | `null` → `CrossEntropyLoss` / `BCEWithLogitsLoss` |
| `model.pretrained` | `False` |
| `data.module._target_` | `medclass3d.data.datamodules.Class_DataModule` |
| `data.cv.k` | `5` (5-fold cross-validation) |
| `data.num_classes` | `2` (set to your dataset's number of classes) |
| `trainer.max_epochs` | `200` |
| Callbacks | `EarlyStopping` (monitor `Val/loss`, patience 50), `ModelCheckpoint` (monitor `Val/Accuracy`, mode `max`) |

Things you must fill in before training:
- `data.module.img_dir` — path to your preprocessed `.b2nd` files
- `data.module.csv_file` — path to your splits/labels CSV
- `data.num_classes` — number of classes in your dataset
- `trainer.logger.project` / `trainer.logger.name` — W&B project and run name
- `output_dir` — either via the `EXPERIMENT_LOCATION` env var or by replacing the `<path_to_output>` fallback

Common knobs to flip:

```bash
# Weighted CE for class imbalance (auto pulls class_weights from train split)
python scripts/train.py --config-name=train_classification model.loss_fn=weighted_ce

# Multilabel
python scripts/train.py --config-name=train_classification model.subtask=multilabel

# Balanced batch sampling (K-class equal-balance with replacement)
python scripts/train.py --config-name=train_classification data.module.use_balanced_sampling=True
```

(See [tasks-and-losses.md](tasks-and-losses.md) for the full list of `loss_fn` values.)

## MLP-head variant

The default `ResEncoder_Classifier` uses a single-linear head. There's also `ResEncoder_Classifier_MLP`, which swaps it for a two-layer MLP (256 → 128 → num_classes). To use it, change the `_target_`:

```yaml
model:
  _target_: medclass3d.models.backbones.resenc.ResEncoder_Classifier_MLP
```

No new config file needed.

## Adding your own experiment

Copy `configs/train_classification.yaml` to `configs/train_<your_name>.yaml`, edit the placeholders, and launch with `--config-name=train_<your_name>`. See [custom-datasets.md](custom-datasets.md) for the full step-by-step.

## Structure of a config file

Every `train_*.yaml` is organized as:

```yaml
# Top-level: output paths, hydra meta, run misc, metrics
output_dir: ...
hydra: { ... }
seed: False
val_only: False
metrics: [ ... ]   # acc, balanced_acc, f1, f1_per_class, pr, top5acc, auroc, ap

# Data: datamodule, transforms, CV, num_classes, patch_size
data:
  module: { _target_, name, img_dir, csv_file, label_column, batch_size, use_balanced_sampling, ... }
  cv: { k }
  num_classes: ...
  patch_size: ...

# Model: target class, task + subtask + loss_fn, optimizer, scheduler, regularization
model:
  _target_: ...
  task: 'Classification'
  subtask: 'multiclass'
  loss_fn: ...
  classification_head_dropout: ...
  optimizer: ...
  # ...

# Trainer: Lightning trainer args, callbacks (early_stopping, checkpoint, LR monitor, progress bar), logger (W&B)
trainer:
  _target_: lightning.pytorch.Trainer
  devices: ...
  callbacks: { early_stopping, checkpoint, ... }
  logger: { ... }
```

Any key can be overridden on the CLI with dotted-path syntax (`model.lr=5e-4`, `trainer.devices=2`, `data.module.batch_size=8`).
