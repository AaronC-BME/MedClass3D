# Adding your own dataset

If your dataset can be described as "a directory of preprocessed `.b2nd` files plus a CSV of splits/labels/folds," you don't need to write any new Python — just copy an existing training config and point it at your data.

## The common case: copy a training config

1. **Preprocess your data** with `scripts/preprocess_ct.py` or `scripts/preprocess_mri.py`. The output layout is:

   ```
   <out-root>/
       preprocessing.json      <- needed by predict_external.py later; cli.py auto-copies it into runs
       preprocessed_b2nd/
           <image_id>/
               <image_id>.b2nd          <- image
               <image_id>_mask.b2nd     <- mask (only if you passed --mask-dir)
   ```

   To feed an ROI/segmentation mask as a second input channel, preprocess with
   `--mask-dir` and set `data.module.use_mask: True` + `model.input_channels: 2`
   in your config. See the preprocessing docs for details.

2. **Build a CSV** with `image_name`, `split`, `fold`, and a label column (integer class indices for classification). See [data-csv-format.md](data-csv-format.md) for the schema.

3. **Copy `configs/train_classification.yaml` to `configs/train_<your_name>.yaml`** and edit the placeholders:

   ```yaml
   data:
     module:
       _target_: medclass3d.data.datamodules.Class_DataModule
       name: YourDatasetName
       img_dir: /path/to/<out-root>/preprocessed_b2nd   # points at the b2nd files
       csv_file: /path/to/splits_labels.csv
       label_column: label                  # or "pathology", etc.
       use_balanced_sampling: False         # set True for KClassBalancedBatchSampler
       batch_size: 4
     cv:
       k: 1                                 # 1 = single run, >1 = k-fold CV
     num_classes: 4                         # number of classes in your dataset
     patch_size: [160, 160, 160]

   model:
     subtask: 'multiclass'                  # or 'multilabel'
     loss_fn: null                          # null=CE; or focal / weighted_focal / weighted_ce / topk10

   trainer:
     logger:
       project: YourDatasetName
       name: your_run_name                  # or ${make_group_name:} for an auto timestamp
   ```

   `output_dir` defaults to the `EXPERIMENT_LOCATION` env var; set it once in your shell (`export EXPERIMENT_LOCATION=/path/to/outputs`) and every config picks it up. Or hard-code the fallback in your config.

4. **Run training.**

   ```bash
   python scripts/train.py --config-name=train_<your_name>
   ```

   Or the installed console script:

   ```bash
   medclass-train --config-name=train_<your_name>
   ```

   Any key can be overridden on the CLI for quick experiments:

   ```bash
   python scripts/train.py --config-name=train_<your_name> \
       trainer.devices=2 \
       model.lr=5e-4 \
       data.module.batch_size=8 \
       model.loss_fn=weighted_ce
   ```

## When you actually need a custom `DataModule`

Write a new `DataModule` only if your data doesn't fit the `Class_DataModule` pattern — for example:

- **Multiple images per case** (e.g., paired modalities, multi-channel inputs).
- **Per-fold splits** where the same image is `train` in one fold and `val` in another.
- **Non-trivial multilabel encoding** beyond a single integer column (e.g., a serialized binary vector per row).
- **A different file format** than `.b2nd`.

In that case, mirror [`src/medclass3d/data/datamodules.py`](../src/medclass3d/data/datamodules.py) as a starting point: subclass `BaseDataModule`, accept your paths via `__init__`, and instantiate your `Dataset` in `setup()`. If you want class-weighted losses (`weighted_ce` / `weighted_focal`) to keep working, expose a `class_weights` attribute on the datamodule (a 1-D `torch.FloatTensor` of length `num_classes`); the trainer picks it up in `setup()`. Then point your config's `data.module._target_` at your new class.
