# Adding your own dataset

If your dataset can be described as "a directory of preprocessed `.b2nd` files plus a CSV of splits/labels/folds," you don't need to write any new Python — just copy an existing training config and point it at your data.

## The common case: copy a training config

1. **Preprocess your data** with `scripts/preprocess_ct.py` or `scripts/preprocess_mri.py`. The output layout is:

   ```
   <out-root>/
       preprocessing.json      <- needed by predict_external.py later; cli.py auto-copies it into runs
       preprocessed_b2nd/
           <image_id>.b2nd
   ```

2. **Build a CSV** with `image_name`, `split`, `fold`, and a label column. See [data-csv-format.md](data-csv-format.md) for the schema.

3. **Copy a training config.** Pick the closest existing `configs/train_*.yaml` to your task:
   - `configs/train_age_ord_reg.yaml` — ordinal regression (CORAL)
   - `configs/train_age_reg.yaml` — plain regression (MSE)

   Then copy it to `configs/train_<your_name>.yaml` and edit the placeholders:

   ```yaml
   data:
     module:
       _target_: medregression3d.data.datamodules.AgeReg_DataModule
       name: YourDatasetName
       img_dir: /path/to/<out-root>/preprocessed_b2nd   # points at the b2nd files
       csv_file: /path/to/splits_labels.csv
       label_column: label                  # or "age", etc.
       batch_size: 4
     cv:
       k: 1                                 # 1 = single run, >1 = k-fold CV
     num_classes: 100                       # ordinal levels (e.g. 100 = ages 0..99)
     patch_size: [160, 160, 160]

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
   medreg-train --config-name=train_<your_name>
   ```

   Any key can be overridden on the CLI for quick experiments:

   ```bash
   python scripts/train.py --config-name=train_<your_name> \
       trainer.devices=2 \
       model.lr=5e-4 \
       data.module.batch_size=8
   ```

## When you actually need a custom `DataModule`

Write a new `DataModule` only if your data doesn't fit the `AgeReg_DataModule` pattern — for example:

- **Multiple images per case** (e.g., paired modalities, multi-channel inputs).
- **Per-fold splits** where the same image is `train` in one fold and `val` in another.
- **Non-standard label structure** (e.g., multi-label classification, segmentation targets, censored survival times).
- **A different file format** than `.b2nd`.

In that case, mirror [`src/medregression3d/data/datamodules.py`](../src/medregression3d/data/datamodules.py) as a starting point: subclass `BaseDataModule`, accept your paths via `__init__`, and instantiate your `Dataset` in `setup()`. Then point your config's `data.module._target_` at your new class.
