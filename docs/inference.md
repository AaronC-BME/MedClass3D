# Inference

There are two prediction scripts, each for a distinct use case:

| Script | Use case |
|---|---|
| [`scripts/predict_test.py`](../scripts/predict_test.py) | Re-run val + test inference on the splits defined in the training CSV. For evaluating a trained model on its own held-out data. |
| [`scripts/predict_external.py`](../scripts/predict_external.py) | Run a trained model on a directory of raw `.nii.gz` files outside the training CSV. Does the preprocessing internally using the same parameters the model was trained on. |

Both scripts use argparse (no Hydra config to maintain) — pass `--help` to either for the full flag list.

## How preprocessing parameters travel from training to external prediction

For external inference to be valid, the preprocessing applied at inference time must exactly match what the model was trained on. The flow is:

```
preprocess_{ct,mri}.py
   writes  ──►  <out_root>/preprocessing.json
                <out_root>/preprocessed_b2nd/<image_id>/<image_id>.b2nd  (+ <image_id>_mask.b2nd)

train.py
   reads  ──►   <out_root>/preprocessing.json   (next to img_dir)
   copies ──►   <run_dir>/Configs/preprocessing.json   (alongside config.yaml)

predict_external.py
   reads  ──►   <run_dir>/Configs/preprocessing.json
   replays the same target_spacing + normalization on each new .nii.gz
```

The sidecar records:
- `modality` — `'ct'` or `'mri'`.
- `target_spacing` — Z/Y/X in mm. Null if `--skip-resample` was used.
- `skip_resample`, `resampling_order`.
- For CT: `stats` (`mean`, `std`, `percentile_00_5`, `percentile_99_5`) and `foreground_hu_threshold`.
- For MRI: `foreground_threshold` and `normalization: per_case_zscore` (per-case, no shared state).
- `has_masks` — `true` if the dataset was preprocessed with `--mask-dir` (image + mask 2-channel input).

If a run's `Configs/` is missing `preprocessing.json` (older runs from before this mechanism existed), `predict_external.py` will refuse to run. Either re-preprocess and re-train, or manually copy a sidecar with the correct values into the run's `Configs/` directory.

If `has_masks` is `true`, `predict_external.py` requires `--mask-dir` (a directory of `<image_id>_mask.nii.gz` masks co-registered to the input images); it preprocesses and feeds them as the second channel exactly as at training time.

---

## `predict_test.py` — re-run held-out splits

For each of `val` and `test`, this script:

1. Builds the datamodule from `<run_dir>/Configs/config.yaml`.
2. Loads the best-Val_acc checkpoint from `<run_dir>/folds/<fold>/` (or `last.ckpt` with `--prefer-last`).
3. Runs `trainer.predict()` and reads ground-truth class labels from the training CSV.
4. Writes:
   - `predictions_<split>_fold<k>.xlsx` — `PatientID`, `GroundTruth`, `Pred`, `Prob_0`, `Prob_1`, …
   - `confusion_matrix_<split>_fold<k>.png` — raw confusion matrix.
   - `summary_<split>.csv` — `N`, `Accuracy`, `BalancedAccuracy`, `F1_macro`, `AUROC`.

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--run-dir` | — (required) | Training run directory (`<output_dir>/<dataset>/<run_name>`). Must contain `Configs/config.yaml` and `folds/<k>/*.ckpt`. |
| `--fold` | `0` | Which fold's checkpoint to load. |
| `--pred-dir` | `<run-dir>/predictions/` | Output directory. |
| `--metrics` | `acc balanced_acc f1 auroc` | Metric names forwarded to the model. |
| `--prefer-last` | off | Use `last.ckpt` instead of best-Val_acc. |

### Examples

```bash
# Evaluate the default fold of a run
python scripts/predict_test.py --run-dir /path/to/<output_dir>/<dataset>/<run_name>

# Pick a different fold and write outputs elsewhere
python scripts/predict_test.py \
    --run-dir /path/to/<output_dir>/<dataset>/<run_name> \
    --fold 2 \
    --pred-dir /tmp/eval-fold2
```

---

## `predict_external.py` — predict on raw NIfTI files

For inference on data that wasn't part of the training CSV. The script preprocesses each input on the fly using the training-time parameters, then runs the model.

### Per-input pipeline

1. Load the `.nii.gz`.
2. Resample to the training-time target spacing (skipped if the input is already at target or `skip_resample` was set at preprocessing time).
3. Crop to the non-zero bounding box.
4. Normalize:
   - **CT** — clip to the stored `[p0.5, p99.5]` and z-score with the stored mean/std.
   - **MRI** — per-case z-score on the foreground (voxels > 0).
5. Save as a temporary `.b2nd`.

After all inputs are preprocessed, the temp dir is fed to the model's `test_transforms` (center crop to patch size) and `trainer.predict()` runs once. The temp `.b2nd` files are deleted at the end unless `--keep-preprocessed` is set.

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--run-dir` | — (required) | Training run dir. Must contain `Configs/{config.yaml, preprocessing.json}`. |
| `--input-dir` | — (required) | Directory of raw `.nii.gz` images to predict on. |
| `--fold` | `0` | Which fold's checkpoint to load. |
| `--pred-dir` | `<run-dir>/predictions_external/` | Where `predictions.csv` is written. |
| `--keep-preprocessed` | off | Keep the intermediate `.b2nd` files (in `<pred-dir>/external_preprocessed_b2nd/`) instead of deleting them. |
| `--metrics` | `acc balanced_acc f1 auroc` | Metric names forwarded to the model. |
| `--prefer-last` | off | Use `last.ckpt` instead of best-Val_acc. |

### Output

`<pred_dir>/predictions.csv` — columns `PatientID`, `Pred` (argmax class), and one `Prob_<i>` per class (softmax probabilities for multiclass / sigmoid for multilabel). One row per successfully-preprocessed input. No ground truth — external data has no labels.

### Examples

```bash
# Predict on every .nii.gz in /data/new_cohort/ using fold 0 of a trained run
python scripts/predict_external.py \
    --run-dir /path/to/<output_dir>/<dataset>/<run_name> \
    --input-dir /data/new_cohort

# Keep the intermediate .b2nd files for re-use with a different checkpoint later
python scripts/predict_external.py \
    --run-dir /path/to/<output_dir>/<dataset>/<run_name> \
    --input-dir /data/new_cohort \
    --pred-dir /tmp/new_cohort_eval \
    --keep-preprocessed
```
