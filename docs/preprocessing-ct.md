# CT preprocessing — pipeline details

For the command and argument reference, see the **Dataset preprocessing** section in the [README](../README.md).

The CT script is dataset-agnostic and works on any CT dataset given one or more directories of `.nii.gz` images. Output layout:

```
<out-root>/
    preprocessing.json          <- modality, target spacing, CT stats (mean/std/p0.5/p99.5)
    preprocessed_b2nd/
        <image_id>.b2nd         <- one file per input image
```

The `preprocessing.json` sidecar records every knob needed to replay the same preprocessing at external-inference time. `cli.py` copies it into the training run's `Configs/` directory; `predict_external.py` reads it back from there. See [inference.md](inference.md) for the full flow.

## Two-pass pipeline

**Pass 1 — Compute dataset-wide intensity statistics.** Scans all `.nii.gz` files once, sampling up to 10,000 foreground voxels per case (HU > -500). Aggregates global mean, std, and 0.5 / 99.5 percentiles. You can skip this pass by supplying stats directly via `--stats-mean`, `--stats-std`, `--stats-pct-00-5`, and `--stats-pct-99-5`.

**Pass 2 — Per-case processing.**
1. Resample to a target spacing. By default the script reads headers across every input image, takes the per-axis median spacing, and uses that. Override with `--target-spacing Z Y X` to force a specific spacing (e.g. `1 1 1` for 1mm isotropic). Cases already at the target spacing skip resampling automatically.
2. Crop to the non-zero bounding box (trims zero-padded edges; CT air at -1000 HU is preserved).
3. CT-normalize: clip to the dataset-wide `[percentile_00_5, percentile_99_5]` range, then z-score using the dataset-wide mean and std.
4. Save as Blosc2 at `<out-root>/preprocessed_b2nd/<image_id>.b2nd`.

## Alternative invocations

Force 1mm isotropic spacing:
```bash
python scripts/preprocess_ct.py \
    --in-dir /path/to/raw/CT/images \
    --out-root /path/to/dataset/Dataset001_LiverROI \
    --target-spacing 1 1 1 \
    --num-workers 8
```
