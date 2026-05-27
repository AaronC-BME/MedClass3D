# MRI preprocessing — pipeline details

For the command and argument reference, see the **Dataset preprocessing** section in the [README](../README.md).

Output layout:

```
<out-root>/
    preprocessing.json          <- modality, target spacing, normalization='per_case_zscore'
    preprocessed_b2nd/
        <image_id>.b2nd         <- one file per input image
```

The `preprocessing.json` sidecar records every knob needed to replay the same preprocessing at external-inference time. `cli.py` copies it into the training run's `Configs/` directory; `predict_external.py` reads it back from there. See [inference.md](inference.md) for the full flow.

The MRI script mirrors `preprocess_ct.py` but with two key differences:

- **No dataset-wide stats pass.** MRI has no absolute intensity reference (unlike CT's HU scale), so intensities are not comparable across scanners or sequences. Each case is z-scored independently on its own foreground.
- **Foreground = voxels > 0.** This assumes the input MRIs are skull-stripped or otherwise have a zero background. For data with non-zero air background, mask it out first.

## Per-case pipeline

1. Resample to a target spacing. By default the script reads headers across every input image, takes the per-axis median spacing, and uses that. Override with `--target-spacing Z Y X`.
2. Crop to the non-zero bounding box (typically trims a sizeable margin on skull-stripped MRI).
3. Per-case z-score normalization on the foreground mask.
4. Save as Blosc2 at `<out-root>/preprocessed_b2nd/<image_id>.b2nd`.

## Alternative invocations

Force 1mm isotropic spacing:
```bash
python scripts/preprocess_mri.py \
    --in-dir /path/to/raw/MRI/images \
    --out-root /path/to/dataset/Dataset017_OpenNeuro \
    --target-spacing 1 1 1 \
    --num-workers 8
```
