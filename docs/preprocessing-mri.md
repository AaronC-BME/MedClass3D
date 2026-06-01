# MRI preprocessing — pipeline details

For the command and argument reference, see the **Dataset preprocessing** section in the [README](../README.md).

Output layout (one sub-directory per subject):

```
<out-root>/
    preprocessing.json          <- modality, target spacing, normalization='per_case_zscore', has_masks
    preprocessed_b2nd/
        <image_id>/
            <image_id>.b2nd          <- image
            <image_id>_mask.b2nd     <- mask (only when --mask-dir is given)
```

The `preprocessing.json` sidecar records every knob needed to replay the same preprocessing at external-inference time. `cli.py` copies it into the training run's `Configs/` directory; `predict_external.py` reads it back from there. See [inference.md](inference.md) for the full flow.

The MRI script mirrors `preprocess_ct.py` but with two key differences:

- **No dataset-wide stats pass.** MRI has no absolute intensity reference (unlike CT's HU scale), so intensities are not comparable across scanners or sequences. Each case is z-scored independently on its own foreground.
- **Foreground = voxels > 0.** This assumes the input MRIs are skull-stripped or otherwise have a zero background. For data with non-zero air background, mask it out first.

## Per-case pipeline

1. Resample to a target spacing. By default the script reads headers across every input image, takes the per-axis median spacing, and uses that. Override with `--target-spacing Z Y X`.
2. Crop to the non-zero bounding box (typically trims a sizeable margin on skull-stripped MRI).
3. Per-case z-score normalization on the foreground mask.
4. Save as Blosc2 at `<out-root>/preprocessed_b2nd/<image_id>/<image_id>.b2nd`.

## Masks as a second input channel

Supplying `--mask-dir` attaches a co-registered segmentation/ROI mask to each image, letting you train with image + mask as a 2-channel input. The behavior is identical to the CT script (see [preprocessing-ct.md](preprocessing-ct.md) for the full rationale); the only difference is that MRI normalization is per-case z-score and, like CT normalization, is applied to the **image only** — the mask is never normalized.

**Naming.** For an image `<image_id>.nii.gz`, the matching mask must be named `<image_id>_mask.nii.gz` inside `--mask-dir` (a `.nii` extension also works). Every image must have a matching mask — a missing one aborts the run with the list of offenders.

**Co-registration requirement.** The mask must share its image's voxel grid (identical array shape and spacing before preprocessing), because it is cropped with the bounding box derived from the image. The script verifies this per case and skips any mask whose shape or spacing disagrees.

**How the mask is processed:**
1. Resample to the target spacing with **nearest-neighbour** interpolation (`is_seg=True, order=0`), preserving label values exactly.
2. Crop with the **image's** non-zero bounding box (not the mask's own), keeping the channels voxel-aligned and avoiding the `-1` background labels that the image's nonzero-cropping path would inject.
3. **No normalization** — saved raw as `int8`.
4. Save as Blosc2 at `<out-root>/preprocessed_b2nd/<image_id>/<image_id>_mask.b2nd`.

`preprocessing.json` records `"has_masks": true`, which `predict_external.py` reads back to require a `--mask-dir` at inference time. At training time the mask rides batchgenerators' `segmentation` key (spatial augmentation only, no intensity transforms) and the datamodule concatenates it as channel 1.

**To train on it**, set `data.module.use_mask: True` and `model.input_channels: 2` in your config.

Example:
```bash
python scripts/preprocess_mri.py \
    --in-dir /path/to/raw/MRI/images \
    --mask-dir /path/to/raw/MRI/masks \
    --out-root /path/to/dataset/Dataset017_OpenNeuro \
    --num-workers 8
```

## Alternative invocations

Force 1mm isotropic spacing:
```bash
python scripts/preprocess_mri.py \
    --in-dir /path/to/raw/MRI/images \
    --out-root /path/to/dataset/Dataset017_OpenNeuro \
    --target-spacing 1 1 1 \
    --num-workers 8
```
