# CT preprocessing — pipeline details

For the command and argument reference, see the **Dataset preprocessing** section in the [README](../README.md).

The CT script is dataset-agnostic and works on any CT dataset given one or more directories of `.nii.gz` images. Output layout (one sub-directory per subject):

```
<out-root>/
    preprocessing.json          <- modality, target spacing, CT stats (mean/std/p0.5/p99.5), has_masks
    preprocessed_b2nd/
        <image_id>/
            <image_id>.b2nd          <- image
            <image_id>_mask.b2nd     <- mask (only when --mask-dir is given)
```

The `preprocessing.json` sidecar records every knob needed to replay the same preprocessing at external-inference time. `cli.py` copies it into the training run's `Configs/` directory; `predict_external.py` reads it back from there. See [inference.md](inference.md) for the full flow.

## Two-pass pipeline

**Pass 1 — Compute dataset-wide intensity statistics.** Scans all `.nii.gz` files once, sampling up to 10,000 foreground voxels per case (HU > -500). Aggregates global mean, std, and 0.5 / 99.5 percentiles. You can skip this pass by supplying stats directly via `--stats-mean`, `--stats-std`, `--stats-pct-00-5`, and `--stats-pct-99-5`.

**Pass 2 — Per-case processing.**
1. Resample to a target spacing. By default the script reads headers across every input image, takes the per-axis median spacing, and uses that. Override with `--target-spacing Z Y X` to force a specific spacing (e.g. `1 1 1` for 1mm isotropic). Cases already at the target spacing skip resampling automatically.
2. Crop to the non-zero bounding box (trims zero-padded edges; CT air at -1000 HU is preserved).
3. CT-normalize: clip to the dataset-wide `[percentile_00_5, percentile_99_5]` range, then z-score using the dataset-wide mean and std.
4. Save as Blosc2 at `<out-root>/preprocessed_b2nd/<image_id>/<image_id>.b2nd`.

## Masks as a second input channel

Supplying `--mask-dir` attaches a co-registered segmentation/ROI mask to each image, letting you train with image + mask as a 2-channel input. This is useful when an ROI (organ, lesion, region) should be made explicit to the network rather than learned implicitly.

**Naming.** For an image `<image_id>.nii.gz`, the matching mask must be named `<image_id>_mask.nii.gz` inside `--mask-dir` (a `.nii` extension also works). Every image must have a matching mask — a missing one aborts the run with the list of offenders, so the dataset can never silently drop to a 1-channel case.

**Co-registration requirement.** The mask must live on the *same voxel grid* as its image: identical array shape and voxel spacing *before* preprocessing (i.e. it is a segmentation of that exact image, not a resampled/reoriented copy). The script verifies this per case and skips any mask whose shape or spacing disagrees. The reason is step 2 below: the mask is cropped with the bounding box derived from the **image**, which is only valid if the two share a grid.

**How the mask is processed** (mirrors the image's geometry, but never its intensities):
1. Resample to the target spacing with **nearest-neighbour** interpolation (`is_seg=True, order=0`), so label values are preserved exactly — no fractional labels, correct for multi-label masks.
2. Crop with the **image's** non-zero bounding box, keeping the two channels voxel-aligned. (The mask is deliberately *not* run through the image's nonzero-cropping path, which would write `-1` into the background and pollute the raw label values.)
3. **No intensity normalization** — the mask is saved raw as `int8`, preserving its 0/1 (or 0..N label) values.
4. Save as Blosc2 at `<out-root>/preprocessed_b2nd/<image_id>/<image_id>_mask.b2nd`.

`preprocessing.json` records `"has_masks": true`, which `predict_external.py` reads back to require a `--mask-dir` at inference time.

**Augmentation.** At training time the mask rides batchgenerators' `segmentation` key: it receives only the spatial transforms (rotation, scaling, mirroring — label-preserving), never the intensity transforms (noise, blur, brightness, contrast, gamma) that would corrupt a label channel. The datamodule then concatenates it onto the image as channel 1.

**To train on it**, set in your config:
```yaml
data:
  module:
    use_mask: True
model:
  input_channels: 2
```

Example:
```bash
python scripts/preprocess_ct.py \
    --in-dir /path/to/raw/CT/images \
    --mask-dir /path/to/raw/CT/masks \
    --out-root /path/to/dataset/Dataset001_LiverROI \
    --num-workers 8
```

## Alternative invocations

Force 1mm isotropic spacing:
```bash
python scripts/preprocess_ct.py \
    --in-dir /path/to/raw/CT/images \
    --out-root /path/to/dataset/Dataset001_LiverROI \
    --target-spacing 1 1 1 \
    --num-workers 8
```
