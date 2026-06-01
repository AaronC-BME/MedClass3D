"""
General-purpose CT preprocessing for MedClass3D.

Reads raw NIfTI CT images, computes dataset-wide intensity statistics on the
foreground (HU > -500), then for each case:
  1. Resamples to a target spacing (auto-computed median by default, or set
     explicitly with --target-spacing).
  2. Crops to the non-zero bounding box (matches nnssl behavior).
  3. Applies CT normalization (clip to dataset percentiles + dataset z-score).
  4. Saves as Blosc2 in this layout (one sub-directory per subject):

        <out_root>/
            preprocessing.json          <- modality, target spacing, dataset stats
            preprocessed_b2nd/
                <id>/
                    <id>.b2nd           <- image, one per input image
                    <id>_mask.b2nd      <- mask (only when --mask-dir is given)

When --mask-dir is supplied, each image <id>.nii.gz must have a co-registered
mask named <id>_mask.nii.gz in that directory. The mask is resampled with
nearest-neighbour, cropped with the image's bounding box, saved raw (no
intensity normalization) as a second-channel file, and recorded via
"has_masks": true in preprocessing.json. Train with model.input_channels: 2
and data.module.use_mask: True.

The `preprocessing.json` sidecar lets `scripts/predict_external.py` faithfully
replay the same preprocessing on raw NIfTI files at inference time. `cli.py`
also copies it into each training run's `Configs/` dir so the run is
self-describing.

Notes:
  - This script is dataset-agnostic. Point --out-root at whatever directory
    you want to hold the preprocessed dataset (e.g. .../<your_dataset>/).
  - --in-dir accepts one or more directories. All .nii.gz files across all
    directories are processed into the same output folder, and dataset stats
    + median target spacing (when auto-computed) are taken across all of them.
    Useful when train/val images live in separate folders but must end up
    at a consistent resampled spacing.
  - When --target-spacing is omitted, the script does a header-only pass over
    every input image and uses the per-axis median spacing.
    Pass --target-spacing Z Y X to override.
  - Resampling order is 3 (cubic) for CT data, matching the SSL3D template.
  - If your data is already at the target spacing, resampling is a near-no-op
    (one short shape check) and adds negligible time. Pass --skip-resample to
    skip it explicitly.
  - Random patch extraction to 160^3 happens at training time via
    batchgenerators, not here. We preserve full resampled volumes.

Usage:
    python preprocess_ct.py \\
        --in-dir <dir_1> [<dir_2> ...] \\
        --out-root <path_to_output_dir> \\
        --num-workers 8
"""
import sys
import os
import argparse
import json
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, List

# Make the parent package importable when running this script directly,
# matching the pattern used by template_brain_preprocessing.py.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
)

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

from medclass3d.data.preprocessing.cropping import crop_to_nonzero
from medclass3d.data.preprocessing.normalization import CTNormalization
from medclass3d.data.preprocessing.blosc_helper import save_case, comp_blosc2_params
from medclass3d.data.preprocessing.default_resampling import resample_data_or_seg_to_spacing

# Suffix appended to a case id to find its mask in --mask-dir, and the suffix
# used for the preprocessed mask file on disk: <id>_mask.nii.gz -> <id>_mask.b2nd
MASK_SUFFIX = "_mask"

# CT-specific: voxels above this HU threshold count as foreground (tissue).
# Air -1000, lung parenchyma -800, fat -100, water 0, soft tissue +30 to +60,
# bone +400 and up. -500 cleanly separates air/lung from everything else.
CT_FOREGROUND_HU_THRESHOLD = -500.0

# How many foreground voxels to sample per case when computing dataset stats.
# 10000 matches nnU-Net's default. Subsampling keeps memory bounded; the
# percentiles and mean/std are accurate to within rounding error.
NUM_FOREGROUND_SAMPLES_PER_CASE = 10_000

# Cubic interpolation for CT image data. Matches SSL3D template default.
RESAMPLING_ORDER = 3


def _strip_nifti_ext(name: str) -> str:
    """Strip a trailing .nii.gz or .nii from a filename, returning the case id."""
    if name.endswith(".nii.gz"):
        return name[:-len(".nii.gz")]
    if name.endswith(".nii"):
        return name[:-len(".nii")]
    return name


def read_image_spacing(image_path: str) -> tuple:
    """Header-only read of voxel spacing. Returns (Z, Y, X) in mm."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    reader.ReadImageInformation()
    # SimpleITK GetSpacing returns (x, y, z); flip to match (Z, Y, X) array order.
    return reader.GetSpacing()[::-1]


# --------------------------------------------------------------------------- #
# Stats computation (Pass 1)
# --------------------------------------------------------------------------- #
def _sample_foreground_one_case(image_path: str) -> np.ndarray:
    """
    Load one NIfTI and return up to NUM_FOREGROUND_SAMPLES_PER_CASE
    foreground (HU > -500) voxel samples.

    Stats are computed on the *raw* (pre-resample) volume. This is correct
    because resampling preserves intensity distribution, and computing on raw
    data avoids double-paying the resample cost.
    """
    img = sitk.ReadImage(image_path)
    data = sitk.GetArrayFromImage(img)  # (Z, Y, X), HU values
    foreground = data[data > CT_FOREGROUND_HU_THRESHOLD]
    if foreground.size == 0:
        # Degenerate case (volume is mostly air?) - fall back to non-zero
        foreground = data[data != 0]
    if foreground.size == 0:
        return np.empty((0,), dtype=np.float32)

    if foreground.size > NUM_FOREGROUND_SAMPLES_PER_CASE:
        # Deterministic per-file seed so the stats are reproducible
        rng = np.random.default_rng(seed=hash(image_path) & 0xFFFFFFFF)
        idx = rng.choice(foreground.size, size=NUM_FOREGROUND_SAMPLES_PER_CASE, replace=False)
        foreground = foreground[idx]
    return foreground.astype(np.float32)


def compute_dataset_stats(
    image_paths: List[str],
    num_workers: int = 8,
) -> dict:
    """
    Compute dataset-wide foreground intensity statistics for CT normalization.
    Returns a dict with keys: mean, std, median, min, max, percentile_00_5,
    percentile_99_5.
    """
    print(f"Computing CT intensity stats from {len(image_paths)} images...")
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            samples_per_case = list(
                tqdm(pool.imap_unordered(_sample_foreground_one_case, image_paths),
                     total=len(image_paths))
            )
    else:
        samples_per_case = [_sample_foreground_one_case(p) for p in tqdm(image_paths)]

    samples_per_case = [s for s in samples_per_case if s.size > 0]
    if not samples_per_case:
        raise RuntimeError("Could not extract any foreground voxels from any case.")

    all_samples = np.concatenate(samples_per_case)
    stats = {
        "mean": float(np.mean(all_samples)),
        "std": float(np.std(all_samples)),
        "median": float(np.median(all_samples)),
        "min": float(np.min(all_samples)),
        "max": float(np.max(all_samples)),
        "percentile_00_5": float(np.percentile(all_samples, 0.5)),
        "percentile_99_5": float(np.percentile(all_samples, 99.5)),
    }
    return stats


# --------------------------------------------------------------------------- #
# Per-case processing (Pass 2)
# --------------------------------------------------------------------------- #
def _save_b2nd(data: np.ndarray, out_path_truncated: str) -> None:
    """Save a (C, Z, Y, X) array as Blosc2. ``out_path_truncated`` is the path
    without the ``.b2nd`` extension (save_case appends it)."""
    os.makedirs(os.path.dirname(out_path_truncated), exist_ok=True)
    block_size, chunk_size = comp_blosc2_params(
        data.shape, (160, 160, 160), data.itemsize
    )
    save_case(data, out_path_truncated, chunks=chunk_size, blocks=block_size)


def process_one_case(args: tuple) -> Optional[str]:
    """
    Worker function. Takes a tuple so it works with Pool.imap.
    Returns the case ID on success, None on failure.

    Pipeline order matters here:
        1. Load image (and mask, if provided)
        2. Resample to target spacing (changes voxel grid, preserves anatomy)
        3. Crop to nonzero (safe to do after resampling - bbox tracks anatomy)
        4. Normalize using dataset stats (image only; mask stays raw)
        5. Save

    The mask, when present, must be co-registered to its image (identical voxel
    grid). It is resampled with nearest-neighbour (is_seg=True, order 0) so label
    values are preserved, and cropped with the *image's* bounding box so the two
    channels stay aligned. It is NOT normalized.
    """
    (image_path, mask_path, out_dir, case_id,
     intensity_properties, target_spacing, skip_resample) = args

    try:
        # ---- 1. Load ----
        img = sitk.ReadImage(image_path)
        # SimpleITK GetSpacing returns (x, y, z); array is (z, y, x) — invert.
        original_spacing = img.GetSpacing()[::-1]
        data = sitk.GetArrayFromImage(img)  # (Z, Y, X)

        seg = None
        if mask_path is not None:
            mask_img = sitk.ReadImage(mask_path)
            mask_spacing = mask_img.GetSpacing()[::-1]
            seg = sitk.GetArrayFromImage(mask_img)  # (Z, Y, X)
            # ---- Co-registration check ----
            # The mask must live on the same voxel grid as its image, otherwise
            # the image-derived crop bbox would not apply to it.
            if seg.shape != data.shape:
                print(f"[skip] {case_id}: mask shape {seg.shape} != image shape "
                      f"{data.shape}; mask must be co-registered to its image")
                return None
            if not all(abs(a - b) < 1e-3 for a, b in zip(mask_spacing, original_spacing)):
                print(f"[skip] {case_id}: mask spacing {mask_spacing} != image "
                      f"spacing {original_spacing}; mask must be co-registered")
                return None

        # Add channel dim for compatibility with cropping/resampling utilities
        data = data[np.newaxis, ...].astype(np.float32, copy=False)  # (1, Z, Y, X)
        if seg is not None:
            seg = seg[np.newaxis, ...].astype(np.float32, copy=False)  # (1, Z, Y, X)

        # Sanity check
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print(f"[skip] {case_id}: NaN/Inf in input")
            return None

        # ---- 2. Resample to target spacing ----
        if not skip_resample:
            target = list(target_spacing)
            # Skip the resample if already at target spacing (saves time, avoids
            # unnecessary interpolation noise).
            already_target = all(
                abs(o - t) < 1e-3 for o, t in zip(original_spacing, target)
            )
            if not already_target:
                data = resample_data_or_seg_to_spacing(
                    data,
                    original_spacing,
                    target,
                    is_seg=False,
                    order=RESAMPLING_ORDER,
                )
                if seg is not None:
                    seg = resample_data_or_seg_to_spacing(
                        seg,
                        original_spacing,
                        target,
                        is_seg=True,
                        order=0,
                    )

        # ---- 3. Crop to non-zero bounding box ----
        # For CT this typically only trims explicitly zero-padded edges. Real
        # tissue has nonzero HU values so it survives the crop.
        data, _seg, bbox = crop_to_nonzero(data, seg=None)
        if seg is not None:
            # Apply the *image's* bbox to the mask directly. We deliberately do
            # NOT pass seg into crop_to_nonzero — that path writes -1 into the
            # background, which would pollute the raw label channel.
            slicer = (slice(None),) + tuple(bounding_box_to_slice(bbox))
            seg = seg[slicer]

        # ---- 4. CT-normalize using dataset-wide stats (image only) ----
        normalizer = CTNormalization()
        normalizer.intensityproperties = intensity_properties
        data = normalizer.run(data)

        # ---- 5. Save as Blosc2 in the per-subject sub-directory ----
        # Layout: <out_dir>/<case_id>.b2nd (+ <case_id>_mask.b2nd)
        _save_b2nd(data, os.path.join(out_dir, case_id))
        if seg is not None:
            _save_b2nd(seg.astype(np.int8, copy=False),
                       os.path.join(out_dir, case_id + MASK_SUFFIX))

        return case_id

    except Exception as e:
        print(f"[error] {case_id}: {e}")
        return None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in-dir", required=True, type=Path, nargs="+",
                        help="One or more directories containing raw .nii.gz CT images.")
    parser.add_argument("--out-root", required=True, type=Path,
                        help="Output directory for this dataset. The script writes to "
                             "<out-root>/preprocessed_b2nd/<id>/<id>.b2nd and "
                             "<out-root>/preprocessing.json")
    parser.add_argument("--mask-dir", type=Path, default=None,
                        help="Optional directory of co-registered masks. For each "
                             "image <id>.nii.gz the matching mask must be named "
                             "<id>_mask.nii.gz. When set, the mask is saved as a "
                             "second channel (<id>_mask.b2nd) and every image must "
                             "have a matching mask (missing masks are an error).")
    parser.add_argument("--target-spacing", type=float, nargs=3,
                        default=None, metavar=("Z", "Y", "X"),
                        help="Target spacing in mm as three floats: Z Y X. "
                             "If omitted, the per-axis median spacing across "
                             "all input images is computed and used.")
    parser.add_argument("--skip-resample", action="store_true",
                        help="Skip resampling entirely (use the data as-is).")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel processes. Default: 8")

    # Optional pre-supplied stats (skip the dataset scan)
    parser.add_argument("--stats-mean", type=float, default=None)
    parser.add_argument("--stats-std", type=float, default=None)
    parser.add_argument("--stats-pct-00-5", type=float, default=None)
    parser.add_argument("--stats-pct-99-5", type=float, default=None)

    args = parser.parse_args()

    for d in args.in_dir:
        if not d.is_dir():
            raise SystemExit(f"--in-dir does not exist: {d}")

    image_paths = []
    for d in args.in_dir:
        image_paths.extend(sorted(str(p) for p in d.glob("*.nii.gz")))
    if not image_paths:
        raise SystemExit(f"No .nii.gz files found in any of: {args.in_dir}")

    print(f"Found {len(image_paths)} CT images across {len(args.in_dir)} directory(ies).")

    # ---- Resolve + validate masks (optional) ---- #
    use_mask = args.mask_dir is not None
    mask_for_image = {}  # image_path -> mask_path
    if use_mask:
        if not args.mask_dir.is_dir():
            raise SystemExit(f"--mask-dir does not exist: {args.mask_dir}")
        missing = []
        for img_path in image_paths:
            case_id = _strip_nifti_ext(Path(img_path).name)
            mask_path = args.mask_dir / f"{case_id}{MASK_SUFFIX}.nii.gz"
            if not mask_path.is_file():
                alt = args.mask_dir / f"{case_id}{MASK_SUFFIX}.nii"
                mask_path = alt if alt.is_file() else mask_path
            if not mask_path.is_file():
                missing.append(case_id)
            else:
                mask_for_image[img_path] = str(mask_path)
        if missing:
            listed = "\n  ".join(missing)
            raise SystemExit(
                f"--mask-dir set but {len(missing)} image(s) have no matching "
                f"'<id>{MASK_SUFFIX}.nii[.gz]' in {args.mask_dir}:\n  {listed}"
            )
        print(f"[note] matched masks for all {len(image_paths)} images in {args.mask_dir}")

    # ---- Resolve target spacing ---- #
    if args.skip_resample:
        print("[note] --skip-resample set. Volumes will be saved at their native spacing.")
        target_spacing = None
    elif args.target_spacing is not None:
        target_spacing = tuple(args.target_spacing)
        print(f"[note] user-specified target spacing (Z Y X): {target_spacing} mm")
    else:
        print(f"\nComputing median spacing from {len(image_paths)} image headers...")
        if args.num_workers > 1:
            with Pool(processes=args.num_workers) as pool:
                spacings = list(tqdm(
                    pool.imap_unordered(read_image_spacing, image_paths),
                    total=len(image_paths),
                ))
        else:
            spacings = [read_image_spacing(p) for p in tqdm(image_paths)]
        spacings_arr = np.asarray(spacings, dtype=np.float64)  # (N, 3) in (Z, Y, X)
        target_spacing = tuple(np.median(spacings_arr, axis=0).tolist())
        print(f"[note] median target spacing (Z Y X): {target_spacing} mm")

    # ---- Resolve intensity stats ---- #
    have_args_stats = all(
        v is not None for v in [args.stats_mean, args.stats_std, args.stats_pct_00_5, args.stats_pct_99_5]
    )

    if have_args_stats:
        intensity_properties = {
            "mean": args.stats_mean,
            "std": args.stats_std,
            "percentile_00_5": args.stats_pct_00_5,
            "percentile_99_5": args.stats_pct_99_5,
        }
        print(f"Using user-supplied stats: {intensity_properties}")
    else:
        intensity_properties = compute_dataset_stats(image_paths, num_workers=args.num_workers)
        print("Computed stats:")
        for k, v in intensity_properties.items():
            print(f"  {k:>20s}: {v:.4f}")

    # ---- Build per-case output paths and dispatch ---- #
    dataset_dir = args.out_root
    b2nd_dir = dataset_dir / "preprocessed_b2nd"
    b2nd_dir.mkdir(parents=True, exist_ok=True)
    if use_mask:
        print(f"\nWriting outputs to {b2nd_dir}/<id>/<id>.b2nd (+ <id>_mask.b2nd)")
    else:
        print(f"\nWriting outputs to {b2nd_dir}/<id>/<id>.b2nd")

    # ---- Write preprocessing.json sidecar ---- #
    # Records every knob predict_external.py needs to replay this preprocessing
    # on raw NIfTI files at inference time.
    sidecar = {
        "modality": "ct",
        "target_spacing": list(target_spacing) if target_spacing is not None else None,
        "skip_resample": bool(args.skip_resample),
        "resampling_order": RESAMPLING_ORDER,
        "foreground_hu_threshold": CT_FOREGROUND_HU_THRESHOLD,
        "has_masks": use_mask,
        "stats": {
            "mean": intensity_properties["mean"],
            "std": intensity_properties["std"],
            "percentile_00_5": intensity_properties["percentile_00_5"],
            "percentile_99_5": intensity_properties["percentile_99_5"],
        },
        "preprocess_script": "preprocess_ct.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    sidecar_path = dataset_dir / "preprocessing.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[note] wrote sidecar {sidecar_path}")

    job_args = []
    for img_path in image_paths:
        case_id = _strip_nifti_ext(Path(img_path).name)
        out_dir = str(b2nd_dir / case_id)
        job_args.append((
            img_path,
            mask_for_image.get(img_path),  # None when --mask-dir not set
            out_dir,
            case_id,
            intensity_properties,
            target_spacing,
            args.skip_resample,
        ))

    print(f"\nProcessing {len(job_args)} cases with {args.num_workers} workers...")
    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_one_case, job_args),
                                total=len(job_args)))
    else:
        results = [process_one_case(j) for j in tqdm(job_args)]

    n_ok = sum(1 for r in results if r is not None)
    n_fail = len(results) - n_ok
    print(f"\nDone. {n_ok} succeeded, {n_fail} failed.")
    if n_fail:
        print("Failed cases were printed above with [error] or [skip] tags.")


if __name__ == "__main__":
    main()