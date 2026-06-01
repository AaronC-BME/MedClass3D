"""
Run a trained model on raw NIfTI files outside of the training CSV.

Reads `<run_dir>/Configs/preprocessing.json` (snapshotted at training time from
the dataset's `preprocessing.json`) so the exact same preprocessing pipeline
that the model was trained on is replayed on the new inputs. Then loads the
matching checkpoint and writes one prediction per input image.

Pipeline per input `.nii.gz`:
  1. Load the NIfTI.
  2. Resample to the training-time target spacing (skipped if --skip-resample
     was used at training time, or if the input is already at target).
  3. Crop to the non-zero bounding box.
  4. Normalize:
       - CT  : clip to training-time [p0.5, p99.5], z-score with training-time mean/std.
       - MRI : per-case z-score on the foreground (voxels > 0).
  5. Save as a temporary `.b2nd`.
After all inputs are preprocessed, the temp dir is fed to the model's
`test_transforms` (center crop to patch size) and `trainer.predict()` runs.
The temp `.b2nd` files are deleted at the end unless --keep-preprocessed is set.

Output:
    <pred_dir>/predictions.csv     -- columns: PatientID, Pred, Prob_0, Prob_1, ...

Usage:
    python scripts/predict_external.py \\
        --run-dir /path/to/<output_dir>/<dataset>/<run_name> \\
        --input-dir /path/to/raw/nifti \\
        --fold 0
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Allow the script to import the medclass3d package even if launched
# directly without the editable install on PYTHONPATH (matches the convention
# in preprocess_ct.py / preprocess_mri.py).
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
)

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

from medclass3d.data.datamodules import Class_Data
from medclass3d.data.preprocessing.blosc_helper import (
    comp_blosc2_params,
    save_case,
)
from medclass3d.data.preprocessing.cropping import crop_to_nonzero
from medclass3d.data.preprocessing.default_resampling import (
    resample_data_or_seg_to_spacing,
)
from medclass3d.data.preprocessing.normalization import (
    CTNormalization,
    ZScoreNormalization,
)
from medclass3d.utils.parsing import make_omegaconf_resolvers

# Mask file naming, kept in sync with preprocess_ct.py / preprocess_mri.py:
# image <id>.nii.gz -> mask <id>_mask.nii.gz -> preprocessed <id>_mask.b2nd
MASK_SUFFIX = "_mask"


def _strip_nifti_ext(name: str) -> str:
    """Strip a trailing .nii.gz or .nii from a filename, returning the case id."""
    if name.endswith(".nii.gz"):
        return name[:-len(".nii.gz")]
    if name.endswith(".nii"):
        return name[:-len(".nii")]
    return name


# --------------------------------------------------------------------------- #
# Checkpoint selection (kept in sync with predict_test.py)
# --------------------------------------------------------------------------- #
def _select_best_ckpt(ckp_paths, prefer_best=True):
    ckp_paths = [Path(p) for p in ckp_paths]
    last = [p for p in ckp_paths if p.name == "last.ckpt"]
    not_last = [p for p in ckp_paths if p.name != "last.ckpt"]

    if not prefer_best:
        return last[0] if last else (ckp_paths[0] if ckp_paths else None)

    def _parse_acc(p):
        try:
            tag = str(p).split("Val_acc=")[1]
            return float(tag.split(".ckpt")[0])
        except (IndexError, ValueError):
            return float("-inf")

    if not_last:
        not_last.sort(key=_parse_acc, reverse=True)
        if _parse_acc(not_last[0]) != float("-inf"):
            return not_last[0]

    return last[0] if last else (ckp_paths[0] if ckp_paths else None)


# --------------------------------------------------------------------------- #
# Per-case preprocessing — mirrors preprocess_ct.py / preprocess_mri.py
# --------------------------------------------------------------------------- #
def _preprocess_one(
    image_path: str,
    modality: str,
    target_spacing: Optional[Tuple[float, float, float]],
    skip_resample: bool,
    resampling_order: int,
    stats: Optional[dict] = None,
    mask_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return ``(image, mask)`` ready to be saved as .b2nd.

    ``image`` is a (1, Z, Y, X) float32 normalized volume. ``mask`` is a
    (1, Z, Y, X) int8 array co-registered + cropped to match the image, or None
    when ``mask_path`` is not given. The pipeline mirrors preprocess_ct.py /
    preprocess_mri.py so the model sees inputs identical to training.
    """
    img = sitk.ReadImage(image_path)
    original_spacing = img.GetSpacing()[::-1]
    data = sitk.GetArrayFromImage(img)

    seg = None
    if mask_path is not None:
        mask_img = sitk.ReadImage(mask_path)
        mask_spacing = mask_img.GetSpacing()[::-1]
        seg = sitk.GetArrayFromImage(mask_img)
        if seg.shape != data.shape:
            raise ValueError(
                f"mask shape {seg.shape} != image shape {data.shape}; "
                "mask must be co-registered to its image"
            )
        if not all(abs(a - b) < 1e-3 for a, b in zip(mask_spacing, original_spacing)):
            raise ValueError(
                f"mask spacing {mask_spacing} != image spacing {original_spacing}; "
                "mask must be co-registered to its image"
            )

    data = data[np.newaxis, ...].astype(np.float32, copy=False)
    if seg is not None:
        seg = seg[np.newaxis, ...].astype(np.float32, copy=False)

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError(f"NaN/Inf in input {image_path}")

    # 1. Resample
    if not skip_resample and target_spacing is not None:
        target = list(target_spacing)
        already_target = all(
            abs(o - t) < 1e-3 for o, t in zip(original_spacing, target)
        )
        if not already_target:
            data = resample_data_or_seg_to_spacing(
                data,
                original_spacing,
                target,
                is_seg=False,
                order=resampling_order,
            )
            if seg is not None:
                seg = resample_data_or_seg_to_spacing(
                    seg, original_spacing, target, is_seg=True, order=0,
                )

    # 2. Crop to non-zero bounding box (mask follows the image's bbox)
    data, _seg, bbox = crop_to_nonzero(data, seg=None)
    if seg is not None:
        slicer = (slice(None),) + tuple(bounding_box_to_slice(bbox))
        seg = seg[slicer]

    # 3. Normalize per modality (image only; mask stays raw)
    if modality == "ct":
        if stats is None:
            raise ValueError("CT modality requires stats in preprocessing.json")
        normalizer = CTNormalization()
        normalizer.intensityproperties = stats
        data = normalizer.run(data)
    elif modality == "mri":
        foreground_mask = data[0] > 0
        if not foreground_mask.any():
            raise ValueError(f"empty foreground after cropping for {image_path}")
        normalizer = ZScoreNormalization()
        data = normalizer.run(data, seg=foreground_mask[np.newaxis, ...])
    else:
        raise ValueError(
            f"Unknown modality in preprocessing.json: {modality!r}. "
            "Expected 'ct' or 'mri'."
        )

    if seg is not None:
        seg = seg.astype(np.int8, copy=False)
    return data, seg


def _save_b2nd(data: np.ndarray, out_path_no_ext: str) -> None:
    os.makedirs(os.path.dirname(out_path_no_ext), exist_ok=True)
    block_size, chunk_size = comp_blosc2_params(
        data.shape, (160, 160, 160), data.itemsize,
    )
    save_case(data, out_path_no_ext, chunks=chunk_size, blocks=block_size)


def _preprocess_directory(
    input_dir: Path,
    sidecar: dict,
    out_dir: Path,
    mask_dir: Optional[Path] = None,
) -> list:
    """Preprocess every .nii.gz under `input_dir` into the per-subject layout
    ``out_dir/<id>/<id>.b2nd`` (+ ``<id>_mask.b2nd`` when `mask_dir` is given),
    returning the list of successfully-processed image IDs.

    When `mask_dir` is set, every image must have a matching
    ``<id>_mask.nii[.gz]`` mask (missing masks are an error)."""
    image_paths = sorted(str(p) for p in input_dir.glob("*.nii.gz"))
    if not image_paths:
        raise SystemExit(f"No .nii.gz files found in {input_dir}")

    modality = sidecar["modality"]
    target_spacing = sidecar.get("target_spacing")
    skip_resample = bool(sidecar.get("skip_resample", False))
    resampling_order = int(sidecar.get("resampling_order", 3))
    stats = sidecar.get("stats")  # only present for CT

    if target_spacing is not None:
        target_spacing = tuple(target_spacing)

    # Resolve + validate masks up front so a missing mask fails loudly.
    mask_for_image = {}
    if mask_dir is not None:
        missing = []
        for img_path in image_paths:
            case_id = _strip_nifti_ext(Path(img_path).name)
            mp = mask_dir / f"{case_id}{MASK_SUFFIX}.nii.gz"
            if not mp.is_file():
                alt = mask_dir / f"{case_id}{MASK_SUFFIX}.nii"
                mp = alt if alt.is_file() else mp
            if not mp.is_file():
                missing.append(case_id)
            else:
                mask_for_image[img_path] = str(mp)
        if missing:
            listed = "\n  ".join(missing)
            raise SystemExit(
                f"--mask-dir set but {len(missing)} image(s) have no matching "
                f"'<id>{MASK_SUFFIX}.nii[.gz]' in {mask_dir}:\n  {listed}"
            )

    print(
        f"Preprocessing {len(image_paths)} {modality.upper()} image(s) using "
        f"target_spacing={target_spacing}, skip_resample={skip_resample}, "
        f"masks={mask_dir is not None}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ok_ids = []
    for img_path in tqdm(image_paths):
        case_id = _strip_nifti_ext(Path(img_path).name)
        try:
            data, seg = _preprocess_one(
                image_path=img_path,
                modality=modality,
                target_spacing=target_spacing,
                skip_resample=skip_resample,
                resampling_order=resampling_order,
                stats=stats,
                mask_path=mask_for_image.get(img_path),
            )
            _save_b2nd(data, str(out_dir / case_id / case_id))
            if seg is not None:
                _save_b2nd(seg, str(out_dir / case_id / f"{case_id}{MASK_SUFFIX}"))
            ok_ids.append(case_id)
        except Exception as e:
            print(f"[skip] {case_id}: {e}")

    return ok_ids


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Training-run directory containing Configs/{config.yaml, preprocessing.json}.")
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory of raw .nii.gz images to predict on.")
    parser.add_argument("--mask-dir", type=Path, default=None,
                        help="Directory of co-registered masks (<id>_mask.nii.gz). "
                             "REQUIRED when the run was trained with masks "
                             "(preprocessing.json has_masks=true); ignored otherwise.")
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold's checkpoint to load. Default: 0")
    parser.add_argument("--pred-dir", type=Path, default=None,
                        help="Where to write predictions.csv. Default: <run-dir>/predictions_external/")
    parser.add_argument("--keep-preprocessed", action="store_true",
                        help="Keep the intermediate .b2nd files instead of deleting them at the end.")
    parser.add_argument("--metrics", nargs="+", default=["acc", "balanced_acc", "f1", "auroc"],
                        help="Metric names forwarded to the model. Default: acc balanced_acc f1 auroc")
    parser.add_argument("--prefer-last", action="store_true",
                        help="Use last.ckpt instead of the best-Val_acc checkpoint.")
    args = parser.parse_args()

    make_omegaconf_resolvers()

    # ---- Resolve run dir + load snapshots ---- #
    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir does not exist: {run_dir}")

    if not args.input_dir.is_dir():
        raise SystemExit(f"--input-dir does not exist: {args.input_dir}")

    training_config_path = run_dir / "Configs" / "config.yaml"
    if not training_config_path.is_file():
        raise SystemExit(f"Training config not found at {training_config_path}")

    sidecar_path = run_dir / "Configs" / "preprocessing.json"
    if not sidecar_path.is_file():
        raise SystemExit(
            f"preprocessing.json not found at {sidecar_path}.\n"
            "This run was trained before the preprocessing-sidecar mechanism existed.\n"
            "To use predict_external.py on this run, re-run the preprocess script\n"
            "to produce a preprocessing.json, then re-train (or manually copy the\n"
            "sidecar into the run's Configs/ directory)."
        )

    with open(sidecar_path) as f:
        sidecar = json.load(f)
    print(f"Using preprocessing sidecar: {sidecar_path}")
    print(f"  modality       = {sidecar.get('modality')}")
    print(f"  target_spacing = {sidecar.get('target_spacing')}")
    if sidecar.get("modality") == "ct":
        print(f"  stats          = {sidecar.get('stats')}")

    # ---- Mask handling: must match how the run was trained ---- #
    has_masks = bool(sidecar.get("has_masks", False))
    print(f"  has_masks      = {has_masks}")
    if has_masks:
        if args.mask_dir is None:
            raise SystemExit(
                "This run was trained with masks (preprocessing.json has_masks=true), "
                "so --mask-dir is required. Provide a directory containing a "
                f"'<id>{MASK_SUFFIX}.nii[.gz]' for every input image."
            )
        if not args.mask_dir.is_dir():
            raise SystemExit(f"--mask-dir does not exist: {args.mask_dir}")
        mask_dir = args.mask_dir
    else:
        if args.mask_dir is not None:
            print("[warn] --mask-dir given but this run was trained without masks; ignoring it.")
        mask_dir = None

    fold_id = str(args.fold)
    ckp_dir = run_dir / "folds" / fold_id
    ckp_list = list(ckp_dir.glob("*.ckpt"))
    if not ckp_list:
        raise SystemExit(f"No checkpoints found under {ckp_dir}")
    ckp_path = _select_best_ckpt(ckp_list, prefer_best=not args.prefer_last)
    print(f"[fold {fold_id}] using checkpoint: {ckp_path}")

    pred_dir = args.pred_dir if args.pred_dir else run_dir / "predictions_external"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ---- Preprocess external NIfTI -> temp .b2nd ---- #
    if args.keep_preprocessed:
        b2nd_dir = pred_dir / "external_preprocessed_b2nd"
        b2nd_dir.mkdir(parents=True, exist_ok=True)
        tmp_root = None
    else:
        tmp_root = tempfile.mkdtemp(prefix="medreg_external_", dir=str(pred_dir))
        b2nd_dir = Path(tmp_root) / "preprocessed_b2nd"

    try:
        ok_ids = _preprocess_directory(args.input_dir, sidecar, b2nd_dir, mask_dir=mask_dir)
        if not ok_ids:
            raise SystemExit("No images preprocessed successfully.")
        print(f"[preprocess] {len(ok_ids)} image(s) written to {b2nd_dir}")

        # ---- Build model + trainer from the snapshot ---- #
        used_training_cfg = OmegaConf.load(training_config_path)
        used_training_cfg.trainer.pop("logger", None)
        used_training_cfg.trainer.pop("callbacks", None)
        used_training_cfg.model.metrics = list(args.metrics)
        used_training_cfg.trainer.devices = 1
        used_training_cfg.trainer.strategy = "auto"
        used_training_cfg.trainer.sync_batchnorm = False
        used_training_cfg.data.module.fold = int(fold_id)

        model = instantiate(used_training_cfg.model)
        state = torch.load(ckp_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        model.eval()

        trainer = instantiate(used_training_cfg.trainer)

        # ---- Build a synthetic manifest CSV pointing at the temp b2nd dir ---- #
        label_column = used_training_cfg.data.module.get("label_column", "label")
        manifest_path = pred_dir / "_external_manifest.csv"
        pd.DataFrame({
            "image_name": ok_ids,
            "split": "test",
            "fold": 0,
            label_column: 0.0,
        }).to_csv(manifest_path, index=False)
        print(f"[predict] manifest with {len(ok_ids)} cases at {manifest_path}")

        # ---- Build dataset using the snapshot's test_transforms ---- #
        test_transforms = instantiate(used_training_cfg.data.module.test_transforms)
        batch_size = int(used_training_cfg.data.module.batch_size)
        num_workers = int(used_training_cfg.data.module.get("num_workers", 4))

        predict_ds = Class_Data(
            img_dir=str(b2nd_dir),
            csv_file=str(manifest_path),
            split="test",
            fold=0,
            label_column=label_column,
            transform=test_transforms,
            train=False,
            use_mask=has_masks,
        )
        loader = DataLoader(
            predict_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # ---- Inference ---- #
        predictions = trainer.predict(model, dataloaders=loader)
        _, y_hats = zip(*predictions)

        # Model emits [B, num_classes] logits; convert to probs + argmax.
        logits = torch.cat([yh.detach().cpu() for yh in y_hats], dim=0)
        subtask = used_training_cfg.model.get("subtask", "multiclass")
        if subtask == "multilabel":
            probs = torch.sigmoid(logits)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
        preds = probs.argmax(dim=1)

        if len(ok_ids) != len(preds):
            raise RuntimeError(
                f"Length mismatch: {len(ok_ids)} ids vs {len(preds)} preds."
            )

        # ---- Output CSV ---- #
        out_path = pred_dir / "predictions.csv"
        df_data = {
            "PatientID": ok_ids,
            "Pred": preds.numpy(),
        }
        for i in range(probs.shape[-1]):
            df_data[f"Prob_{i}"] = probs[:, i].numpy()
        pd.DataFrame(df_data).to_csv(out_path, index=False)
        print(f"[done] wrote {len(ok_ids)} predictions to {out_path}")

    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)
            print(f"[cleanup] removed temp dir {tmp_root}")


if __name__ == "__main__":
    main()
