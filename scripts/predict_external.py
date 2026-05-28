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
    <pred_dir>/predictions.csv     -- columns: PatientID, Prediction

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


# --------------------------------------------------------------------------- #
# Checkpoint selection (kept in sync with predict_test.py)
# --------------------------------------------------------------------------- #
def _select_best_ckpt(ckp_paths, prefer_best=True):
    ckp_paths = [Path(p) for p in ckp_paths]
    last = [p for p in ckp_paths if p.name == "last.ckpt"]
    not_last = [p for p in ckp_paths if p.name != "last.ckpt"]

    if not prefer_best:
        return last[0] if last else (ckp_paths[0] if ckp_paths else None)

    def _parse_mae(p):
        try:
            tag = str(p).split("Val_mae=")[1]
            return float(tag.split(".ckpt")[0])
        except (IndexError, ValueError):
            return float("inf")

    if not_last:
        not_last.sort(key=_parse_mae)
        if _parse_mae(not_last[0]) != float("inf"):
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
) -> np.ndarray:
    """Return a (1, Z, Y, X) float32 normalized volume ready to be saved as .b2nd."""
    img = sitk.ReadImage(image_path)
    original_spacing = img.GetSpacing()[::-1]
    data = sitk.GetArrayFromImage(img)
    data = data[np.newaxis, ...].astype(np.float32, copy=False)

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

    # 2. Crop to non-zero bounding box
    data, _seg, _bbox = crop_to_nonzero(data, seg=None)

    # 3. Normalize per modality
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

    return data


def _save_b2nd(data: np.ndarray, out_path_no_ext: str) -> None:
    block_size, chunk_size = comp_blosc2_params(
        data.shape, (160, 160, 160), data.itemsize,
    )
    save_case(data, out_path_no_ext, chunks=chunk_size, blocks=block_size)


def _preprocess_directory(
    input_dir: Path,
    sidecar: dict,
    out_dir: Path,
) -> list:
    """Preprocess every .nii.gz under `input_dir`, write .b2nd files to `out_dir`,
    return the list of successfully-processed image IDs (without extension)."""
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

    print(
        f"Preprocessing {len(image_paths)} {modality.upper()} image(s) using "
        f"target_spacing={target_spacing}, skip_resample={skip_resample}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ok_ids = []
    for img_path in tqdm(image_paths):
        case_id = Path(img_path).name
        if case_id.endswith(".nii.gz"):
            case_id = case_id[:-len(".nii.gz")]
        elif case_id.endswith(".nii"):
            case_id = case_id[:-len(".nii")]

        try:
            data = _preprocess_one(
                image_path=img_path,
                modality=modality,
                target_spacing=target_spacing,
                skip_resample=skip_resample,
                resampling_order=resampling_order,
                stats=stats,
            )
            _save_b2nd(data, str(out_dir / case_id))
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
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold's checkpoint to load. Default: 0")
    parser.add_argument("--pred-dir", type=Path, default=None,
                        help="Where to write predictions.csv. Default: <run-dir>/predictions_external/")
    parser.add_argument("--keep-preprocessed", action="store_true",
                        help="Keep the intermediate .b2nd files instead of deleting them at the end.")
    parser.add_argument("--metrics", nargs="+", default=["mae", "mse"],
                        help="Metric names forwarded to the model. Default: mae mse")
    parser.add_argument("--prefer-last", action="store_true",
                        help="Use last.ckpt instead of the best-Val_MAE checkpoint.")
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
        ok_ids = _preprocess_directory(args.input_dir, sidecar, b2nd_dir)
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

        task = used_training_cfg.model.task
        if task == "Ordinal_Regression":
            # Model's forward returns (logits, probs) for ordinal heads.
            probas = torch.cat([yh[1].detach().cpu() for yh in y_hats], dim=0)
            preds = (probas > 0.5).sum(dim=1).float()
        else:
            # Plain regression — y_hat is a single tensor of scalars.
            preds = torch.cat(
                [yh.detach().cpu().view(-1) for yh in y_hats], dim=0,
            ).float()

        if len(ok_ids) != len(preds):
            raise RuntimeError(
                f"Length mismatch: {len(ok_ids)} ids vs {len(preds)} preds."
            )

        # ---- Output CSV ---- #
        out_path = pred_dir / "predictions.csv"
        pd.DataFrame({
            "PatientID": ok_ids,
            "Prediction": preds.numpy(),
        }).to_csv(out_path, index=False)
        print(f"[done] wrote {len(ok_ids)} predictions to {out_path}")

    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)
            print(f"[cleanup] removed temp dir {tmp_root}")


if __name__ == "__main__":
    main()
