"""Run 3D Grad-CAM / Layer-CAM on an external cohort of raw NIfTI images, per fold.

Preprocesses ``--input-dir/*.nii.gz`` once using the run's
``Configs/preprocessing.json`` sidecar (reusing predict_external.py's pipeline),
then for each fold loads the best checkpoint and writes per-image overlays + npz
arrays under ``<out_dir>/<method-subdir>/fold_<k>/``. External cases have no
ground-truth label.

Usage:
    python scripts/gradcam_external.py --run-dir /path/to/run \\
        --input-dir /path/to/nifti --mask-dir /path/to/masks --methods layercam
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch

# predict_external lives alongside this script; reuse its NIfTI -> .b2nd pipeline.
from predict_external import _preprocess_directory  # noqa: E402

from medclass3d.utils.parsing import make_omegaconf_resolvers
from medclass3d.gradcam import (
    load_fold_model, class_names_for, build_external_patients,
    run_gradcam_for_patients, reorganize_to_per_image_subdirs, derive_method_subdir,
)
from medclass3d.gradcam.runner import default_z_slices


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--mask-dir", type=Path, default=None,
                   help="Required when the run was trained with masks.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <run-dir>/gradcam/external")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--methods", type=str, nargs="+", default=["layercam"])
    p.add_argument("--target-class", type=int, default=-1)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--occ-mask-size", type=int, default=8)
    p.add_argument("--npz-only", action="store_true")
    p.add_argument("--manuscript-mode", action="store_true")
    p.add_argument("--modality-names", type=str, nargs="+", default=None)
    p.add_argument("--per-modality-occlusion", action="store_true")
    p.add_argument("--no-side-by-side", action="store_true")
    p.add_argument("--keep-preprocessed", action="store_true")
    p.add_argument("--prefer-last", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    make_omegaconf_resolvers()
    device = torch.device(args.device if args.device else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out_base = (args.out_dir or args.run_dir / "gradcam" / "external") / derive_method_subdir(args.methods)
    out_base.mkdir(parents=True, exist_ok=True)

    sidecar_path = args.run_dir / "Configs" / "preprocessing.json"
    if not sidecar_path.is_file():
        raise SystemExit(f"preprocessing.json not found at {sidecar_path}")
    sidecar = json.load(open(sidecar_path))
    has_masks = bool(sidecar.get("has_masks", False))
    if has_masks and args.mask_dir is None:
        raise SystemExit("Run was trained with masks; --mask-dir is required.")
    print(f"Run dir: {args.run_dir}\nOutput : {out_base}\nDevice : {device}\n"
          f"modality={sidecar.get('modality')} has_masks={has_masks}")

    # ---- Preprocess the cohort once (fold-independent) ----
    if args.keep_preprocessed:
        b2nd_dir = out_base / "external_preprocessed_b2nd"
        b2nd_dir.mkdir(parents=True, exist_ok=True)
        tmp_root = None
    else:
        tmp_root = tempfile.mkdtemp(prefix="gradcam_ext_", dir=str(out_base))
        b2nd_dir = Path(tmp_root) / "preprocessed_b2nd"
    try:
        ok_ids = _preprocess_directory(args.input_dir, sidecar, b2nd_dir,
                                       mask_dir=args.mask_dir if has_masks else None)
        if not ok_ids:
            raise SystemExit("No external images preprocessed successfully.")
        print(f"[preprocess] {len(ok_ids)} case(s) -> {b2nd_dir}")

        common = dict(methods=args.methods, target_class=args.target_class,
                      occ_mask_size=args.occ_mask_size, npz_only=args.npz_only,
                      manuscript_mode=args.manuscript_mode, modality_names=args.modality_names,
                      per_modality_occlusion=args.per_modality_occlusion,
                      side_by_side=not args.no_side_by_side)

        for fold in args.folds:
            print(f"\n{'#'*80}\n# FOLD {fold}\n{'#'*80}")
            try:
                model, cfg, best = load_fold_model(args.run_dir, fold, device,
                                                   prefer_last=args.prefer_last)
                if model is None:
                    continue
                class_names = class_names_for(int(cfg.data.num_classes))
                patients = build_external_patients(cfg, b2nd_dir, has_masks, ok_ids,
                                                   model, device, max_cases=args.max_cases)
                if not patients:
                    del model; torch.cuda.empty_cache(); continue
                z = default_z_slices(int(patients[0]["input"].shape[1]))
                fold_dir = out_base / f"fold_{fold}"
                run_gradcam_for_patients(model, patients, fold_dir, class_names=class_names,
                                         z_slices=z, label=f"fold {fold} | external", **common)
                reorganize_to_per_image_subdirs(fold_dir, [p["patient_id"] for p in patients])
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"[fold {fold}] FAILED: {e}")
                traceback.print_exc()
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\nDone. Output tree under {out_base}")


if __name__ == "__main__":
    main()
