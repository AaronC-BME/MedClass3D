"""Run 3D Grad-CAM / Layer-CAM on a trained run's held-out test split, per fold.

Loads each fold's best checkpoint from ``<run_dir>/folds/<k>``, assembles the test
split via the training DataModule, and writes per-image overlays + npz arrays under
``<out_dir>/<method-subdir>/fold_<k>/``. With ``--per-class-confusion`` cases are
bucketed into ``gt_<G>_pred_<P>/`` subdirs.

Usage:
    python scripts/gradcam_testset.py --run-dir /path/to/run --methods layercam
    python scripts/gradcam_testset.py --run-dir /path/to/run \\
        --per-class-confusion --max-per-cell 3
"""
import argparse
from pathlib import Path

import torch

from medclass3d.utils.parsing import make_omegaconf_resolvers
from medclass3d.gradcam import (
    load_fold_model, class_names_for, build_testset_patients,
    bucket_patients_by_confusion_cell, print_confusion_bucket_summary,
    run_gradcam_for_patients, reorganize_to_per_image_subdirs, derive_method_subdir,
)
from medclass3d.gradcam.runner import default_z_slices


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <run-dir>/gradcam/testset")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--methods", type=str, nargs="+", default=["layercam"])
    p.add_argument("--target-class", type=int, default=-1)
    p.add_argument("--only-correct", action="store_true")
    p.add_argument("--only-incorrect", action="store_true")
    p.add_argument("--per-class-confusion", action="store_true")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--max-per-cell", type=int, default=None)
    p.add_argument("--occ-mask-size", type=int, default=8)
    p.add_argument("--npz-only", action="store_true")
    p.add_argument("--manuscript-mode", action="store_true")
    p.add_argument("--modality-names", type=str, nargs="+", default=None)
    p.add_argument("--per-modality-occlusion", action="store_true")
    p.add_argument("--no-side-by-side", action="store_true")
    p.add_argument("--prefer-last", action="store_true")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    if args.only_correct and args.only_incorrect:
        p.error("--only-correct and --only-incorrect are mutually exclusive.")
    if args.per_class_confusion and (args.only_correct or args.only_incorrect):
        p.error("--per-class-confusion supersedes --only-correct/--only-incorrect.")
    if args.max_per_cell is not None and not args.per_class_confusion:
        p.error("--max-per-cell requires --per-class-confusion.")
    return args


def main():
    args = parse_args()
    make_omegaconf_resolvers()
    device = torch.device(args.device if args.device else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out_base = (args.out_dir or args.run_dir / "gradcam" / "testset") / derive_method_subdir(args.methods)
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {args.run_dir}\nOutput : {out_base}\nDevice : {device}")

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
            num_classes = int(cfg.data.num_classes)
            class_names = class_names_for(num_classes)
            patients = build_testset_patients(
                cfg, model, device, max_cases=args.max_cases,
                only_correct=args.only_correct, only_incorrect=args.only_incorrect,
                per_class_confusion=args.per_class_confusion)
            if not patients:
                del model; torch.cuda.empty_cache(); continue

            z = default_z_slices(int(patients[0]["input"].shape[1]))
            fold_dir = out_base / f"fold_{fold}"

            if args.per_class_confusion:
                buckets = bucket_patients_by_confusion_cell(patients, class_names, args.max_per_cell)
                print_confusion_bucket_summary(buckets, class_names, num_classes)
                for (gt, pred), cell in sorted(buckets.items()):
                    if not cell:
                        continue
                    cell_dir = fold_dir / f"gt_{class_names[gt]}_pred_{class_names[pred]}"
                    run_gradcam_for_patients(model, cell, cell_dir, class_names=class_names,
                                             z_slices=z, label=f"fold {fold} | {cell_dir.name}",
                                             **common)
                    reorganize_to_per_image_subdirs(cell_dir, [p["patient_id"] for p in cell])
            else:
                run_gradcam_for_patients(model, patients, fold_dir, class_names=class_names,
                                         z_slices=z, label=f"fold {fold}", **common)
                reorganize_to_per_image_subdirs(fold_dir, [p["patient_id"] for p in patients])
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"[fold {fold}] FAILED: {e}")
            traceback.print_exc()

    print(f"\nDone. Output tree under {out_base}")


if __name__ == "__main__":
    main()
