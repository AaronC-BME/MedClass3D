"""Fold-average saved Layer-CAM npz arrays and render summary maps.

Searches ``--array-root`` for ``*_layercam_arrays.npz`` (written by
gradcam_testset.py / gradcam_external.py), averages the normalized per-stage maps
across folds per patient, and writes averaged npz + PNGs + a summary CSV under
``--out-dir``. Does not rerun the model.

Usage:
    python scripts/average_layercam.py \\
        --array-root /path/to/run/gradcam/testset/layerCAM \\
        --out-dir   /path/to/run/gradcam/testset/layerCAM/fold_averaged
"""
import argparse
from pathlib import Path

from medclass3d.gradcam import average_layercam


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--array-root", required=True, type=Path,
                   help="Root to search for *_layercam_arrays.npz.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <array-root>/fold_averaged")
    p.add_argument("--min-folds", type=int, default=2)
    p.add_argument("--z-mode", choices=("stride", "largest-mask"), default="stride")
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--npz-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.array_root / "fold_averaged")
    average_layercam(args.array_root, out_dir, min_folds=args.min_folds,
                     z_mode=args.z_mode, alpha=args.alpha, dpi=args.dpi,
                     npz_only=args.npz_only)


if __name__ == "__main__":
    main()
