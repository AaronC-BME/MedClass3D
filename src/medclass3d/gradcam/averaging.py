"""Fold-averaged Layer-CAM maps from saved ``*_layercam_arrays.npz`` files.

Native MedClass3D implementation (no model rerun): for each patient, average the
normalized per-stage maps across the folds where the patient appears, then render
a summary grid. Reads the npz archives written by the gradcam runner (which stores
``stage*_normed`` plus ``input_ch0_normed`` / ``input_ch1_raw``).
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

ARRAY_SUFFIX = "_layercam_arrays.npz"
FOLD_RE = re.compile(r"fold_(\d+)")
STAGE_RE = re.compile(r"^stage(\d+)_normed$")


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def _patient_id_from_path(path: Path) -> str:
    n = path.name
    return n[:-len(ARRAY_SUFFIX)] if n.endswith(ARRAY_SUFFIX) else path.stem


def _infer_fold(path: Path) -> Optional[int]:
    for part in path.parts:
        m = FOLD_RE.fullmatch(part)
        if m:
            return int(m.group(1))
    return None


def _stage_keys(npz) -> List[str]:
    pairs = [(int(STAGE_RE.match(k).group(1)), k) for k in npz.files if STAGE_RE.match(k)]
    return [k for _, k in sorted(pairs)]


def _average_group(paths):
    accum, bases, masks, shape = defaultdict(list), [], [], None
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as npz:
            keys = _stage_keys(npz)
            if not keys:
                continue
            shape = shape or tuple(npz[keys[0]].shape)
            for k in keys:
                accum[k].append(npz[k].astype(np.float32))
            if "input_ch0_normed" in npz:
                bases.append(npz["input_ch0_normed"].astype(np.float32))
            if "input_ch1_raw" in npz:
                masks.append((npz["input_ch1_raw"] > 0.5).astype(np.float32))
    averaged = {k: np.mean(v, axis=0).astype(np.float32) for k, v in sorted(accum.items())}
    base = np.mean(bases, axis=0).astype(np.float32) if bases else np.zeros(shape, np.float32)
    mask = np.mean(masks, axis=0).astype(np.float32) if masks else None
    return averaged, base, mask


def _stride_z(depth: int) -> List[int]:
    step = max(depth // 8, 1)
    return list(range(step, depth, step))


def _largest_mask_z(mask, depth: int) -> List[int]:
    if mask is not None and mask.size:
        per_z = mask.reshape(mask.shape[0], -1).sum(axis=1)
        if float(per_z.max()) > 0:
            return [int(per_z.argmax())]
    return [min(depth // 2, depth - 1)]


def _draw_contour(ax, mask_slice):
    if mask_slice is None or float(mask_slice.max()) <= 0:
        return
    for c in measure.find_contours((mask_slice > 0.5).astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], color="black", lw=3.0, ls=":")
        ax.plot(c[:, 1], c[:, 0], color="magenta", lw=1.8, ls=":")


def _render_avg(out_path, pid, averaged, base, mask, folds, z, alpha, dpi):
    stages = list(averaged)
    cols = min(3, len(stages))
    rows = int(np.ceil(len(stages) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    axf = axes.ravel()
    im = None
    for i, s in enumerate(stages):
        ax = axf[i]
        ax.imshow(base[z], cmap="gray")
        im = ax.imshow(averaged[s][z] * (base[z] > 0.05), cmap="jet", alpha=alpha, vmin=0, vmax=1)
        if mask is not None:
            _draw_contour(ax, mask[z])
        ax.set_title(s, fontsize=12, fontweight="bold")
        ax.axis("off")
    for i in range(len(stages), len(axf)):
        axf[i].axis("off")
    fig.suptitle(f"{pid} | fold-averaged Layer-CAM | z={z} | folds={folds}",
                 fontsize=13, fontweight="bold")
    if im is not None:
        fig.colorbar(im, ax=axf[:len(stages)], fraction=0.025, pad=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def average_layercam(array_root, out_dir, *, min_folds=2, z_mode="stride",
                     alpha=0.35, dpi=150, npz_only=False) -> int:
    """Average Layer-CAM npz archives found under ``array_root`` and write
    summary npz/PNGs + a CSV under ``out_dir``. Returns the number of patients
    averaged."""
    array_root = Path(array_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = [p for p in sorted(array_root.rglob(f"*{ARRAY_SUFFIX}"))
              if p.parent.name != "fold_averaged" and "fold_averaged" not in p.parts]
    groups = defaultdict(list)
    for p in arrays:
        groups[_patient_id_from_path(p)].append(p)
    print(f"Found {len(arrays)} array file(s) across {len(groups)} patient(s).")

    summary_rows = []
    for pid, paths in sorted(groups.items()):
        folds = sorted({f for f in (_infer_fold(p) for p in paths) if f is not None})
        if len(folds) < min_folds:
            continue
        averaged, base, mask = _average_group(paths)
        if not averaged:
            continue
        npz_out = out_dir / "npz" / f"{_sanitize(pid)}_avg_layercam.npz"
        npz_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_out,
            **{k: v.astype(np.float32) for k, v in averaged.items()},
            input_ch0_normed_mean=base.astype(np.float32),
            input_ch1_mask_mean=(mask if mask is not None else np.zeros_like(base)).astype(np.float32),
            folds=np.array(folds, dtype=np.int16),
        )
        z_slices = (_largest_mask_z(mask, base.shape[0]) if z_mode == "largest-mask"
                    else _stride_z(base.shape[0]))
        if not npz_only:
            for z in z_slices:
                _render_avg(out_dir / "png" / f"{_sanitize(pid)}_avg_layercam_z{z}.png",
                            pid, averaged, base, mask, folds, z, alpha, dpi)
        summary_rows.append({"patient_id": pid, "n_folds": len(folds),
                             "folds": " ".join(map(str, folds)), "n_stages": len(averaged),
                             "z_slices": " ".join(map(str, z_slices))})
        print(f"  {pid}: folds={folds} -> {len(z_slices)} averaged map(s)")

    if summary_rows:
        with open(out_dir / "averaged_layercam_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"Averaged {len(summary_rows)} patient(s). Output: {out_dir}")
    return len(summary_rows)
