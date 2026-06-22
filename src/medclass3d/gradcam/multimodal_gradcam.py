"""
multimodal_gradcam.py — Multi-modal extensions for gradcam3d_viz.

When a model takes channel-stacked inputs (e.g. T1c+T2w), the standard
gradcam3d_viz produces figures with a blended background and aggregates
the model's response across all modalities. This module:

  1. Adds per-modality occlusion methods (occlusion_t1c, occlusion_t2w)
     by zeroing only one channel at each cube position.
  2. Renders side-by-side figures showing each modality and the attention
     overlay separately, so you can compare attention against each
     modality's anatomy.

The standard methods (notgradcam, layercam, etc.) work unchanged — they
just use whatever the model does internally with multi-channel input.
This module changes how those methods are *visualized*, not computed.

Single-modal data is unaffected: run_multimodal_gradcam delegates to
gradcam3d_viz.run_gradcam when C == 1.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# We pull in the private compute/normalize/validate functions from
# gradcam3d_viz. They're underscore-prefixed but importable.
from .gradcam3d_viz import (
    GradcamConfig,
    run_gradcam,
    ALL_METHODS,
    _default_extract_logits,
    _validate_input,
    _validate_gt_seg,
    _pick_target_class,
    _compute_notgradcam,
    _compute_truegradcam,
    _compute_guided_gradcam,
    _compute_layercam,
    _compute_integrated_gradients,
    _compute_integrated_gradcam,
    _normalize_per_stage,
    _normalize_volume,
    _brain_mask_slice,
    _classify_outcome,
    _sanitize_for_path,
    _draw_contours_on_ax,
    _get_z_slices,
    _get_grid_shape,
    _compute_slice_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-modality occlusion
# ─────────────────────────────────────────────────────────────────────────────

def _compute_per_channel_occlusion(model, x_batch, target_shape, target_cls,
                                   extract_logits_fn, channel_idx,
                                   mask_size, overlap):
    """Occlusion sensitivity that zeros only one input channel at a time.

    For each spatial cube position, replace x_batch[0, channel_idx, z..z+m, ...]
    with zeros (other channels untouched), forward, record score drop. Stitch
    drops back into a 3D heatmap.

    Args:
        model: torch model, eval mode
        x_batch: (1, C, D, H, W) tensor on the model's device
        target_shape: (D, H, W) tuple — input spatial size
        target_cls: int — class to track
        extract_logits_fn: callable(model_out) -> (B, num_classes) tensor
        channel_idx: which channel to occlude (0 or 1 typically)
        mask_size: cube edge length (voxels)
        overlap: fraction of cube to step (0.5 = 50% overlap)

    Returns:
        np.ndarray of shape target_shape — score-drop heatmap. Higher values
        mean "occluding this region of the chosen modality hurt the model's
        confidence in target_cls more."
    """
    device = x_batch.device
    D, H, W = target_shape
    stride = max(1, int(mask_size * (1.0 - overlap)))

    # Reference score with no occlusion
    with torch.no_grad():
        logits_ref = extract_logits_fn(model(x_batch))
        score_ref = float(logits_ref[0, target_cls].item())

    # Accumulators for averaging overlapping cube positions
    score_drop = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    # Build the list of cube origin positions
    z_origins = list(range(0, max(1, D - mask_size + 1), stride))
    y_origins = list(range(0, max(1, H - mask_size + 1), stride))
    x_origins = list(range(0, max(1, W - mask_size + 1), stride))
    # Make sure we cover the trailing edge
    if z_origins[-1] + mask_size < D:
        z_origins.append(D - mask_size)
    if y_origins[-1] + mask_size < H:
        y_origins.append(H - mask_size)
    if x_origins[-1] + mask_size < W:
        x_origins.append(W - mask_size)

    n_total = len(z_origins) * len(y_origins) * len(x_origins)
    n_done = 0

    with torch.no_grad():
        for z0 in z_origins:
            for y0 in y_origins:
                for x0 in x_origins:
                    z1 = min(z0 + mask_size, D)
                    y1 = min(y0 + mask_size, H)
                    x1 = min(x0 + mask_size, W)

                    # Clone and zero the chosen channel in this cube
                    x_occ = x_batch.clone()
                    x_occ[0, channel_idx, z0:z1, y0:y1, x0:x1] = 0.0

                    logits_occ = extract_logits_fn(model(x_occ))
                    score_occ = float(logits_occ[0, target_cls].item())
                    drop = score_ref - score_occ  # positive = important region

                    score_drop[z0:z1, y0:y1, x0:x1] += drop
                    count[z0:z1, y0:y1, x0:x1] += 1.0
                    n_done += 1

    # Average overlapping cubes
    score_drop /= np.clip(count, 1.0, None)
    return score_drop


# ─────────────────────────────────────────────────────────────────────────────
# Multi-modal config
# ─────────────────────────────────────────────────────────────────────────────

class MultiModalConfig:
    """Add-on config for multi-modal runs.

    Attributes:
        modality_names: List of human-readable names for each input channel,
                        e.g. ["T1c", "T2w"]. Used in figure column labels and
                        method names (occlusion_T1c, occlusion_T2w).
        per_modality_occlusion: If True and C > 1, run per-channel occlusion
                                in addition to whatever methods the user asked
                                for via GradcamConfig.methods.
        side_by_side_figures: If True and C > 1, render figures with one
                              column per modality + one for the attention
                              overlay. If False, fall back to the standard
                              blended-background figures from run_gradcam.
    """
    def __init__(self,
                 modality_names: Optional[List[str]] = None,
                 per_modality_occlusion: bool = True,
                 side_by_side_figures: bool = True):
        self.modality_names = modality_names
        self.per_modality_occlusion = per_modality_occlusion
        self.side_by_side_figures = side_by_side_figures


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side figure rendering
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_per_modality(x_np: np.ndarray) -> List[np.ndarray]:
    """Per-channel min-max normalize a (C, D, H, W) volume. Returns list of
    normalized (D, H, W) arrays — one per channel."""
    out = []
    for c in range(x_np.shape[0]):
        v = x_np[c].astype(np.float32)
        vmin, vmax = float(v.min()), float(v.max())
        n = (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)
        out.append(np.clip(n, 0.0, 1.0))
    return out


def _make_multimodal_grid_figure(
    pid: str,
    z: int,
    modality_volumes: List[np.ndarray],   # list of (D, H, W), per modality
    modality_names: List[str],
    composite_slices: Dict[str, Tuple[np.ndarray, np.ndarray, int]],
    gt_seg_vol: Optional[np.ndarray],
    method_label: str,
    target_cls: int,
    outcome_label: str,
    outcome_color: str,
    gt_label_val: Optional[int],
    pred_label_val: Optional[int],
    class_names: Dict[int, str],
    alpha: float,
    dpi: int,
    out_path: Path,
    brain_mask_thr: float = 0.05,
):
    """Multi-modal version of _make_grid_figure.

    Layout: rows = encoder stages, columns = (modality_1, modality_2, ...,
    modality_N, attention_overlay). The attention overlay panel shows the
    attention map on top of a *blend* of all modalities, so anatomy isn't
    biased toward one channel.
    """
    stage_keys = list(composite_slices.keys())
    n_stages = len(stage_keys)
    n_modalities = len(modality_volumes)
    n_cols = n_modalities + 1  # +1 for the attention overlay column

    fig_w = 4.0 * n_cols
    fig_h = 4.0 * n_stages
    fig, axes = plt.subplots(n_stages, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    last_im = None
    blended = np.mean(modality_volumes, axis=0)  # (D, H, W)

    for row_idx, stage_name in enumerate(stage_keys):
        sl_base, sl_att, z_s = composite_slices[stage_name]
        gt_seg_slice = gt_seg_vol[z_s] if gt_seg_vol is not None else None

        # Modality columns: just show each modality slice, with GT contour
        for c in range(n_modalities):
            ax = axes[row_idx, c]
            mod_slice = modality_volumes[c][z_s]
            ax.imshow(mod_slice, cmap="gray")
            _draw_contours_on_ax(ax, gt_seg_slice, color='lime', linewidth=1.5)
            if row_idx == 0:
                ax.set_title(modality_names[c], fontsize=14, fontweight="bold")
            if c == 0:
                # Stage name in left margin (only on the leftmost column)
                ax.set_ylabel(f"{stage_name}\nz={z_s}", fontsize=10,
                              rotation=0, labelpad=40, va="center")
            ax.set_xticks([])
            ax.set_yticks([])

        # Last column: attention overlay on blended background
        ax = axes[row_idx, -1]
        ax.imshow(blended[z_s], cmap="gray")
        im = ax.imshow(sl_att, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        _draw_contours_on_ax(ax, gt_seg_slice, color='lime', linewidth=1.5)
        last_im = im
        if row_idx == 0:
            ax.set_title(f"{method_label}", fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Add a single colorbar on the far right
    if last_im is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
        fig.colorbar(last_im, cax=cbar_ax)

    # Title + outcome badge
    gt_name = class_names.get(gt_label_val, "?")
    pred_name = class_names.get(pred_label_val, "?")
    title = (f"{pid}    z={z}    {method_label}\n"
             f"GT: {gt_label_val} ({gt_name})  |  "
             f"Pred: {pred_label_val} ({pred_name})")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.98, 0.995, outcome_label, fontsize=18, fontweight="bold",
             color=outcome_color, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _make_multimodal_single_figure(
    pid: str,
    z: int,
    modality_volumes: List[np.ndarray],
    modality_names: List[str],
    base_slice_blend: np.ndarray,
    att_slice: np.ndarray,
    gt_seg_slice: Optional[np.ndarray],
    method_label: str,
    target_cls: int,
    outcome_label: str,
    outcome_color: str,
    gt_label_val: Optional[int],
    pred_label_val: Optional[int],
    class_names: Dict[int, str],
    alpha: float,
    dpi: int,
    out_path: Path,
):
    """Single-row version for input-resolution methods (occlusion, IG)."""
    n_modalities = len(modality_volumes)
    n_cols = n_modalities + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 4.5),
                             squeeze=False)
    axes = axes[0]

    for c in range(n_modalities):
        ax = axes[c]
        ax.imshow(modality_volumes[c][z], cmap="gray")
        _draw_contours_on_ax(ax, gt_seg_slice, color='lime', linewidth=1.5)
        ax.set_title(modality_names[c], fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Overlay panel
    ax = axes[-1]
    ax.imshow(base_slice_blend, cmap="gray")
    im = ax.imshow(att_slice, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    _draw_contours_on_ax(ax, gt_seg_slice, color='lime', linewidth=1.5)
    ax.set_title(method_label, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    gt_name = class_names.get(gt_label_val, "?")
    pred_name = class_names.get(pred_label_val, "?")
    title = (f"{pid}    z={z}    {method_label}\n"
             f"GT: {gt_label_val} ({gt_name})  |  "
             f"Pred: {pred_label_val} ({pred_name})")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.98, 0.995, outcome_label, fontsize=18, fontweight="bold",
             color=outcome_color, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-modal orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_multimodal_gradcam(model, patients, config: GradcamConfig,
                           mm_config: MultiModalConfig):
    """Multi-modal gradcam runner.

    For single-channel patients, delegates to gradcam3d_viz.run_gradcam.
    For multi-channel patients, replicates run_gradcam's logic but renders
    side-by-side figures and adds per-modality occlusion methods.
    """
    if not patients:
        return []

    # Decide single- vs multi-modal based on first patient
    sample_C = patients[0]["input"].shape[0]
    if sample_C == 1 or not mm_config.side_by_side_figures:
        # Single-modal or user explicitly disabled side-by-side: standard path
        print(f"  (multi-modal wrapper) C={sample_C}, side_by_side="
              f"{mm_config.side_by_side_figures} — delegating to standard "
              f"run_gradcam.")
        return run_gradcam(model, patients, config)

    # Multi-modal path with side-by-side figures
    n_modalities = sample_C
    if mm_config.modality_names is None:
        modality_names = [f"ch{i}" for i in range(n_modalities)]
    elif len(mm_config.modality_names) != n_modalities:
        print(f"  WARNING: {len(mm_config.modality_names)} modality names "
              f"provided but input has {n_modalities} channels. Falling back "
              f"to ch0/ch1/...")
        modality_names = [f"ch{i}" for i in range(n_modalities)]
    else:
        modality_names = mm_config.modality_names

    print(f"  Multi-modal mode: {n_modalities} modalities = {modality_names}")
    if mm_config.per_modality_occlusion:
        print(f"  Per-modality occlusion enabled.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    stages = config.get_stages_fn(model)
    stage_names = [f"stage{i}" for i in range(len(stages))]

    # Output dirs — same as run_gradcam, plus per-modality occlusion dirs
    out_base = Path(config.out_dir)
    method_subdirs = {
        "notgradcam": "notgradcam",
        "truegradcam": "truegradcam",
        "guided_gradcam": "guided_gradcam",
        "layercam": "layercam",
        "occlusion": "occlusion",
        "integrated_gradients": "integrated_grad",
        "integrated_gradcam": "integrated_gradcam",
    }
    out_dirs = {m: out_base / method_subdirs[m] for m in config.methods}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    if mm_config.per_modality_occlusion:
        for mname in modality_names:
            d = out_base / f"occlusion_{_sanitize_for_path(mname)}"
            d.mkdir(parents=True, exist_ok=True)
            out_dirs[f"occlusion_{mname}"] = d

    all_meta = []

    for p_idx, patient in enumerate(patients):
        pid = patient["patient_id"]
        pid_sanitized = _sanitize_for_path(pid)
        x = patient["input"]
        gt_label = patient.get("gt_label")
        pred_label = patient.get("pred_label")
        gt_seg_vol = patient.get("gt_seg")

        print(f"\n[{p_idx + 1}/{len(patients)}] Patient: {pid}")

        _validate_input(x, pid)
        C, D, H, W = x.shape
        target_shape = (D, H, W)
        if gt_seg_vol is not None:
            _validate_gt_seg(gt_seg_vol, x.shape, pid)

        # Per-modality and blended bases
        x_np = x.cpu().numpy()
        modality_volumes = _normalize_per_modality(x_np)  # list of (D,H,W)
        base_n = np.mean(modality_volumes, axis=0)        # blended

        x_batch = x.unsqueeze(0).to(device).float()

        # Resolve pred and target class
        if pred_label is None:
            with torch.no_grad():
                logits_check = config.extract_logits_fn(model(x_batch))
                if logits_check.shape[-1] == 1:
                    pred_label = 1 if logits_check[0, 0].item() > 0 else 0
                else:
                    pred_label = int(torch.argmax(logits_check[0]).item())
        outcome_label, outcome_color = _classify_outcome(gt_label, pred_label)
        print(f"  GT={gt_label}  Pred={pred_label}  -> {outcome_label}")

        with torch.no_grad():
            logits0 = config.extract_logits_fn(model(x_batch))
        target_cls, _ = _pick_target_class(logits0, config.target_class)
        print(f"  Target class: {target_cls}")

        # Z-slices
        z_slices = (patient.get("z_slices") or
                    _get_z_slices(gt_seg_vol, D, config.default_z_slices))
        print(f"  Z-slices: {len(z_slices)} [{min(z_slices)}..{max(z_slices)}]")

        # ── Compute methods ──
        pclip = config.percentile_clip
        thr = config.brain_mask_threshold

        notgradcam_normed = {}
        truegradcam_raw = {}
        truegradcam_normed = {}
        guided_gradcam_normed = {}
        layercam_raw = {}
        layercam_normed = {}
        occlusion_normed = None
        intgrad_normed = None
        integrated_gradcam_normed = {}
        per_modality_occ_normed = {}  # name -> normalized volume

        if "notgradcam" in config.methods:
            print("  Computing NotGradCam...")
            raw = _compute_notgradcam(model, x_batch, stages, stage_names,
                                      target_shape, pid)
            notgradcam_normed = _normalize_per_stage(raw, base_n, pclip, thr)

        if any(m in config.methods for m in ("truegradcam", "guided_gradcam",
                                             "integrated_gradcam")):
            print("  Computing TrueGradCam...")
            truegradcam_raw = _compute_truegradcam(
                model, x_batch, stages, stage_names, target_shape,
                target_cls, config.extract_logits_fn, pid)
            truegradcam_normed = _normalize_per_stage(truegradcam_raw, base_n,
                                                     pclip, thr)

        if "guided_gradcam" in config.methods:
            print("  Computing Guided Grad-CAM...")
            raw = _compute_guided_gradcam(
                model, x_batch, truegradcam_raw, stages, stage_names,
                target_shape, target_cls, config.extract_logits_fn, pid)
            guided_gradcam_normed = _normalize_per_stage(raw, base_n, pclip, thr)

        if "layercam" in config.methods:
            print("  Computing Layer-CAM...")
            layercam_raw = _compute_layercam(
                model, x_batch, stages, stage_names, target_shape,
                target_cls, config.extract_logits_fn, pid)
            layercam_normed = _normalize_per_stage(
                layercam_raw, base_n, pclip, thr)

            arr_path = out_dirs["layercam"] / (
                f"{pid_sanitized}_layercam_arrays.npz")
            np.savez_compressed(
                arr_path,
                **{f"{s}_raw": v.astype(np.float32)
                   for s, v in layercam_raw.items()},
                **{f"{s}_normed": v.astype(np.float32)
                   for s, v in layercam_normed.items()},
                input_ch0_raw=x_np[0].astype(np.float32),
                input_ch1_raw=x_np[1].astype(np.float32),
                input_ch0_normed=modality_volumes[0].astype(np.float32),
                input_ch1_normed=modality_volumes[1].astype(np.float32),
            )
            print(f"  Layer-CAM arrays: {arr_path}")

        if getattr(config, "npz_only", False):
            continue

        if "occlusion" in config.methods:
            print(f"  Computing standard Occlusion (mask={config.occ_mask_size})...")
            from .gradcam3d_viz import _compute_occlusion
            occ_arr = _compute_occlusion(
                model, x_batch, target_shape, target_cls,
                config.extract_logits_fn, config, pid)
            occlusion_normed = _normalize_volume(occ_arr, base_n, pclip, thr)

        if "integrated_gradients" in config.methods:
            print(f"  Computing Integrated Gradients...")
            ig_arr = _compute_integrated_gradients(
                model, x_batch, target_shape, target_cls,
                config.extract_logits_fn, config, pid)
            intgrad_normed = _normalize_volume(ig_arr, base_n, pclip, thr)

        if "integrated_gradcam" in config.methods:
            print("  Computing Integrated Grad-CAM...")
            raw = _compute_integrated_gradcam(
                model, x_batch, stages, stage_names, target_shape,
                target_cls, config.extract_logits_fn, config, pid)
            integrated_gradcam_normed = _normalize_per_stage(raw, base_n,
                                                            pclip, thr)

        if mm_config.per_modality_occlusion:
            for c_idx, mname in enumerate(modality_names):
                print(f"  Computing Occlusion for {mname} only "
                      f"(mask={config.occ_mask_size})...")
                arr = _compute_per_channel_occlusion(
                    model, x_batch, target_shape, target_cls,
                    config.extract_logits_fn, c_idx,
                    config.occ_mask_size, config.occ_overlap)
                per_modality_occ_normed[mname] = _normalize_volume(
                    arr, base_n, pclip, thr)

        # ── Render figures ──
        gt_suffix = "_GT" if gt_seg_vol is not None else ""
        slice_metadata = []
        saved = 0

        # Per-stage methods (notgradcam, truegradcam, guided, layercam, igcam)
        per_stage_jobs = [
            ("notgradcam", notgradcam_normed, "Activation Map (NotGradCam)"),
            ("truegradcam", truegradcam_normed, "True Grad-CAM"),
            ("guided_gradcam", guided_gradcam_normed, "Guided Grad-CAM"),
            ("layercam", layercam_normed, "Layer-CAM"),
            ("integrated_gradcam", integrated_gradcam_normed,
             "Integrated Grad-CAM"),
        ]

        for z in z_slices:
            gt_seg_z = gt_seg_vol[z] if gt_seg_vol is not None else None

            # Per-stage methods get a multi-row grid figure (one row per stage)
            for method_key, normed_dict, label in per_stage_jobs:
                if method_key not in config.methods or not normed_dict:
                    continue
                slices = {}
                for sname, att_n in normed_dict.items():
                    att_z = (att_n[z] if method_key == "notgradcam"
                             else _brain_mask_slice(base_n[z], att_n[z], thr))
                    slices[sname] = (base_n[z].copy(), att_z.copy(), z)
                path = out_dirs[method_key] / (
                    f"{pid_sanitized}_{method_key}_z{z}.png"
                    if method_key == "layercam" else
                    f"{pid_sanitized}_{method_key}_z{z}_class{target_cls}_"
                    f"{outcome_label}{gt_suffix}.png")
                _make_multimodal_grid_figure(
                    pid=pid, z=z,
                    modality_volumes=modality_volumes,
                    modality_names=modality_names,
                    composite_slices=slices,
                    gt_seg_vol=gt_seg_vol,
                    method_label=label,
                    target_cls=target_cls,
                    outcome_label=outcome_label,
                    outcome_color=outcome_color,
                    gt_label_val=gt_label,
                    pred_label_val=pred_label,
                    class_names=config.class_names,
                    alpha=config.alpha,
                    dpi=config.dpi,
                    out_path=path,
                )
                saved += 1
                for sname, att_n in normed_dict.items():
                    m = _compute_slice_metrics(att_n[z], gt_seg_z)
                    if m:
                        m.update({"method": method_key, "stage": sname,
                                  "z": int(z), "figure": path.name})
                        slice_metadata.append(m)

            # Single-map methods (occlusion, integrated_gradients)
            single_jobs = []
            if "occlusion" in config.methods and occlusion_normed is not None:
                single_jobs.append((
                    "occlusion",
                    _brain_mask_slice(base_n[z], occlusion_normed[z], thr),
                    "Occlusion (full)"))
            if "integrated_gradients" in config.methods and intgrad_normed is not None:
                single_jobs.append((
                    "integrated_gradients",
                    _brain_mask_slice(base_n[z], intgrad_normed[z], thr),
                    "Integrated Gradients"))
            for mname, occ_norm in per_modality_occ_normed.items():
                key = f"occlusion_{mname}"
                single_jobs.append((
                    key,
                    _brain_mask_slice(base_n[z], occ_norm[z], thr),
                    f"Occlusion ({mname} only)"))

            for key, att_z, label in single_jobs:
                path = out_dirs[key] / (
                    f"{pid_sanitized}_{key}_z{z}_class{target_cls}_"
                    f"{outcome_label}{gt_suffix}.png")
                _make_multimodal_single_figure(
                    pid=pid, z=z,
                    modality_volumes=modality_volumes,
                    modality_names=modality_names,
                    base_slice_blend=base_n[z].copy(),
                    att_slice=att_z.copy(),
                    gt_seg_slice=gt_seg_z,
                    method_label=label,
                    target_cls=target_cls,
                    outcome_label=outcome_label,
                    outcome_color=outcome_color,
                    gt_label_val=gt_label,
                    pred_label_val=pred_label,
                    class_names=config.class_names,
                    alpha=config.alpha,
                    dpi=config.dpi,
                    out_path=path,
                )
                saved += 1
                m = _compute_slice_metrics(att_z, gt_seg_z)
                if m:
                    m.update({"method": key, "stage": "input_resolution",
                              "z": int(z), "figure": path.name})
                    slice_metadata.append(m)

        # Save metadata JSON
        patient_meta = {
            "patient_id": pid,
            "outcome": outcome_label,
            "gt_label": int(gt_label) if gt_label is not None else None,
            "pred_label": int(pred_label) if pred_label is not None else None,
            "target_class": int(target_cls),
            "n_modalities": n_modalities,
            "modality_names": modality_names,
            "n_slices": len(z_slices),
            "z_slices": [int(z) for z in z_slices],
            "has_gt_segmentation": gt_seg_vol is not None,
            "slices": slice_metadata,
        }
        if slice_metadata:
            meta_path = out_base / f"{pid_sanitized}_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(patient_meta, f, indent=2)

        all_meta.append(patient_meta)
        print(f"  Saved {saved} figures for {pid}")

    print(f"\nDone. {len(patients)} patients, output: {out_base}")
    return all_meta
