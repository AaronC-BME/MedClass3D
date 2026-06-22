#!/usr/bin/env python3
"""
gradcam3d_viz.py — Project-Independent 3D Grad-CAM Visualization

Generates per-slice attribution map figures for 3D classification models.
Supports 7 methods: notgradcam, truegradcam, guided_gradcam, layercam,
occlusion, integrated_gradients, integrated_gradcam.

Usage:
    from gradcam3d_viz import run_gradcam, GradcamConfig

    config = GradcamConfig(
        get_stages_fn=lambda m: list(m.encoder.stages),
        class_names={0: "Healthy", 1: "Disease"},
        methods=["notgradcam", "truegradcam"],
        out_dir="./gradcam_output",
    )
    run_gradcam(model, patients, config)

See main() at the bottom for a full usage example.

Requirements: torch, numpy, matplotlib, scipy, scikit-image, monai
"""

import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from skimage import measure

# NOTE: monai is imported lazily inside the methods that need it
# (_compute_truegradcam / _compute_guided_gradcam / _compute_occlusion) so the
# package stays importable for the Layer-CAM / NotGradCAM / Integrated-Gradients
# methods even when monai is not installed.


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

ALL_METHODS = [
    "notgradcam", "truegradcam", "guided_gradcam", "layercam",
    "occlusion", "integrated_gradients", "integrated_gradcam",
]

METHOD_SUBDIRS = {
    "notgradcam": "notgradcam",
    "truegradcam": "truegradcam",
    "guided_gradcam": "guided_gradcam",
    "layercam": "layercam",
    "occlusion": "occlusion",
    "integrated_gradients": "integrated_grad",
    "integrated_gradcam": "integrated_gradcam",
}


def _default_extract_logits(model_out):
    """Default logits extractor. Handles tensor, tuple, and dict outputs."""
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, (list, tuple)) and torch.is_tensor(model_out[0]):
        return model_out[0]
    if isinstance(model_out, dict):
        for k in ("logits", "pred", "y_hat", "outputs", "out"):
            if k in model_out and torch.is_tensor(model_out[k]):
                return model_out[k]
    raise RuntimeError(f"Cannot extract logits from {type(model_out)}. "
                       "Provide a custom extract_logits_fn in GradcamConfig.")


@dataclass
class GradcamConfig:
    """Configuration for run_gradcam().

    Required:
        get_stages_fn:  callable(model) -> list[nn.Module]
                        Returns the encoder stages to visualize.
        class_names:    dict mapping class index to display name.

    Optional (all have sensible defaults):
        extract_logits_fn:  callable(model_output) -> logits Tensor
        target_class:       -1 = use predicted class
        methods:            list of method names, or ["all"]
        out_dir:            output directory path
    """
    # ── Model ──
    get_stages_fn: Callable = None  # REQUIRED: model -> list[nn.Module]
    extract_logits_fn: Callable = field(default_factory=lambda: _default_extract_logits)

    # ── Task ──
    class_names: Dict[int, str] = field(default_factory=lambda: {0: "Class 0", 1: "Class 1"})
    target_class: int = -1  # -1 = predicted class

    # ── Methods ──
    methods: List[str] = field(default_factory=lambda: ["notgradcam", "truegradcam"])

    # ── Figure ──
    alpha: float = 0.3
    dpi: int = 150
    manuscript_mode: bool = False

    # ── Method-specific params ──
    occ_mask_size: int = 20
    occ_overlap: float = 0.5
    occ_n_batch: int = 8
    ig_n_steps: int = 50
    ig_batch_size: int = 4
    ig_smooth_sigma: float = 2.0  # 0 to disable
    igc_n_steps: int = 50

    # ── Normalization ──
    brain_mask_threshold: float = 0.05
    percentile_clip: Tuple[float, float] = (1, 99)

    # ── Slice selection ──
    default_z_slices: Optional[List[int]] = None  # fallback when no GT seg

    # ── Output ──
    out_dir: str = "./gradcam_output"

    def __post_init__(self):
        if self.get_stages_fn is None:
            raise ValueError("get_stages_fn is required. Example: "
                             "lambda m: list(m.encoder.stages)")
        if "all" in self.methods:
            self.methods = list(ALL_METHODS)
        for m in self.methods:
            if m not in ALL_METHODS:
                raise ValueError(f"Unknown method '{m}'. Choose from: {ALL_METHODS}")


# ═══════════════════════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_input(x: torch.Tensor, patient_id: str):
    """Validate input tensor shape."""
    assert isinstance(x, torch.Tensor), (
        f"[{patient_id}] 'input' must be a torch.Tensor, got {type(x)}")
    assert x.ndim == 4, (
        f"[{patient_id}] 'input' must be (C, D, H, W), got shape {tuple(x.shape)}")
    C, D, H, W = x.shape
    assert C < D and C < H and C < W, (
        f"[{patient_id}] 'input' shape {tuple(x.shape)} looks wrong — "
        f"channel dim ({C}) should be smallest. Expected (C, D, H, W).")


def _validate_gt_seg(gt_seg: np.ndarray, input_shape: tuple, patient_id: str):
    """Validate GT segmentation matches input spatial dims."""
    C, D, H, W = input_shape
    assert isinstance(gt_seg, np.ndarray), (
        f"[{patient_id}] 'gt_seg' must be numpy array, got {type(gt_seg)}")
    assert gt_seg.ndim == 3, (
        f"[{patient_id}] 'gt_seg' must be (D, H, W), got shape {tuple(gt_seg.shape)}")
    assert gt_seg.shape == (D, H, W), (
        f"[{patient_id}] 'gt_seg' shape {tuple(gt_seg.shape)} != "
        f"input spatial shape ({D}, {H}, {W}). "
        "GT seg must be pre-aligned to match the input tensor.")


def _validate_att_volume(att: np.ndarray, target_shape: tuple, stage_name: str,
                         method: str, patient_id: str):
    """Validate attention volume after upsampling to input resolution."""
    assert att.shape == target_shape, (
        f"[{patient_id}] {method}/{stage_name}: attention shape {tuple(att.shape)} != "
        f"expected {target_shape} after upsampling. This is a bug.")


def _validate_slice_shapes(base_slice: np.ndarray, att_slice: np.ndarray,
                           seg_slice: Optional[np.ndarray], z: int,
                           patient_id: str, method: str):
    """Validate all 2D slices match before overlay."""
    H, W = base_slice.shape
    assert att_slice.shape == (H, W), (
        f"[{patient_id}] {method} z={z}: att_slice shape {att_slice.shape} != "
        f"base_slice shape ({H}, {W})")
    if seg_slice is not None:
        assert seg_slice.shape == (H, W), (
            f"[{patient_id}] {method} z={z}: seg_slice shape {seg_slice.shape} != "
            f"base_slice shape ({H}, {W})")


def _validate_z_in_range(z: int, D: int, patient_id: str):
    """Validate z-slice index is within volume depth."""
    assert 0 <= z < D, (
        f"[{patient_id}] z-slice {z} is out of range [0, {D})")


# ═══════════════════════════════════════════════════════════════════════════════
# Guided backpropagation
# ═══════════════════════════════════════════════════════════════════════════════

class _GuidedLeakyReLUFunc(torch.autograd.Function):
    """Autograd function that gates backward pass for guided backpropagation."""
    @staticmethod
    def forward(ctx, x, negative_slope):
        ctx.save_for_backward(x)
        ctx.negative_slope = negative_slope
        return torch.nn.functional.leaky_relu(x, negative_slope)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (grad_output > 0).float() * (x > 0).float(), None


class GuidedLeakyReLU(nn.Module):
    """LeakyReLU replacement for guided backpropagation.
    Forward: standard LeakyReLU. Backward: gates by (grad > 0) * (input > 0)."""
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return _GuidedLeakyReLUFunc.apply(x, self.negative_slope)


class _GuidedReLUFunc(torch.autograd.Function):
    """Same as _GuidedLeakyReLUFunc but for standard ReLU."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (grad_output > 0).float() * (x > 0).float()


class GuidedReLU(nn.Module):
    """ReLU replacement for guided backpropagation."""
    def forward(self, x):
        return _GuidedReLUFunc.apply(x)


@contextmanager
def guided_backprop_context(model):
    """Temporarily replace all ReLU/LeakyReLU modules with guided variants.

    Iterates model._modules recursively (not __dict__) to find registered
    submodules — this is critical because PyTorch stores submodules in
    _modules, and some models name them 'nonlin' rather than 'relu'.
    """
    originals = {}
    for name, mod in model.named_modules():
        for attr_name, submod in list(mod._modules.items()):
            if isinstance(submod, nn.LeakyReLU):
                originals[(mod, attr_name)] = submod
                mod._modules[attr_name] = GuidedLeakyReLU(submod.negative_slope)
            elif isinstance(submod, nn.ReLU):
                originals[(mod, attr_name)] = submod
                mod._modules[attr_name] = GuidedReLU()
    print(f"    [guided_backprop] Replaced {len(originals)} ReLU/LeakyReLU modules")
    try:
        yield
    finally:
        for (parent, attr_name), orig in originals.items():
            parent._modules[attr_name] = orig


# ═══════════════════════════════════════════════════════════════════════════════
# MONAI wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class _MonaiGradCamWrapper(nn.Module):
    """Wraps any model so MONAI GradCAM can call forward(x, **kwargs) -> logits."""
    def __init__(self, model, target_layer, extract_logits_fn):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self._extract_logits = extract_logits_fn

    def forward(self, x, **kwargs):
        return self._extract_logits(self.model(x))


def _get_module_name(root, target):
    """Find the dotted module name of target inside root."""
    for name, mod in root.named_modules():
        if mod is target:
            return name
    raise RuntimeError("Could not find target module inside wrapper.")


def _pick_target_class(logits, user_target_class):
    """Resolve target class: user-specified or predicted."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.shape[1] == 1:
        cls = 0 if (user_target_class == -1 or user_target_class == 0) else 1
        score = logits[0, 0] if cls == 0 else -logits[0, 0]
        return cls, score
    cls = int(torch.argmax(logits[0]).item()) if user_target_class == -1 else int(user_target_class)
    return cls, logits[0, cls]


# ═══════════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_per_stage(raw_dict, brain_vol, percentile_clip=(1, 99),
                         threshold=0.05):
    """Normalize each stage's attention volume to [0,1] using percentile clipping
    within a brain mask."""
    normed = {}
    mask = (brain_vol > threshold) if brain_vol is not None else None
    for sname, att in raw_dict.items():
        if mask is not None and mask.sum() > 0:
            vals = att[mask]
            low = np.percentile(vals, percentile_clip[0])
            high = np.percentile(vals, percentile_clip[1])
        else:
            low, high = float(att.min()), float(att.max())
        if high - low < 1e-8:
            low, high = float(att.min()), float(att.max())
        normed[sname] = np.clip((att - low) / (high - low + 1e-8), 0.0, 1.0)
    return normed


def _normalize_volume(arr, brain_vol, percentile_clip=(1, 99), threshold=0.05):
    """Normalize a single volume to [0,1] using percentile clipping within
    brain mask to avoid edge artifacts dominating the scale."""
    if brain_vol is not None:
        mask = brain_vol > threshold
        if mask.sum() > 0:
            vals = arr[mask]
            low = np.percentile(vals, percentile_clip[0])
            high = np.percentile(vals, percentile_clip[1])
        else:
            low = np.percentile(arr, percentile_clip[0])
            high = np.percentile(arr, percentile_clip[1])
    else:
        low = np.percentile(arr, percentile_clip[0])
        high = np.percentile(arr, percentile_clip[1])
    if high - low < 1e-8:
        low, high = float(arr.min()), float(arr.max())
    return np.clip((arr - low) / (high - low + 1e-8), 0.0, 1.0)


def _brain_mask_slice(sl_base, sl_att, thr=0.05):
    """Zero out attention in background voxels."""
    return sl_att * (sl_base > thr)


# ═══════════════════════════════════════════════════════════════════════════════
# Outcome classification
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_outcome(gt, pred):
    """Map (gt_label, pred_label) -> (outcome_str, color)."""
    if gt is None or pred is None:
        return "N/A", "white"
    if gt == 1 and pred == 1:
        return "TP", "#2ecc71"
    if gt == 0 and pred == 0:
        return "TN", "#3498db"
    if gt == 0 and pred == 1:
        return "FP", "#e67e22"
    if gt == 1 and pred == 0:
        return "FN", "#e74c3c"
    return "??", "white"


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize_for_path(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def _draw_contours_on_ax(ax, seg_slice, color='lime', linewidth=2):
    if seg_slice is None or seg_slice.max() == 0:
        return
    for c in measure.find_contours(seg_slice.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=linewidth)


def _add_gt_activation_indicator(cbar, val, color='lime'):
    if val is None:
        return
    cbar.ax.annotate('', xy=(1.0, val), xytext=(1.5, val),
                     xycoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))
    cbar.ax.text(1.6, val, f'GT:{val:.2f}', transform=cbar.ax.transAxes,
                 va='center', ha='left', fontsize=8, fontweight='bold', color=color)


def _compute_gt_activation(att_slice, seg_slice):
    if seg_slice is None or seg_slice.max() == 0:
        return None
    mask = seg_slice > 0
    return float(att_slice[mask].mean()) if mask.sum() > 0 else None


def _compute_slice_metrics(att_slice, seg_slice, high_act_threshold=0.5):
    """Quantitative metrics: activation overlap between attention and GT seg."""
    if seg_slice is None or seg_slice.max() == 0:
        return None
    seg_mask = seg_slice > 0
    high_act_mask = att_slice >= high_act_threshold
    seg_count = int(seg_mask.sum())
    high_act_count = int(high_act_mask.sum())
    total = int(seg_slice.size)
    overlap = seg_mask & high_act_mask
    overlap_count = int(overlap.sum())
    return {
        "mean_activation_in_seg": float(att_slice[seg_mask].mean()) if seg_count > 0 else 0.0,
        "mean_activation_outside_seg": float(att_slice[~seg_mask].mean()) if (total - seg_count) > 0 else 0.0,
        "seg_coverage_by_high_act": overlap_count / seg_count if seg_count > 0 else 0.0,
        "high_act_coverage_by_seg": overlap_count / high_act_count if high_act_count > 0 else 0.0,
        "seg_voxel_count": seg_count,
        "high_act_voxel_count": high_act_count,
        "overlap_voxel_count": overlap_count,
        "total_voxels": total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Z-slice selection
# ═══════════════════════════════════════════════════════════════════════════════

def _get_z_slices(gt_seg_vol, D, default_z_slices=None):
    """Determine which z-slices to visualize.

    Priority:
      1. If GT segmentation exists: every slice containing GT voxels.
      2. If default_z_slices provided in config: use those.
      3. Fallback: single slice at z = D // 2 (center of volume).
    """
    if gt_seg_vol is not None:
        z_slices = [z for z in range(gt_seg_vol.shape[0]) if gt_seg_vol[z].max() > 0]
        if z_slices:
            return z_slices

    if default_z_slices is not None:
        return [z for z in default_z_slices if 0 <= z < D]

    return [min(D // 2, D - 1)]


# ═══════════════════════════════════════════════════════════════════════════════
# Figure rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _get_grid_shape(n):
    """Dynamic grid layout based on number of stages."""
    if n <= 0:
        return (1, 1)
    if n <= 3:
        return (1, n)
    if n <= 6:
        return (2, 3)
    if n <= 9:
        return (3, 3)
    return (4, math.ceil(n / 4))


def _make_grid_figure(
    pid, z_used, base_n, composite_slices, gt_seg_vol,
    method_label, target_cls, outcome_label, outcome_color,
    gt_label_val, pred_label_val, class_names, alpha, dpi, out_path,
    manuscript_mode=False, brain_mask_thr=0.05,
):
    """Save a grid figure for per-stage methods at one z-slice."""
    if manuscript_mode:
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif']

    stage_keys = list(composite_slices.keys())
    n_stages = len(stage_keys)
    rows, cols = _get_grid_shape(n_stages)
    overlay_alpha = 0.5 if manuscript_mode else alpha

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_stages == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    last_im = None
    for i, s in enumerate(stage_keys):
        sl_base, sl_att, z_s = composite_slices[s]
        _validate_slice_shapes(sl_base, sl_att,
                               gt_seg_vol[z_s] if gt_seg_vol is not None else None,
                               z_s, pid, method_label)
        axes[i].imshow(sl_base, cmap="gray")
        im = axes[i].imshow(sl_att, cmap="jet", alpha=overlay_alpha, vmin=0, vmax=1)
        last_im = im

        gt_seg_slice = gt_seg_vol[z_s] if gt_seg_vol is not None else None
        _draw_contours_on_ax(axes[i], gt_seg_slice, color='lime', linewidth=1.5)

        if manuscript_mode:
            axes[i].set_title(f"Stage {i}", fontsize=20, fontweight="bold")
        else:
            gt_act = _compute_gt_activation(sl_att, gt_seg_slice)
            cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            if gt_act is not None:
                _add_gt_activation_indicator(cbar, gt_act, color='lime')
            gt_str = f" (GT:{gt_act:.2f})" if gt_act is not None else ""
            axes[i].set_title(f"{s}  z={z_s}{gt_str}", fontsize=12, fontweight="bold")

        axes[i].axis("off")

    for j in range(n_stages, rows * cols):
        axes[j].axis("off")

    if manuscript_mode:
        fig.tight_layout(rect=[0, 0, 0.92, 1.0])
        if last_im is not None:
            cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
            fig.colorbar(last_im, cax=cbar_ax)
    else:
        gt_name = class_names.get(gt_label_val, "?")
        pred_name = class_names.get(pred_label_val, "?")
        gt_str = " [GT overlay]" if gt_seg_vol is not None else ""
        title = (
            f"{pid}    z={z_used}    {method_label}{gt_str}\n"
            f"GT: {gt_label_val} ({gt_name})  |  Pred: {pred_label_val} ({pred_name})"
        )
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
        fig.text(0.98, 0.99, outcome_label, fontsize=22, fontweight="bold",
                 color=outcome_color, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
        fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _make_single_map_figure(
    pid, z_used, base_slice, att_slice, gt_seg_slice,
    method_label, target_cls, outcome_label, outcome_color,
    gt_label_val, pred_label_val, class_names, alpha, dpi, out_path,
):
    """Save a single-panel figure for input-resolution methods."""
    _validate_slice_shapes(base_slice, att_slice, gt_seg_slice, z_used, pid, method_label)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(base_slice, cmap="gray")
    im = ax.imshow(att_slice, cmap="jet", alpha=alpha, vmin=0, vmax=1)

    _draw_contours_on_ax(ax, gt_seg_slice, color='lime', linewidth=1.5)
    gt_act = _compute_gt_activation(att_slice, gt_seg_slice)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if gt_act is not None:
        _add_gt_activation_indicator(cbar, gt_act, color='lime')

    ax.axis("off")

    gt_name = class_names.get(gt_label_val, "?")
    pred_name = class_names.get(pred_label_val, "?")
    gt_str = " [GT overlay]" if gt_seg_slice is not None else ""
    title = (
        f"{pid}    z={z_used}    {method_label}{gt_str}\n"
        f"GT: {gt_label_val} ({gt_name})  |  Pred: {pred_label_val} ({pred_name})"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.98, 0.99, outcome_label, fontsize=22, fontweight="bold",
             color=outcome_color, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Method computation
# ═══════════════════════════════════════════════════════════════════════════════

def _upsample_to(att, target_shape):
    """Upsample 3D attention volume to target spatial shape via trilinear."""
    if att.shape == target_shape:
        return att
    return zoom(att,
                [target_shape[i] / att.shape[i] for i in range(3)],
                order=1)


def _compute_notgradcam(model, x_batch, stages, stage_names, target_shape, pid):
    """Activation maps: mean channel activation per stage (no gradients)."""
    raw = {}
    for sname, stage_mod in zip(stage_names, stages):
        features = []
        def hook_fn(mod, inp, out, _f=features):
            _f.append(out)
        handle = stage_mod.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(x_batch)
        handle.remove()
        feat = features[0][0]
        att = feat.mean(dim=0).float().cpu().numpy()
        att = _upsample_to(att, target_shape)
        _validate_att_volume(att, target_shape, sname, "notgradcam", pid)
        raw[sname] = att
    return raw


def _compute_truegradcam(model, x_batch, stages, stage_names, target_shape,
                         target_cls, extract_logits_fn, pid):
    """True Grad-CAM via MONAI, with inversion fix."""
    from monai.visualize import GradCAM
    raw = {}
    for sname, stage_mod in zip(stage_names, stages):
        wrapped = _MonaiGradCamWrapper(model, stage_mod, extract_logits_fn)
        wrapped = wrapped.to(x_batch.device)
        wrapped.eval()
        tl_name = _get_module_name(wrapped, stage_mod)
        cam = GradCAM(nn_module=wrapped, target_layers=[tl_name])
        # MONAI's default_normalizer maps (min,max)->(1,0), inverting the map.
        # We undo this inversion.
        cam_t = cam(x_batch, class_idx=target_cls)
        if cam_t.ndim == 5:
            cam_t = cam_t[:, 0]
        att = cam_t[0].detach().cpu().numpy().astype(np.float32)
        att = 1.0 - att  # undo MONAI inversion
        att = _upsample_to(att, target_shape)
        _validate_att_volume(att, target_shape, sname, "truegradcam", pid)
        raw[sname] = att
    return raw


def _compute_guided_gradcam(model, x_batch, truegradcam_raw, stages, stage_names,
                            target_shape, target_cls, extract_logits_fn, pid):
    """Guided Grad-CAM: guided backprop saliency * Grad-CAM per stage."""
    device = x_batch.device
    with guided_backprop_context(model):
        x_input = x_batch.clone().requires_grad_(True)
        logits_gb = extract_logits_fn(model(x_input))
        logits_gb[0, target_cls].backward()
        guided_bp = x_input.grad[0].mean(dim=0).cpu().numpy()
    model.zero_grad()

    raw = {}
    for sname in stage_names:
        gc = truegradcam_raw[sname]
        gc_relu = np.maximum(gc, 0)
        raw[sname] = guided_bp * gc_relu
        _validate_att_volume(raw[sname], target_shape, sname, "guided_gradcam", pid)
    return raw


def _compute_layercam(model, x_batch, stages, stage_names, target_shape,
                      target_cls, extract_logits_fn, pid):
    """Layer-CAM: ReLU(grad * activation) summed over channels, per stage."""
    device = x_batch.device
    activations = {}
    gradients = {}
    handles = []

    for sname, stage_mod in zip(stage_names, stages):
        def fwd_hook(mod, inp, out, _s=sname):
            activations[_s] = out.detach()
        def bwd_hook(mod, grad_in, grad_out, _s=sname):
            gradients[_s] = grad_out[0].detach()
        handles.append(stage_mod.register_forward_hook(fwd_hook))
        handles.append(stage_mod.register_full_backward_hook(bwd_hook))

    x_lc = x_batch.clone().requires_grad_(True)
    logits_lc = extract_logits_fn(model(x_lc))
    logits_lc[0, target_cls].backward()
    for h in handles:
        h.remove()
    model.zero_grad()

    raw = {}
    for sname in stage_names:
        act = activations[sname]
        grad = gradients[sname]
        lc = F.relu(grad * act).sum(dim=1)[0].cpu().numpy()
        lc = _upsample_to(lc, target_shape)
        _validate_att_volume(lc, target_shape, sname, "layercam", pid)
        raw[sname] = lc
    return raw


def _compute_occlusion(model, x_batch, target_shape, target_cls,
                       extract_logits_fn, config, pid):
    """Occlusion Sensitivity via MONAI."""
    from monai.visualize import OcclusionSensitivity
    device = x_batch.device
    wrapped = _MonaiGradCamWrapper(model, None, extract_logits_fn).to(device)
    wrapped.eval()
    occ = OcclusionSensitivity(nn_module=wrapped, mask_size=config.occ_mask_size,
                               n_batch=config.occ_n_batch)
    occ_map, _ = occ(x_batch, class_idx=target_cls, overlap=config.occ_overlap)
    occ_arr = occ_map[0, 0].cpu().numpy()
    occ_arr = _upsample_to(occ_arr, target_shape)
    _validate_att_volume(occ_arr, target_shape, "input_resolution", "occlusion", pid)
    return occ_arr


def _compute_integrated_gradients(model, x_batch, target_shape, target_cls,
                                  extract_logits_fn, config, pid):
    """Integrated Gradients with optional Gaussian smoothing."""
    device = x_batch.device
    baseline = torch.zeros_like(x_batch)
    n_steps = config.ig_n_steps
    batch_size = config.ig_batch_size
    alphas = torch.linspace(0, 1, n_steps + 1, device=device)
    accumulated_grads = torch.zeros_like(x_batch, device="cpu")

    for i in range(0, n_steps + 1, batch_size):
        batch_alphas = alphas[i:i + batch_size]
        diff = x_batch - baseline
        x_interp = baseline + batch_alphas.view(-1, 1, 1, 1, 1) * diff
        x_interp = x_interp.detach().requires_grad_(True)
        logits_ig = extract_logits_fn(model(x_interp))
        logits_ig[:, target_cls].sum().backward()
        accumulated_grads += x_interp.grad.sum(dim=0, keepdim=True).cpu()
        model.zero_grad()

    ig_attr = (accumulated_grads / (n_steps + 1)) * (x_batch.cpu() - baseline.cpu())
    ig_arr = ig_attr[0].mean(dim=0).numpy()
    ig_arr = np.abs(ig_arr)

    if config.ig_smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        ig_arr = gaussian_filter(ig_arr, sigma=config.ig_smooth_sigma)

    _validate_att_volume(ig_arr, target_shape, "input_resolution",
                         "integrated_gradients", pid)
    return ig_arr


def _compute_integrated_gradcam(model, x_batch, stages, stage_names,
                                target_shape, target_cls, extract_logits_fn,
                                config, pid):
    """Integrated Grad-CAM (Sattarzadeh et al., ICASSP 2021).
    Path integral of Grad-CAM maps from black baseline to input."""
    device = x_batch.device
    n_steps = config.igc_n_steps
    baseline = torch.zeros_like(x_batch)

    # Pre-compute baseline activations (single forward pass)
    baseline_acts = {}
    baseline_hooks = []
    for sname, stage_mod in zip(stage_names, stages):
        acts = []
        def _hook(mod, inp, out, _a=acts):
            _a.append(out.detach())
        h = stage_mod.register_forward_hook(_hook)
        baseline_hooks.append((h, sname, acts))
    with torch.no_grad():
        model(baseline)
    for h, sname, acts in baseline_hooks:
        baseline_acts[sname] = acts[0]
        h.remove()

    integrated_raw = {sname: None for sname in stage_names}

    for t in range(1, n_steps + 1):
        alpha = t / n_steps
        x_interp = baseline + alpha * (x_batch - baseline)
        x_interp = x_interp.detach().requires_grad_(True)

        interp_acts = {}
        hooks = []
        for sname, stage_mod in zip(stage_names, stages):
            acts_list = []
            def _hook(mod, inp, out, _a=acts_list):
                out.retain_grad()
                _a.append(out)
            h = stage_mod.register_forward_hook(_hook)
            hooks.append((h, sname, acts_list))

        logits_igc = extract_logits_fn(model(x_interp))
        for h, sname, acts_list in hooks:
            interp_acts[sname] = acts_list[0]
            h.remove()

        logits_igc[0, target_cls].backward()

        for sname in stage_names:
            act = interp_acts[sname]
            delta_act = act - baseline_acts[sname].to(device)
            grad = act.grad
            if grad is None:
                continue
            cam_map = torch.sum(grad * delta_act, dim=1, keepdim=True)
            cam_map = torch.clamp(cam_map, min=0)
            cam_np = cam_map[0, 0].detach().cpu().numpy().astype(np.float32)
            if integrated_raw[sname] is None:
                integrated_raw[sname] = cam_np / n_steps
            else:
                integrated_raw[sname] += cam_np / n_steps

        model.zero_grad()

    for sname in stage_names:
        arr = integrated_raw[sname]
        if arr is not None:
            arr = _upsample_to(arr, target_shape)
            _validate_att_volume(arr, target_shape, sname, "integrated_gradcam", pid)
            integrated_raw[sname] = arr

    return integrated_raw


# ═══════════════════════════════════════════════════════════════════════════════
# Optional utility: GT segmentation alignment
# ═══════════════════════════════════════════════════════════════════════════════

def align_gt_seg(seg_path, target_shape, target_spacing,
                 raw_img_path=None, bbox=None):
    """Optional utility to align a raw GT segmentation NIfTI to the
    preprocessed input tensor space.

    This reproduces a common preprocessing pipeline:
      raw seg -> copy spacing from raw image -> bbox crop -> resample -> center crop

    Args:
        seg_path:        Path to GT segmentation NIfTI file.
        target_shape:    (D, H, W) of the preprocessed input tensor.
        target_spacing:  (sx, sy, sz) target voxel spacing used during preprocessing.
        raw_img_path:    Optional path to raw image NIfTI (to copy spacing from).
        bbox:            Optional bounding box [[z0,z1],[y0,y1],[x0,x1]] applied
                         during preprocessing. Format varies by pipeline:
                         - nnssl: from .pkl 'bbox_used_for_cropping'
                         - nnU-Net: from dataset_properties 'crop_to_nonzero'

    Returns:
        np.ndarray of shape target_shape, dtype uint8 (binary mask).
    """
    import SimpleITK as sitk

    seg_path = Path(seg_path)
    assert seg_path.exists(), f"GT seg not found: {seg_path}"

    gt_seg_img = sitk.ReadImage(str(seg_path))

    # Copy spacing from raw image if available (some segs have wrong headers)
    if raw_img_path is not None and Path(raw_img_path).exists():
        raw_img = sitk.ReadImage(str(raw_img_path))
        if gt_seg_img.GetSize() == raw_img.GetSize():
            gt_seg_img.SetSpacing(raw_img.GetSpacing())

    original_spacing = gt_seg_img.GetSpacing()
    gt_seg_arr = sitk.GetArrayFromImage(gt_seg_img)

    # Handle 4D segmentations (multiple labels -> collapse)
    if gt_seg_arr.ndim == 4:
        gt_seg_arr = gt_seg_arr.max(axis=0)

    assert gt_seg_arr.ndim == 3, (
        f"GT seg must be 3D after collapsing, got shape {gt_seg_arr.shape}")

    # Apply bbox crop if provided
    if bbox is not None:
        gt_seg_arr = gt_seg_arr[
            bbox[0][0]:bbox[0][1],
            bbox[1][0]:bbox[1][1],
            bbox[2][0]:bbox[2][1],
        ]

    # Resample to target spacing
    target_spacing = tuple(float(s) for s in target_spacing)
    if original_spacing != target_spacing:
        gt_seg_img2 = sitk.GetImageFromArray(gt_seg_arr)
        gt_seg_img2.SetSpacing(original_spacing)

        new_size = [
            int(round(gt_seg_arr.shape[2 - i] * original_spacing[i] / target_spacing[i]))
            for i in range(3)
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(gt_seg_img2.GetDirection())
        resampler.SetOutputOrigin(gt_seg_img2.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        gt_seg_arr = sitk.GetArrayFromImage(resampler.Execute(gt_seg_img2))

    # Center crop / pad to target shape
    D, H, W = target_shape
    if gt_seg_arr.shape != (D, H, W):
        result = np.zeros((D, H, W), dtype=gt_seg_arr.dtype)
        for i, (cur, tgt) in enumerate(zip(gt_seg_arr.shape, (D, H, W))):
            pass  # handled below
        starts = [(gt_seg_arr.shape[i] - target_shape[i]) // 2 for i in range(3)]
        src_slices, dst_slices = [], []
        for i in range(3):
            if starts[i] >= 0:
                ss, se = starts[i], starts[i] + target_shape[i]
                ds, de = 0, target_shape[i]
            else:
                ss, se = 0, gt_seg_arr.shape[i]
                ds = -starts[i]
                de = ds + gt_seg_arr.shape[i]
            if se > gt_seg_arr.shape[i]:
                excess = se - gt_seg_arr.shape[i]
                se = gt_seg_arr.shape[i]
                de -= excess
            src_slices.append(slice(ss, se))
            dst_slices.append(slice(ds, de))
        result[dst_slices[0], dst_slices[1], dst_slices[2]] = \
            gt_seg_arr[src_slices[0], src_slices[1], src_slices[2]]
        gt_seg_arr = result

    gt_seg_arr = (gt_seg_arr > 0).astype(np.uint8)

    assert gt_seg_arr.shape == target_shape, (
        f"align_gt_seg: final shape {gt_seg_arr.shape} != target {target_shape}")
    return gt_seg_arr


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_gradcam(model, patients, config):
    """Generate Grad-CAM visualizations for a list of patients.

    Args:
        model:    PyTorch model (nn.Module). Loaded and ready (weights loaded).
                  Will be moved to GPU and set to eval mode.
        patients: List of dicts, each with:
                    - "patient_id": str (required)
                    - "input": Tensor of shape (C, D, H, W) (required)
                    - "gt_label": int (optional, for TP/TN/FP/FN badge)
                    - "pred_label": int (optional, inferred from model if missing)
                    - "gt_seg": np.ndarray of shape (D, H, W) (optional, pre-aligned)
                    - "z_slices": list[int] (optional, override slice selection)
        config:   GradcamConfig instance.

    Returns:
        List of metadata dicts (one per patient), also saved as JSON files.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Setup model once
    model = model.to(device)
    model.eval()

    # Get encoder stages once
    stages = config.get_stages_fn(model)
    num_stages = len(stages)
    stage_names = [f"stage{i}" for i in range(num_stages)]
    print(f"Encoder stages: {num_stages}")

    # Create output dirs
    out_base = Path(config.out_dir)
    out_dirs = {}
    for m in config.methods:
        d = out_base / METHOD_SUBDIRS[m]
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[m] = d

    all_patient_metadata = []

    for p_idx, patient in enumerate(patients):
        pid = patient["patient_id"]
        x = patient["input"]
        gt_label = patient.get("gt_label")
        pred_label = patient.get("pred_label")
        gt_seg_vol = patient.get("gt_seg")
        forced_z_slices = patient.get("z_slices")

        print(f"\n[{p_idx + 1}/{len(patients)}] Patient: {pid}")

        # ── Validate input ──
        _validate_input(x, pid)
        C, D, H, W = x.shape
        target_shape = (D, H, W)

        if gt_seg_vol is not None:
            _validate_gt_seg(gt_seg_vol, x.shape, pid)

        # ── Prepare base volume ──
        x = x.float()
        x_batch = x.unsqueeze(0).to(device)
        vol_np = x[0].cpu().numpy() if C == 1 else x.mean(dim=0).cpu().numpy()
        base = vol_np.astype(np.float32)
        bmin, bmax = float(base.min()), float(base.max())
        base_n = (base - bmin) / (bmax - bmin) if bmax > bmin else np.zeros_like(base)
        base_n = np.clip(base_n, 0.0, 1.0)

        # ── Infer prediction if missing ──
        if pred_label is None:
            with torch.no_grad():
                logits_check = config.extract_logits_fn(model(x_batch))
                if logits_check.shape[-1] == 1:
                    pred_label = 1 if logits_check[0, 0].item() > 0 else 0
                else:
                    pred_label = int(torch.argmax(logits_check[0]).item())
            print(f"  Prediction inferred from model: {pred_label}")

        outcome_label, outcome_color = _classify_outcome(gt_label, pred_label)
        print(f"  GT={gt_label}  Pred={pred_label}  -> {outcome_label}")

        # ── Determine target class ──
        with torch.no_grad():
            logits0 = config.extract_logits_fn(model(x_batch))
        target_cls, _ = _pick_target_class(logits0, config.target_class)
        print(f"  Target class: {target_cls}")

        # ── Z-slices ──
        if forced_z_slices is not None:
            z_slices = forced_z_slices
        else:
            z_slices = _get_z_slices(gt_seg_vol, D, config.default_z_slices)
        for z in z_slices:
            _validate_z_in_range(z, D, pid)
        print(f"  Z-slices: {len(z_slices)} [{min(z_slices)}..{max(z_slices)}]")

        # ── Compute methods ──
        pclip = config.percentile_clip
        thr = config.brain_mask_threshold

        notgradcam_normed = {}
        if "notgradcam" in config.methods:
            print("  Computing NotGradCam...")
            raw = _compute_notgradcam(model, x_batch, stages, stage_names,
                                      target_shape, pid)
            notgradcam_normed = _normalize_per_stage(raw, base_n, pclip, thr)

        truegradcam_raw = {}
        truegradcam_normed = {}
        if any(m in config.methods for m in ("truegradcam", "guided_gradcam",
                                             "integrated_gradcam")):
            print("  Computing TrueGradCam...")
            truegradcam_raw = _compute_truegradcam(
                model, x_batch, stages, stage_names, target_shape,
                target_cls, config.extract_logits_fn, pid)
            truegradcam_normed = _normalize_per_stage(truegradcam_raw, base_n,
                                                      pclip, thr)

        guided_gradcam_normed = {}
        if "guided_gradcam" in config.methods:
            print("  Computing Guided Grad-CAM...")
            raw = _compute_guided_gradcam(
                model, x_batch, truegradcam_raw, stages, stage_names,
                target_shape, target_cls, config.extract_logits_fn, pid)
            guided_gradcam_normed = _normalize_per_stage(raw, base_n, pclip, thr)

        layercam_normed = {}
        if "layercam" in config.methods:
            print("  Computing Layer-CAM...")
            raw = _compute_layercam(
                model, x_batch, stages, stage_names, target_shape,
                target_cls, config.extract_logits_fn, pid)
            layercam_normed = _normalize_per_stage(raw, base_n, pclip, thr)

        occlusion_normed = None
        if "occlusion" in config.methods:
            print(f"  Computing Occlusion (mask={config.occ_mask_size}, "
                  f"overlap={config.occ_overlap})...")
            occ_arr = _compute_occlusion(
                model, x_batch, target_shape, target_cls,
                config.extract_logits_fn, config, pid)
            occlusion_normed = _normalize_volume(occ_arr, base_n, pclip, thr)

        intgrad_normed = None
        if "integrated_gradients" in config.methods:
            print(f"  Computing Integrated Gradients (steps={config.ig_n_steps})...")
            ig_arr = _compute_integrated_gradients(
                model, x_batch, target_shape, target_cls,
                config.extract_logits_fn, config, pid)
            intgrad_normed = _normalize_volume(ig_arr, base_n, pclip, thr)

        integrated_gradcam_normed = {}
        if "integrated_gradcam" in config.methods:
            print(f"  Computing Integrated Grad-CAM (steps={config.igc_n_steps})...")
            raw = _compute_integrated_gradcam(
                model, x_batch, stages, stage_names, target_shape,
                target_cls, config.extract_logits_fn, config, pid)
            integrated_gradcam_normed = _normalize_per_stage(raw, base_n, pclip, thr)

        # ── Generate figures ──
        pid_sanitized = _sanitize_for_path(pid)
        gt_suffix = "_GT" if gt_seg_vol is not None else ""
        saved_count = 0
        slice_metadata = []

        for z in z_slices:
            gt_seg_z = gt_seg_vol[z] if gt_seg_vol is not None else None

            # -- NotGradCam --
            if "notgradcam" in config.methods:
                slices = {}
                for sname, att_n in notgradcam_normed.items():
                    slices[sname] = (base_n[z].copy(), att_n[z].copy(), z)
                path = out_dirs["notgradcam"] / (
                    f"{pid_sanitized}_attention_z{z}_{outcome_label}{gt_suffix}.png")
                _make_grid_figure(
                    pid=pid, z_used=z, base_n=base_n, composite_slices=slices,
                    gt_seg_vol=gt_seg_vol, method_label="Activation Map (NotGradCam)",
                    target_cls=target_cls, outcome_label=outcome_label,
                    outcome_color=outcome_color, gt_label_val=gt_label,
                    pred_label_val=pred_label, class_names=config.class_names,
                    alpha=config.alpha, dpi=config.dpi, out_path=path,
                    manuscript_mode=config.manuscript_mode,
                    brain_mask_thr=thr,
                )
                saved_count += 1
                for sname, att_n in notgradcam_normed.items():
                    m = _compute_slice_metrics(att_n[z], gt_seg_z)
                    if m:
                        m.update({"method": "notgradcam", "stage": sname,
                                  "z": int(z), "figure": str(path.name)})
                        slice_metadata.append(m)

            # -- TrueGradCam --
            if "truegradcam" in config.methods:
                slices = {}
                for sname, att_n in truegradcam_normed.items():
                    att_z = _brain_mask_slice(base_n[z], att_n[z], thr)
                    slices[sname] = (base_n[z].copy(), att_z.copy(), z)
                path = out_dirs["truegradcam"] / (
                    f"{pid_sanitized}_gradcam_z{z}_class{target_cls}_{outcome_label}{gt_suffix}.png")
                _make_grid_figure(
                    pid=pid, z_used=z, base_n=base_n, composite_slices=slices,
                    gt_seg_vol=gt_seg_vol, method_label="True Grad-CAM",
                    target_cls=target_cls, outcome_label=outcome_label,
                    outcome_color=outcome_color, gt_label_val=gt_label,
                    pred_label_val=pred_label, class_names=config.class_names,
                    alpha=config.alpha, dpi=config.dpi, out_path=path,
                    manuscript_mode=config.manuscript_mode,
                    brain_mask_thr=thr,
                )
                saved_count += 1
                for sname, att_n in truegradcam_normed.items():
                    att_z = _brain_mask_slice(base_n[z], att_n[z], thr)
                    m = _compute_slice_metrics(att_z, gt_seg_z)
                    if m:
                        m.update({"method": "truegradcam", "stage": sname,
                                  "z": int(z), "figure": str(path.name)})
                        slice_metadata.append(m)

            # -- Guided Grad-CAM --
            if "guided_gradcam" in config.methods:
                slices = {}
                for sname, att_n in guided_gradcam_normed.items():
                    att_z = _brain_mask_slice(base_n[z], att_n[z], thr)
                    slices[sname] = (base_n[z].copy(), att_z.copy(), z)
                path = out_dirs["guided_gradcam"] / (
                    f"{pid_sanitized}_guidedgradcam_z{z}_class{target_cls}_{outcome_label}{gt_suffix}.png")
                _make_grid_figure(
                    pid=pid, z_used=z, base_n=base_n, composite_slices=slices,
                    gt_seg_vol=gt_seg_vol, method_label="Guided Grad-CAM",
                    target_cls=target_cls, outcome_label=outcome_label,
                    outcome_color=outcome_color, gt_label_val=gt_label,
                    pred_label_val=pred_label, class_names=config.class_names,
                    alpha=config.alpha, dpi=config.dpi, out_path=path,
                    manuscript_mode=config.manuscript_mode,
                    brain_mask_thr=thr,
                )
                saved_count += 1
                for sname, att_n in guided_gradcam_normed.items():
                    att_z = _brain_mask_slice(base_n[z], att_n[z], thr)
                    m = _compute_slice_metrics(att_z, gt_seg_z)
                    if m:
                        m.update({"method": "guided_gradcam", "stage": sname,
                                  "z": int(z), "figure": str(path.name)})
                        slice_metadata.append(m)

            # -- Layer-CAM --
            if "layercam" in config.methods:
                slices = {}
                for sname, att_n in layercam_normed.items():
                    att_z = _brain_mask_slice(base_n[z], att_n[z], thr)
                    slices[sname] = (base_n[z].copy(), att_z.copy(), z)
                path = out_dirs["layercam"] / (
                    f"{pid_sanitized}_layercam_z{z}_class{target_cls}_{outcome_label}{gt_suffix}.png")
                _make_grid_figure(
                    pid=pid, z_used=z, base_n=base_n, composite_slices=slices,
                    gt_seg_vol=gt_seg_vol, method_label="Layer-CAM",
                    target_cls=target_cls, outcome_label=outcome_label,
                    outcome_color=outcome_color, gt_label_val=gt_label,
                    pred_label_val=pred_label, class_names=config.class_names,
                    alpha=config.alpha, dpi=config.dpi, out_path=path,
                    manuscript_mode=config.manuscript_mode,
                    brain_mask_thr=thr,
                )
                saved_count += 1
                for sname, att_n in layercam_normed.items():
                    att_z = _brain_mask_slice(base_n[z], att_n[z], thr)
                    m = _compute_slice_metrics(att_z, gt_seg_z)
                    if m:
                        m.update({"method": "layercam", "stage": sname,
                                  "z": int(z), "figure": str(path.name)})
                        slice_metadata.append(m)

            # -- Occlusion --
            if "occlusion" in config.methods and occlusion_normed is not None:
                occ_z = _brain_mask_slice(base_n[z], occlusion_normed[z], thr)
                path = out_dirs["occlusion"] / (
                    f"{pid_sanitized}_occlusion_z{z}_class{target_cls}_{outcome_label}{gt_suffix}.png")
                _make_single_map_figure(
                    pid=pid, z_used=z, base_slice=base_n[z].copy(),
                    att_slice=occ_z.copy(), gt_seg_slice=gt_seg_z,
                    method_label="Occlusion Sensitivity",
                    target_cls=target_cls, outcome_label=outcome_label,
                    outcome_color=outcome_color, gt_label_val=gt_label,
                    pred_label_val=pred_label, class_names=config.class_names,
                    alpha=config.alpha, dpi=config.dpi, out_path=path,
                )
                saved_count += 1
                m = _compute_slice_metrics(occ_z, gt_seg_z)
                if m:
                    m.update({"method": "occlusion", "stage": "input_resolution",
                              "z": int(z), "figure": str(path.name)})
                    slice_metadata.append(m)

            # -- Integrated Gradients --
            if "integrated_gradients" in config.methods and intgrad_normed is not None:
                ig_z = _brain_mask_slice(base_n[z], intgrad_normed[z], thr)
                path = out_dirs["integrated_gradients"] / (
                    f"{pid_sanitized}_intgrad_z{z}_class{target_cls}_{outcome_label}{gt_suffix}.png")
                _make_single_map_figure(
                    pid=pid, z_used=z, base_slice=base_n[z].copy(),
                    att_slice=ig_z.copy(), gt_seg_slice=gt_seg_z,
                    method_label="Integrated Gradients",
                    target_cls=target_cls, outcome_label=outcome_label,
                    outcome_color=outcome_color, gt_label_val=gt_label,
                    pred_label_val=pred_label, class_names=config.class_names,
                    alpha=config.alpha, dpi=config.dpi, out_path=path,
                )
                saved_count += 1
                m = _compute_slice_metrics(ig_z, gt_seg_z)
                if m:
                    m.update({"method": "integrated_gradients",
                              "stage": "input_resolution",
                              "z": int(z), "figure": str(path.name)})
                    slice_metadata.append(m)

            # -- Integrated Grad-CAM --
            if "integrated_gradcam" in config.methods and integrated_gradcam_normed:
                slices = {}
                for sname, att_n in integrated_gradcam_normed.items():
                    att_z = _brain_mask_slice(base_n[z], att_n[z], thr)
                    slices[sname] = (base_n[z].copy(), att_z.copy(), z)
                path = out_dirs["integrated_gradcam"] / (
                    f"{pid_sanitized}_intgradcam_z{z}_class{target_cls}_{outcome_label}{gt_suffix}.png")
                _make_grid_figure(
                    pid=pid, z_used=z, base_n=base_n, composite_slices=slices,
                    gt_seg_vol=gt_seg_vol, method_label="Integrated Grad-CAM",
                    target_cls=target_cls, outcome_label=outcome_label,
                    outcome_color=outcome_color, gt_label_val=gt_label,
                    pred_label_val=pred_label, class_names=config.class_names,
                    alpha=config.alpha, dpi=config.dpi, out_path=path,
                    manuscript_mode=config.manuscript_mode,
                    brain_mask_thr=thr,
                )
                saved_count += 1
                for sname, att_n in integrated_gradcam_normed.items():
                    att_z = _brain_mask_slice(base_n[z], att_n[z], thr)
                    m = _compute_slice_metrics(att_z, gt_seg_z)
                    if m:
                        m.update({"method": "integrated_gradcam", "stage": sname,
                                  "z": int(z), "figure": str(path.name)})
                        slice_metadata.append(m)

        # ── Save metadata JSON ──
        patient_meta = {
            "patient_id": pid,
            "outcome": outcome_label,
            "gt_label": int(gt_label) if gt_label is not None else None,
            "pred_label": int(pred_label) if pred_label is not None else None,
            "target_class": int(target_cls),
            "n_slices": len(z_slices),
            "z_slices": [int(z) for z in z_slices],
            "has_gt_segmentation": gt_seg_vol is not None,
            "slices": slice_metadata,
        }
        if slice_metadata:
            meta_path = out_base / f"{pid_sanitized}_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(patient_meta, f, indent=2)
            print(f"  Metadata: {meta_path}")

        all_patient_metadata.append(patient_meta)
        print(f"  Saved {saved_count} figures for {pid}")

    print(f"\nDone. {len(patients)} patients, output: {out_base}")
    return all_patient_metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Usage example
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Usage examples showing how to integrate gradcam3d_viz with different projects.

    Example 1: Generic usage with any 3D classification model
    Example 2: nnU-Net style model
    Example 3: Using align_gt_seg() utility for raw segmentation files
    """

    # =====================================================================
    # EXAMPLE 1: Generic 3D classification model
    # =====================================================================
    #
    # Assumes you have:
    #   - A trained 3D classification model with an encoder that has "stages"
    #   - Preprocessed input tensors (C, D, H, W) ready for the model
    #   - Optionally: GT segmentation volumes aligned to the input space
    #
    # Steps:
    #   1. Load your model
    #   2. Define get_stages_fn to extract encoder stages
    #   3. Build patient list
    #   4. Call run_gradcam()

    print("=" * 60)
    print("EXAMPLE 1: Generic 3D classification model")
    print("=" * 60)

    # --- Step 1: Load your model ---
    # Replace with your actual model loading code:
    #
    #   import torch
    #   model = MyModel(num_classes=2)
    #   ckpt = torch.load("path/to/best_checkpoint.ckpt", map_location="cpu")
    #   model.load_state_dict(ckpt["state_dict"])

    # --- Step 2: Define how to access encoder stages ---
    # This is the ONLY model-specific code you need to write.
    # It must return a list of nn.Module objects (the encoder stages).
    #
    # Common patterns:
    #
    #   # If encoder has a .stages attribute (list/ModuleList):
    #   get_stages = lambda m: list(m.encoder.stages)
    #
    #   # If encoder stages are named attributes:
    #   get_stages = lambda m: [m.layer1, m.layer2, m.layer3, m.layer4]
    #
    #   # If model is a ResNet with layer1-4:
    #   get_stages = lambda m: [m.conv1, m.layer1, m.layer2, m.layer3, m.layer4]
    #
    #   # nnssl / SSL3D ResidualEncoder:
    #   get_stages = lambda m: list(m.encoder.res_unet.encoder.stages)

    # --- Step 3: Build patient list ---
    # Each patient is a dict. Only "patient_id" and "input" are required.
    #
    #   patients = []
    #   for case_id in ["patient_001", "patient_002", "patient_003"]:
    #       # Load your preprocessed tensor however you do it:
    #       x = torch.load(f"preprocessed/{case_id}.pt")  # shape: (C, D, H, W)
    #
    #       # Optionally load pre-aligned GT segmentation:
    #       seg = np.load(f"segs/{case_id}_seg.npy")  # shape: (D, H, W), same space as x
    #
    #       patients.append({
    #           "patient_id": case_id,
    #           "input": x,               # (C, D, H, W) tensor
    #           "gt_label": 1,            # optional: ground truth class
    #           "pred_label": None,       # optional: None = infer from model
    #           "gt_seg": seg,            # optional: (D, H, W) numpy array, pre-aligned
    #           "z_slices": None,         # optional: force specific slices
    #       })

    # --- Step 4: Configure and run ---
    #
    #   config = GradcamConfig(
    #       get_stages_fn=get_stages,
    #       class_names={0: "Healthy", 1: "Disease"},
    #       methods=["notgradcam", "truegradcam"],
    #       out_dir="./gradcam_output",
    #   )
    #   metadata = run_gradcam(model, patients, config)

    # =====================================================================
    # EXAMPLE 2: nnU-Net encoder used for classification
    # =====================================================================
    #
    # nnU-Net models have a different encoder structure than nnssl.
    # The key difference is how you access the encoder stages.

    print("\n" + "=" * 60)
    print("EXAMPLE 2: nnU-Net style model")
    print("=" * 60)

    # --- nnU-Net encoder stage access ---
    # nnU-Net PlainConvEncoder stores stages in self.stages (nn.ModuleList)
    # and initial convolution in self.stem.
    #
    # Option A: All stages including stem
    #   get_stages = lambda m: [m.network.encoder.stem] + list(m.network.encoder.stages)
    #
    # Option B: Just the downsampling stages (no stem)
    #   get_stages = lambda m: list(m.network.encoder.stages)
    #
    # If your nnU-Net wrapper names things differently, adjust accordingly.
    # The point is: return a list of nn.Module, one per encoder resolution.

    # --- nnU-Net data loading ---
    # nnU-Net preprocessed data is stored as .npz files:
    #
    #   import numpy as np
    #   data = np.load("nnUNet_preprocessed/Dataset001/case_0001.npz")
    #   x = torch.from_numpy(data["data"])  # (C, D, H, W)
    #
    # GT segmentation can come from the same .npz:
    #   seg = data["seg"][0]  # (D, H, W)
    #
    # Since both come from the same preprocessed space, they are already aligned!

    # --- nnU-Net spacing from plans ---
    # If you need align_gt_seg() for raw NIfTI segmentations:
    #
    #   import json
    #   plans = json.load(open("nnUNet_preprocessed/Dataset001/nnUNetPlans.json"))
    #   target_spacing = plans["configurations"]["3d_fullres"]["spacing"]

    # --- Full nnU-Net example ---
    #
    #   model = load_nnunet_classifier("path/to/fold_0/checkpoint_best.pth")
    #
    #   get_stages = lambda m: list(m.network.encoder.stages)
    #
    #   patients = []
    #   for npz_path in Path("nnUNet_preprocessed/Dataset001/").glob("*.npz"):
    #       data = np.load(npz_path)
    #       patients.append({
    #           "patient_id": npz_path.stem,
    #           "input": torch.from_numpy(data["data"]).float(),
    #           "gt_label": int(labels_dict[npz_path.stem]),
    #           "gt_seg": (data["seg"][0] > 0).astype(np.uint8),  # already aligned
    #       })
    #
    #   config = GradcamConfig(
    #       get_stages_fn=get_stages,
    #       class_names={0: "Benign", 1: "Malignant"},
    #       methods=["notgradcam", "truegradcam", "layercam"],
    #       target_class=1,
    #       out_dir="./nnunet_gradcam_output",
    #       # Adjust if your nnU-Net model outputs differently:
    #       # extract_logits_fn=lambda out: out["logits"],
    #   )
    #   metadata = run_gradcam(model, patients, config)

    # =====================================================================
    # EXAMPLE 3: Using align_gt_seg() for raw NIfTI segmentations
    # =====================================================================
    #
    # If you have raw (unprocessed) segmentation NIfTI files and know your
    # preprocessing parameters, use align_gt_seg() to align them.

    print("\n" + "=" * 60)
    print("EXAMPLE 3: align_gt_seg() utility")
    print("=" * 60)

    # --- align_gt_seg() handles the full pipeline ---
    #   raw seg -> copy spacing from raw image -> bbox crop -> resample -> center crop
    #
    # You need to provide:
    #   - target_shape: (D, H, W) matching your preprocessed input tensor
    #   - target_spacing: the spacing your preprocessing pipeline resampled to
    #   - bbox (optional): bounding box applied during preprocessing
    #
    # Example for nnssl (1mm isotropic, 160^3, with bbox from .pkl):
    #
    #   import pickle
    #   props = pickle.load(open("preprocessed/patient_001/ses-DEFAULT/scan.pkl", "rb"))
    #   bbox = props["bbox_used_for_cropping"]
    #
    #   seg = align_gt_seg(
    #       seg_path="raw_segs/patient_001_seg.nii.gz",
    #       target_shape=(160, 160, 160),
    #       target_spacing=(1.0, 1.0, 1.0),
    #       raw_img_path="raw_images/patient_001.nii.gz",  # for spacing fix
    #       bbox=bbox,
    #   )
    #   assert seg.shape == (160, 160, 160)  # guaranteed by align_gt_seg
    #
    # Example for nnU-Net (dataset-specific spacing, no fixed crop size):
    #
    #   import json
    #   plans = json.load(open("nnUNetPlans.json"))
    #   spacing = plans["configurations"]["3d_fullres"]["spacing"]
    #   # nnU-Net bbox comes from dataset_properties or dataset_fingerprint
    #   props = json.load(open("dataset_fingerprint.json"))
    #   # Note: check your nnU-Net version for exact key names
    #
    #   input_tensor = torch.load("preprocessed/case_0001.pt")  # (C, D, H, W)
    #   target_shape = tuple(input_tensor.shape[1:])  # (D, H, W)
    #
    #   seg = align_gt_seg(
    #       seg_path="raw_segs/case_0001_seg.nii.gz",
    #       target_shape=target_shape,
    #       target_spacing=tuple(spacing),
    #       bbox=bbox_from_properties,  # or None if no bbox crop was applied
    #   )
    #   assert seg.shape == target_shape

    # =====================================================================
    # EXAMPLE 4: All 7 methods with custom logits extractor
    # =====================================================================

    print("\n" + "=" * 60)
    print("EXAMPLE 4: All methods + custom extract_logits_fn")
    print("=" * 60)

    #   # If your model returns a dict instead of a tensor:
    #   def my_extract_logits(model_output):
    #       return model_output["classification_logits"]
    #
    #   config = GradcamConfig(
    #       get_stages_fn=lambda m: list(m.encoder.stages),
    #       extract_logits_fn=my_extract_logits,
    #       class_names={0: "Grade I", 1: "Grade II", 2: "Grade III", 3: "Grade IV"},
    #       target_class=-1,  # use predicted class
    #       methods=["all"],  # all 7 methods
    #       out_dir="./all_methods_output",
    #       # Tune expensive methods:
    #       occ_mask_size=16,
    #       ig_n_steps=30,
    #       ig_smooth_sigma=0,   # disable smoothing
    #       igc_n_steps=30,
    #   )
    #   metadata = run_gradcam(model, patients, config)

    print("\nAll examples shown as comments. Uncomment and adapt for your project.")
    print("See GradcamConfig docstring for all available options.")


if __name__ == "__main__":
    main()
