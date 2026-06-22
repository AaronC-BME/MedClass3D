"""MedClass3D-specific loading, patient assembly, and gradcam execution.

The model-agnostic compute/rendering lives in ``gradcam3d_viz`` /
``multimodal_gradcam``. This module adapts the SSL3D drivers
(``run_gradcam_testset_5folds.py`` / ``run_gradcam_external_cohort_5folds.py``)
to this repo:

  * checkpoints under ``<run_dir>/folds/<k>/*.ckpt`` (best by ``Val_AUROC=`` /
    legacy ``Val_acc=`` in the filename),
  * config at ``<run_dir>/Configs/config.yaml`` (rebuilt with ``pretrained=False``),
  * ``ResEncoder_Classifier`` whose ``forward`` returns logits and whose encoder
    stages live at ``model.encoder.res_unet.encoder.stages``,
  * the ``Class_Data`` dataset (``(img, label)`` samples; channel 0 = image,
    channel 1 = mask when ``use_mask``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .gradcam3d_viz import GradcamConfig
from .multimodal_gradcam import MultiModalConfig, run_multimodal_gradcam


# ─────────────────────────────────────────────────────────────────────────────
# Encoder stage discovery (ported from run_gradcam_testset_5folds.py)
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_STAGE_PATHS = [
    "encoder.res_unet.encoder.stages",      # MedClass3D ResEncoder_Classifier
    "encoder.stages",
    "network.encoder.stages",
    "backbone.encoder.stages",
    "model.encoder.res_unet.encoder.stages",
    "model.encoder.stages",
]


def _resolve_attr_path(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None, False
        cur = getattr(cur, part)
    return cur, True


def _find_deepest_module_list(model: nn.Module) -> Optional[List[nn.Module]]:
    """Auto-discover encoder stages, scoring encoder-side ModuleLists up."""
    candidates = []
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.ModuleList, nn.Sequential)) or len(mod) <= 1:
            continue
        score = 0
        if ".decoder." in f".{name}." or name.endswith(".decoder"):
            score -= 100
        if "decoder.stages" in name:
            score -= 50
        if ".head" in name or ".classifier" in name:
            score -= 100
        if (".encoder." in f".{name}." or name.endswith(".encoder")
                or name.startswith("encoder.")):
            if ".decoder." not in f".{name}.":
                score += 10
        if name.endswith("stages"):
            score += 5
        score += name.count(".")
        candidates.append((score, name, mod))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    best_score, best_name, best_mod = candidates[0]
    print(f"  [auto] encoder stages -> model.{best_name} "
          f"(score={best_score:+d}, len={len(best_mod)})")
    return list(best_mod)


def get_stages_for_model(model: nn.Module) -> List[nn.Module]:
    for path in KNOWN_STAGE_PATHS:
        val, ok = _resolve_attr_path(model, path)
        if ok and isinstance(val, (nn.ModuleList, nn.Sequential, list)) and len(val) > 1:
            print(f"  Encoder stages located at: model.{path} ({len(val)} stages)")
            return list(val)
    print("  Known stage paths failed — falling back to auto-discovery.")
    found = _find_deepest_module_list(model)
    if found is None:
        raise RuntimeError(
            "Could not locate encoder stages. Inspect print(model) and add the "
            "dotted path to KNOWN_STAGE_PATHS.")
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery (MedClass3D layout: <run_dir>/folds/<k>/*.ckpt)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ckpt_metric(name: str) -> float:
    """Monitored metric from the filename. New runs write ``Val_AUROC=``; older
    ones used ``Val_acc=`` (matches scripts/predict_test.py)."""
    for token in ("Val_AUROC=", "Val_acc="):
        if token in name:
            try:
                return float(name.split(token)[1].split(".ckpt")[0])
            except (IndexError, ValueError):
                continue
    return float("-inf")


def find_best_ckpt_for_fold(folds_dir: Path, fold_idx: int,
                            prefer_last: bool = False) -> Optional[dict]:
    fold_dir = Path(folds_dir) / str(fold_idx)
    if not fold_dir.is_dir():
        print(f"  [fold {fold_idx}] directory not found: {fold_dir}")
        return None
    ckpts = list(fold_dir.glob("*.ckpt"))
    if not ckpts:
        print(f"  [fold {fold_idx}] no .ckpt files in {fold_dir}")
        return None
    last = [p for p in ckpts if p.name == "last.ckpt"]
    not_last = [p for p in ckpts if p.name != "last.ckpt"]
    if prefer_last and last:
        chosen = last[0]
    elif not_last:
        chosen = max(not_last, key=lambda p: _parse_ckpt_metric(p.name))
        if _parse_ckpt_metric(chosen.name) == float("-inf"):
            chosen = last[0] if last else not_last[0]
    else:
        chosen = ckpts[0]
    metric = _parse_ckpt_metric(chosen.name)
    print(f"  [fold {fold_idx}] checkpoint: {chosen.name} "
          f"(metric={'%.4f' % metric if metric != float('-inf') else 'n/a'})")
    return {"path": chosen, "filename": chosen.name, "metric": metric}


# ─────────────────────────────────────────────────────────────────────────────
# Logits + output naming + reorg (ported from run_gradcam_testset_5folds.py)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_logits(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)) and torch.is_tensor(out[0]):
        return out[0]
    if isinstance(out, dict):
        for k in ("logits", "pred", "y_hat", "outputs", "out"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
    raise RuntimeError(f"Cannot extract logits from {type(out)}")


_METHOD_DISPLAY_NAMES = {
    "layercam": "layerCAM", "truegradcam": "gradCAM", "notgradcam": "notGradCAM",
    "guided_gradcam": "guidedGradCAM", "occlusion": "occlusion",
    "integrated_gradients": "integratedGrad", "integrated_gradcam": "integratedGradCAM",
}


def derive_method_subdir(methods: List[str]) -> str:
    seen = []
    for m in methods:
        name = _METHOD_DISPLAY_NAMES.get(m.lower(), m.lower())
        if name not in seen:
            seen.append(name)
    return "-".join(seen) if seen else "gradcam"


def _id_variants(pid: str) -> List[str]:
    variants = [pid]
    stem = pid
    for _ in range(3):
        for suf in (".nii.gz", ".nii", ".mha", ".mhd", ".b2nd", ".gz"):
            if stem.lower().endswith(suf):
                stem = stem[: -len(suf)]
                if stem and stem not in variants:
                    variants.append(stem)
                break
        else:
            break
    return variants


def reorganize_to_per_image_subdirs(root_dir: Path, patient_ids: List[str]) -> None:
    """Move method-named flat outputs under ``root_dir`` into one subdir per image."""
    if not patient_ids:
        return
    root_dir = Path(root_dir)
    unique_ids = sorted(set(patient_ids), key=len, reverse=True)
    pairs: List[Tuple[str, str]] = []
    for pid in unique_ids:
        for v in _id_variants(pid):
            pairs.append((pid, v))
    pairs.sort(key=lambda t: len(t[1]), reverse=True)

    exts = {".png", ".npy", ".npz", ".json", ".pkl", ".pt"}
    files = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    moved = 0
    for src in files:
        matched = next((cid for cid, var in pairs if src.name.startswith(var)), None)
        if matched is None:
            continue
        dest_dir = root_dir / matched
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if src.resolve() == dest.resolve():
            continue
        if dest.exists():
            i = 1
            while (dest_dir / f"{dest.stem}__{i}{dest.suffix}").exists():
                i += 1
            dest = dest_dir / f"{dest.stem}__{i}{dest.suffix}"
        src.rename(dest)
        moved += 1
    for d in sorted([p for p in root_dir.rglob("*") if p.is_dir()],
                    key=lambda p: len(p.parts), reverse=True):
        if d.parent == root_dir and d.name in set(unique_ids):
            continue
        try:
            d.rmdir()
        except OSError:
            pass
    print(f"    Reorganized {moved} file(s) into per-image subdirs")


# ─────────────────────────────────────────────────────────────────────────────
# Per-class confusion bucketing (ported)
# ─────────────────────────────────────────────────────────────────────────────

def bucket_patients_by_confusion_cell(patients, class_names, max_per_cell):
    buckets: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for p in patients:
        if p["gt_label"] is None:
            continue
        buckets[(p["gt_label"], p["pred_label"])].append(p)
    if max_per_cell is not None:
        for key, cell in list(buckets.items()):
            def conf(pp):
                logits = pp["_raw_logits"]
                if logits.numel() == 1:
                    return float(torch.sigmoid(logits).item())
                return float(torch.softmax(logits, dim=0)[pp["pred_label"]].item())
            cell.sort(key=conf, reverse=True)
            buckets[key] = cell[:max_per_cell]
    return buckets


def print_confusion_bucket_summary(buckets, class_names, num_classes):
    print("\n  Confusion-cell counts (rows=GT, cols=Pred):")
    header = "  " + " " * 12 + "  ".join(f"{class_names[c]:>8}" for c in range(num_classes))
    print(header)
    for gt in range(num_classes):
        cells = []
        for pred in range(num_classes):
            n = len(buckets.get((gt, pred), []))
            cells.append(f"{'*' if gt == pred else ' '}{n:>7}")
        print(f"  GT={class_names[gt]:<8}" + "  ".join(cells))
    print("  (* = correct / diagonal)")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (MedClass3D)
# ─────────────────────────────────────────────────────────────────────────────

def load_fold_model(run_dir: Path, fold_idx: int, device, prefer_last: bool = False,
                    metrics=None):
    """Return ``(model, cfg, best_ckpt|None)`` for one fold.

    The model is rebuilt from ``Configs/config.yaml`` with ``pretrained=False``
    (the fold checkpoint supplies all weights; this also avoids needing the
    original pretrain file present), then the best fold checkpoint is loaded.
    """
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(str(run_dir / "Configs" / "config.yaml"))
    if hasattr(cfg, "trainer"):
        cfg.trainer.pop("logger", None)
        cfg.trainer.pop("callbacks", None)
    cfg.data.module.fold = int(fold_idx)
    if hasattr(cfg.model, "pretrained"):
        cfg.model.pretrained = False
    if metrics is not None and hasattr(cfg.model, "metrics"):
        cfg.model.metrics = list(metrics)

    best = find_best_ckpt_for_fold(run_dir / "folds", fold_idx, prefer_last)
    if best is None:
        return None, cfg, None

    model = instantiate(cfg.model)
    state = torch.load(str(best["path"]), map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, cfg, best


def class_names_for(num_classes: int) -> Dict[int, str]:
    if num_classes == 4:
        return {0: "COCA1", 1: "COCA2", 2: "COCA3", 3: "COCA4"}
    return {i: f"Class {i}" for i in range(num_classes)}


# ─────────────────────────────────────────────────────────────────────────────
# Patient assembly
# ─────────────────────────────────────────────────────────────────────────────

def _forward_patient(model, x, device):
    with torch.no_grad():
        logits = _extract_logits(model(x.unsqueeze(0).to(device)))
    if logits.shape[-1] == 1:
        pred = 1 if logits[0, 0].item() > 0 else 0
    else:
        pred = int(torch.argmax(logits[0]).item())
    return pred, logits[0].detach().cpu().clone()


def _patients_from_dataset(ds, model, device, has_gt, only_correct=False,
                           only_incorrect=False, per_class_confusion=False,
                           max_cases=None):
    ids = list(ds.img_files)
    patients = []
    for i in range(len(ds)):
        try:
            x, y = ds[i]
        except Exception as e:
            print(f"    [{i}] sample load failed: {e}")
            continue
        x = x.float() if torch.is_tensor(x) else torch.as_tensor(x).float()
        if x.ndim != 4:
            print(f"    [{i}] unexpected ndim={x.ndim}; skipping")
            continue
        gt_label = int(y) if has_gt else None
        pred_label, raw_logits = _forward_patient(model, x, device)
        if has_gt and not per_class_confusion:
            if only_correct and pred_label != gt_label:
                continue
            if only_incorrect and pred_label == gt_label:
                continue
        patients.append({
            "patient_id": ids[i], "input": x,
            "gt_label": gt_label, "pred_label": pred_label,
            "_raw_logits": raw_logits,
        })
        if not per_class_confusion and max_cases is not None and len(patients) >= max_cases:
            break
    return patients


def build_testset_patients(cfg, model, device, max_cases=None, only_correct=False,
                           only_incorrect=False, per_class_confusion=False):
    dm = instantiate(cfg.data).module
    dm.setup("fit")
    if not getattr(dm, "test_dataset", None):
        print("  No test split for this fold.")
        return []
    ds = dm.test_dataset
    print(f"  Test set size: {len(ds)}")
    pts = _patients_from_dataset(ds, model, device, has_gt=True,
                                 only_correct=only_correct, only_incorrect=only_incorrect,
                                 per_class_confusion=per_class_confusion, max_cases=max_cases)
    print(f"  Collected {len(pts)} test patient(s)")
    return pts


def build_external_patients(cfg, b2nd_dir, has_masks, ok_ids, model, device,
                            max_cases=None):
    """Build patients from already-preprocessed external ``.b2nd`` cases.

    The NIfTI -> .b2nd preprocessing is done by the caller (the external CLI,
    reusing ``predict_external``); here we just wrap them in a ``Class_Data`` via
    a synthetic manifest and run the forward pass. External cases have no GT
    label, so ``gt_label`` is ``None``.
    """
    import pandas as pd
    from medclass3d.data.datamodules import Class_Data

    b2nd_dir = Path(b2nd_dir)
    label_column = cfg.data.module.get("label_column", "label")
    manifest = b2nd_dir.parent / "_external_manifest.csv"
    pd.DataFrame({"image_name": list(ok_ids), "split": "test", "fold": 0,
                  label_column: 0}).to_csv(manifest, index=False)
    test_transforms = instantiate(cfg.data.module.test_transforms)
    ds = Class_Data(img_dir=str(b2nd_dir), csv_file=str(manifest), split="test",
                    fold=0, label_column=label_column, transform=test_transforms,
                    train=False, use_mask=has_masks)
    pts = _patients_from_dataset(ds, model, device, has_gt=False, max_cases=max_cases)
    print(f"  Collected {len(pts)} external patient(s)")
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Gradcam execution
# ─────────────────────────────────────────────────────────────────────────────

def default_z_slices(depth: int) -> List[int]:
    step = max(depth // 8, 1)
    return list(range(step, depth, step))


def run_gradcam_for_patients(model, patients, out_dir, methods, class_names, *,
                             target_class=-1, occ_mask_size=8, occ_overlap=0.5,
                             manuscript_mode=False, npz_only=False,
                             z_slices=None, modality_names=None,
                             per_modality_occlusion=False, side_by_side=True,
                             label="gradcam"):
    """Configure + run multimodal gradcam for one group of patients into ``out_dir``.

    Strips the bookkeeping ``_raw_logits`` key before dispatching (the gradcam
    runner does not expect it), mirroring ``_run_gradcam_on_group``.
    """
    if not patients:
        print(f"  [{label}] no patients — skipping.")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if z_slices is None:
        depth = int(patients[0]["input"].shape[1])
        z_slices = default_z_slices(depth)

    config = GradcamConfig(
        get_stages_fn=get_stages_for_model,
        class_names=class_names,
        methods=list(methods),
        target_class=int(target_class),
        out_dir=str(out_dir),
        default_z_slices=list(z_slices),
        occ_mask_size=int(occ_mask_size),
        occ_overlap=float(occ_overlap),
        manuscript_mode=bool(manuscript_mode),
    )
    config.npz_only = bool(npz_only)
    mm_config = MultiModalConfig(
        modality_names=modality_names,
        per_modality_occlusion=bool(per_modality_occlusion),
        side_by_side_figures=bool(side_by_side),
    )
    clean = [{k: v for k, v in p.items() if k != "_raw_logits"} for p in patients]
    print(f"  [{label}] running {config.methods} on {len(clean)} patient(s) -> {out_dir}")
    run_multimodal_gradcam(model, clean, config, mm_config)
