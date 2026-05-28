"""
Re-run val + test inference on the held-out splits from a trained run's CSV.

Reads the training-run snapshot from `<run_dir>/Configs/config.yaml`, loads the
matching checkpoint from `<run_dir>/folds/<fold>/`, and re-runs the validation
and test splits as defined in the training CSV. Produces:

    <pred_dir>/predictions_<split>_fold<k>.xlsx     -- PatientID, GT, Pred, Prob_0/1/...
    <pred_dir>/confusion_matrix_<split>_fold<k>.png
    <pred_dir>/summary_<split>.csv                  -- Accuracy / BalancedAcc / F1 / AUROC

For inference on a directory of raw NIfTI files outside the training CSV, use
`predict_external.py` instead.

Usage:
    python scripts/predict_test.py \\
        --run-dir /path/to/<output_dir>/<dataset>/<run_name> \\
        --fold 0
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from medclass3d.utils.parsing import make_omegaconf_resolvers


def _select_best_ckpt(ckp_paths, prefer_best=True):
    """
    From a list of checkpoint paths, pick one. If ``prefer_best`` is True,
    return the checkpoint with the highest ``Val_acc`` parsed from its
    filename. Falls back to ``last.ckpt`` if no parseable filename is found.
    """
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


def _compute_metrics(probs, targets, num_classes):
    """Return Accuracy / BalancedAccuracy / Macro-F1 / Macro-AUROC."""
    probs_np = probs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy().astype(int)
    preds_np = probs_np.argmax(axis=1)

    out = {
        "N": int(targets_np.size),
        "Accuracy": float(accuracy_score(targets_np, preds_np)),
        "BalancedAccuracy": float(balanced_accuracy_score(targets_np, preds_np)),
        "F1_macro": float(f1_score(targets_np, preds_np, average="macro", zero_division=0)),
    }
    # AUROC needs at least 2 classes present in y_true
    if num_classes == 2:
        try:
            out["AUROC"] = float(roc_auc_score(targets_np, probs_np[:, 1]))
        except ValueError:
            out["AUROC"] = float("nan")
    else:
        try:
            out["AUROC"] = float(
                roc_auc_score(targets_np, probs_np, multi_class="ovr", average="macro")
            )
        except ValueError:
            out["AUROC"] = float("nan")
    return out


def _save_confusion_matrix(targets, probs, tag, pred_dir, num_classes):
    """Compute + save a raw and row-normalized confusion matrix as PNGs."""
    preds = probs.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy().astype(int)
    cm = confusion_matrix(targets_np, preds, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="viridis", interpolation="nearest")
    ax.set_title(f"Confusion matrix ({tag})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] < cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = os.path.join(pred_dir, f"confusion_matrix_{tag}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved confusion matrix: {out_path}")
    return cm


def _run_split(split, model, dataset, trainer, fold_id, pred_dir, num_classes):
    """Run inference on one split ('val' or 'test') and save artifacts."""
    if split == "val":
        loader = dataset.val_dataloader()
        ids_source = dataset.val_dataset.img_files
    elif split == "test":
        loader = dataset.test_dataloader()
        ids_source = dataset.test_dataset.img_files
    else:
        raise ValueError(f"Unknown split: {split!r}")

    predictions = trainer.predict(model, dataloaders=loader)

    ys, y_hats = zip(*predictions)
    targets = torch.cat([y.detach().cpu() for y in ys]).long()
    logits = torch.cat([y.detach().cpu() for y in y_hats], dim=0)
    probs = F.softmax(logits, dim=-1)
    preds = probs.argmax(dim=1)

    patient_ids = list(ids_source)
    if len(patient_ids) != len(preds):
        raise RuntimeError(
            f"[fold {fold_id} / {split}] Length mismatch between patient IDs "
            f"({len(patient_ids)}) and predictions ({len(preds)})."
        )

    metrics = _compute_metrics(probs, targets, num_classes)
    print(
        f"[fold {fold_id} / {split}] N={metrics['N']}  Acc={metrics['Accuracy']:.4f}  "
        f"BalAcc={metrics['BalancedAccuracy']:.4f}  F1={metrics['F1_macro']:.4f}  "
        f"AUROC={metrics['AUROC']:.4f}"
    )

    # Per-case predictions xlsx
    df_data = {
        "PatientID": patient_ids,
        "GroundTruth": targets.numpy(),
        "Pred": preds.numpy(),
    }
    for i in range(num_classes):
        df_data[f"Prob_{i}"] = probs[:, i].numpy()
    df = pd.DataFrame(df_data)
    pred_path = os.path.join(pred_dir, f"predictions_{split}_fold{fold_id}.xlsx")
    df.to_excel(pred_path, index=False)
    print(f"[fold {fold_id} / {split}] Saved predictions to {pred_path}")

    # Confusion matrix figure
    _save_confusion_matrix(
        targets, probs, tag=f"{split}_fold{fold_id}",
        pred_dir=pred_dir, num_classes=num_classes,
    )

    # Per-split summary CSV
    summary_df = pd.DataFrame([{"Split": split, "Fold": fold_id, **metrics}])
    for col in ("Accuracy", "BalancedAccuracy", "F1_macro", "AUROC"):
        summary_df[col] = summary_df[col].apply(
            lambda v: round(v, 4) if isinstance(v, (int, float)) and not pd.isna(v) else v
        )
    summary_path = os.path.join(pred_dir, f"summary_{split}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[fold {fold_id} / {split}] Saved summary to {summary_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Training-run directory containing Configs/config.yaml and folds/<k>/*.ckpt.")
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold's checkpoint to load. Default: 0")
    parser.add_argument("--pred-dir", type=Path, default=None,
                        help="Where to write predictions/reports. Default: <run-dir>/predictions/")
    parser.add_argument("--metrics", nargs="+", default=["acc", "balanced_acc", "f1", "auroc"],
                        help="Metric names forwarded to the model. Default: acc balanced_acc f1 auroc")
    parser.add_argument("--prefer-last", action="store_true",
                        help="Use last.ckpt instead of the best-Val_acc checkpoint.")
    args = parser.parse_args()

    make_omegaconf_resolvers()

    # ---- Resolve directories from the single run_dir input ---- #
    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir does not exist: {run_dir}")

    fold_id = str(args.fold)
    ckp_dir = run_dir / "folds" / fold_id
    ckp_list = list(ckp_dir.glob("*.ckpt"))
    if not ckp_list:
        raise SystemExit(f"No checkpoints found under {ckp_dir}")

    ckp_path = _select_best_ckpt(ckp_list, prefer_best=not args.prefer_last)
    if ckp_path is None:
        raise SystemExit(f"No usable checkpoint selected from {ckp_dir}")
    print(f"[fold {fold_id}] using checkpoint: {ckp_path}")

    training_config_path = run_dir / "Configs" / "config.yaml"
    if not training_config_path.is_file():
        raise SystemExit(f"Training config not found at {training_config_path}")
    print(f"Using training config: {training_config_path}")

    pred_dir = args.pred_dir if args.pred_dir else run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build model + datamodule + trainer from the saved training config ---- #
    used_training_cfg = OmegaConf.load(training_config_path)
    used_training_cfg.trainer.pop("logger", None)
    used_training_cfg.trainer.pop("callbacks", None)
    used_training_cfg.model.metrics = list(args.metrics)

    # Force single-device prediction so neither val nor test gets sharded/padded
    used_training_cfg.trainer.devices = 1
    used_training_cfg.trainer.strategy = "auto"
    used_training_cfg.trainer.sync_batchnorm = False

    used_training_cfg.data.module.fold = int(fold_id)
    num_classes = int(used_training_cfg.data.num_classes)

    model = instantiate(used_training_cfg.model)
    state = torch.load(ckp_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    trainer = instantiate(used_training_cfg.trainer)

    dataset = instantiate(used_training_cfg.data).module
    dataset.setup("fit")

    for split in ("val", "test"):
        if not hasattr(dataset, f"{split}_dataset"):
            print(f"[skip] no {split!r} split in CSV for fold {fold_id}")
            continue
        _run_split(
            split=split,
            model=model,
            dataset=dataset,
            trainer=trainer,
            fold_id=fold_id,
            pred_dir=str(pred_dir),
            num_classes=num_classes,
        )


if __name__ == "__main__":
    main()
