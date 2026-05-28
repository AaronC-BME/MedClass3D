"""
Re-run val + test inference on the held-out splits from a trained run's CSV.

Reads the training-run snapshot from `<run_dir>/Configs/config.yaml`, loads the
matching checkpoint from `<run_dir>/folds/<fold>/`, and re-runs the validation
and test splits as defined in the training CSV. Produces:

    <pred_dir>/predictions_<split>_fold<k>.xlsx     -- per-case GT vs prediction
    <pred_dir>/error_by_age_bin_<split>_fold<k>.csv -- binned error stats
    <pred_dir>/error_by_age_bin_<split>_fold<k>_MAE.png
    <pred_dir>/error_by_age_bin_<split>_fold<k>_MeanError.png
    <pred_dir>/summary_<split>.csv                  -- MAE/RMSE/MeanError/Pearson_r

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
from hydra.utils import instantiate
from omegaconf import OmegaConf

from medclass3d.utils.parsing import make_omegaconf_resolvers


def _select_best_ckpt(ckp_paths, prefer_best=True):
    """
    From a list of checkpoint paths, pick one.
    If prefer_best is True, return the checkpoint with the lowest 'Val_mae'
    parsed from its filename. Falls back to last.ckpt if no parseable filename
    is found. If prefer_best is False, return last.ckpt.
    """
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


def _compute_metrics(preds, targets):
    """Return a dict of regression metrics given two 1-D float tensors."""
    preds = preds.float()
    targets = targets.float()
    err = preds - targets
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err ** 2).mean()).item()
    me = err.mean().item()
    n = preds.numel()
    if n > 1 and preds.std().item() > 0 and targets.std().item() > 0:
        pearson = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()
    else:
        pearson = float("nan")
    return {"N": n, "MAE": mae, "RMSE": rmse, "MeanError": me, "Pearson_r": pearson}


def _bin_errors_by_age(targets, preds, bin_width=10, max_age=100):
    """
    Group prediction errors into age bins of `bin_width` years based on the
    ground-truth age. Returns a DataFrame with columns:
        Bin, N, MAE, RMSE, MeanError, StdError
    """
    targets_np = targets.detach().cpu().numpy().astype(float)
    preds_np = preds.detach().cpu().numpy().astype(float)
    err_np = preds_np - targets_np
    abs_err_np = np.abs(err_np)

    edges = np.arange(0, max_age + bin_width, bin_width)
    labels = [f"{int(edges[i])}-{int(edges[i + 1]) - 1}" for i in range(len(edges) - 1)]

    rows = []
    for i, lab in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        mask = (targets_np >= lo) & (targets_np < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"Bin": lab, "N": 0, "MAE": np.nan, "RMSE": np.nan,
                         "MeanError": np.nan, "StdError": np.nan})
        else:
            rows.append({
                "Bin": lab,
                "N": n,
                "MAE": float(abs_err_np[mask].mean()),
                "RMSE": float(np.sqrt((err_np[mask] ** 2).mean())),
                "MeanError": float(err_np[mask].mean()),
                "StdError": float(err_np[mask].std(ddof=1)) if n > 1 else 0.0,
            })

    overflow_mask = targets_np >= max_age
    n_over = int(overflow_mask.sum())
    if n_over > 0:
        rows.append({
            "Bin": f">={int(max_age)}",
            "N": n_over,
            "MAE": float(abs_err_np[overflow_mask].mean()),
            "RMSE": float(np.sqrt((err_np[overflow_mask] ** 2).mean())),
            "MeanError": float(err_np[overflow_mask].mean()),
            "StdError": float(err_np[overflow_mask].std(ddof=1)) if n_over > 1 else 0.0,
        })

    return pd.DataFrame(rows)


def _plot_error_bars(bin_df, title, out_path, metric="MAE"):
    """Bar chart of `metric` per age bin. Empty bins drawn as gaps."""
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(bin_df)), 5))

    bins = bin_df["Bin"].tolist()
    values = bin_df[metric].to_numpy(dtype=float)
    counts = bin_df["N"].to_numpy(dtype=int)

    plot_values = np.where(np.isnan(values), 0.0, values)

    bars = ax.bar(bins, plot_values, color="steelblue", edgecolor="black")

    for bar, n, v in zip(bars, counts, values):
        height = bar.get_height()
        if n == 0 or np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, 0,
                    "n=0", ha="center", va="bottom", fontsize=8, color="gray")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"n={n}\n{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Age bin (years)")
    ax.set_ylabel(metric + (" (years)" if metric in ("MAE", "RMSE", "MeanError") else ""))
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if metric == "MeanError":
        ax.axhline(0, color="black", linewidth=0.8)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved plot: {out_path}")


def _save_bin_report(targets, preds, tag, pred_dir, bin_width=10, max_age=100):
    """Compute age-binned errors + save CSV + bar charts."""
    bin_df = _bin_errors_by_age(targets, preds, bin_width=bin_width, max_age=max_age)

    csv_path = os.path.join(pred_dir, f"error_by_age_bin_{tag}.csv")
    bin_df.to_csv(csv_path, index=False)
    print(f"[{tag}] saved per-bin error stats to {csv_path}")

    _plot_error_bars(
        bin_df,
        title=f"MAE per age bin ({tag})",
        out_path=os.path.join(pred_dir, f"error_by_age_bin_{tag}_MAE.png"),
        metric="MAE",
    )
    _plot_error_bars(
        bin_df,
        title=f"Mean signed error per age bin ({tag})",
        out_path=os.path.join(pred_dir, f"error_by_age_bin_{tag}_MeanError.png"),
        metric="MeanError",
    )

    return bin_df


def _run_split(split, model, dataset, trainer, fold_id, pred_dir, bin_width, max_age):
    """Run inference on one split ('val' or 'test') and save artifacts."""
    dataset.setup(split)
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
    targets = torch.cat([y.detach().cpu() for y in ys]).float()
    probas = torch.cat([p.detach().cpu() for (_, p) in y_hats], dim=0)
    preds = (probas > 0.5).sum(dim=1).float()

    patient_ids = list(ids_source)
    if len(patient_ids) != len(preds):
        raise RuntimeError(
            f"[fold {fold_id} / {split}] Length mismatch between patient IDs "
            f"({len(patient_ids)}) and predictions ({len(preds)})."
        )

    metrics = _compute_metrics(preds, targets)
    print(
        f"[fold {fold_id} / {split}] N={metrics['N']}  MAE={metrics['MAE']:.4f}  "
        f"RMSE={metrics['RMSE']:.4f}  ME={metrics['MeanError']:.4f}  "
        f"Pearson={metrics['Pearson_r']:.4f}"
    )

    # Per-case predictions xlsx
    df = pd.DataFrame({
        "PatientID": patient_ids,
        "GroundTruth": targets.numpy(),
        "Prediction": preds.numpy(),
        "AbsError": (preds - targets).abs().numpy(),
        "Error": (preds - targets).numpy(),
    })
    pred_path = os.path.join(pred_dir, f"predictions_{split}_fold{fold_id}.xlsx")
    df.to_excel(pred_path, index=False)
    print(f"[fold {fold_id} / {split}] Saved predictions to {pred_path}")

    # Age-bin report + plots
    _save_bin_report(
        targets, preds,
        tag=f"{split}_fold{fold_id}",
        pred_dir=pred_dir,
        bin_width=bin_width,
        max_age=max_age,
    )

    # Per-split summary CSV
    summary_df = pd.DataFrame([{"Split": split, "Fold": fold_id, **metrics}])
    for col in ["MAE", "RMSE", "MeanError", "Pearson_r"]:
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
    parser.add_argument("--metrics", nargs="+", default=["mae", "mse"],
                        help="Metric names forwarded to the model. Default: mae mse")
    parser.add_argument("--prefer-last", action="store_true",
                        help="Use last.ckpt instead of the best-Val_MAE checkpoint.")
    parser.add_argument("--age-bin-width", type=int, default=10,
                        help="Width in years for the age-bin error report. Default: 10")
    parser.add_argument("--age-bin-max", type=int, default=100,
                        help="Upper limit for age bins. Default: 100")
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

    model = instantiate(used_training_cfg.model)
    state = torch.load(ckp_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    trainer = instantiate(used_training_cfg.trainer)

    dataset = instantiate(used_training_cfg.data).module
    dataset.setup("fit")  # build whatever splits exist in the CSV

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
            bin_width=args.age_bin_width,
            max_age=args.age_bin_max,
        )


if __name__ == "__main__":
    main()
