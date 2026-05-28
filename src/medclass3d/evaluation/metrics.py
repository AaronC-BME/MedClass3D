from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    Precision,
    Recall,
)

from medclass3d.evaluation.balanced_accuracy import BalancedAccuracy


def _build_regression_metrics(metrics_list):
    out = {}
    if "mse" in metrics_list:
        out["MSE"] = MeanSquaredError()
    if "mae" in metrics_list:
        out["MAE"] = MeanAbsoluteError()
    return out


def _build_classification_metrics(metrics_list, num_classes, subtask="multiclass"):
    """Map string metric names to torchmetrics instances for classification.

    Available names: ``acc``, ``balanced_acc``, ``f1``, ``f1_per_class``,
    ``pr`` (Precision+Recall), ``top5acc``, ``auroc``, ``ap``.

    F1_per_class returns a per-class tensor; the trainer's log helper explodes
    it into ``F1_class_0/1/...`` keys with NaN→0 sanitization.
    """
    assert subtask in ("multiclass", "multilabel"), (
        f"Unknown subtask: {subtask!r}. Expected 'multiclass' or 'multilabel'."
    )
    common = dict(task=subtask, num_classes=num_classes, num_labels=num_classes)

    out = {}
    if "acc" in metrics_list:
        out["Accuracy"] = Accuracy(**common)
    if "balanced_acc" in metrics_list:
        out["Balanced_Accuracy"] = BalancedAccuracy(task=subtask, num_classes=num_classes)
    if "f1" in metrics_list:
        out["F1"] = F1Score(average="macro", **common)
    if "f1_per_class" in metrics_list:
        out["F1_per_class"] = F1Score(average=None, **common)
    if "pr" in metrics_list:
        out["Precision"] = Precision(average="macro", **common)
        out["Recall"] = Recall(average="macro", **common)
    if "top5acc" in metrics_list:
        out["Accuracy_top5"] = Accuracy(top_k=5, **common)
    if "auroc" in metrics_list:
        out["AUROC"] = AUROC(average="macro", **common)
    if "ap" in metrics_list:
        out["AP"] = AveragePrecision(**common)
    return out
