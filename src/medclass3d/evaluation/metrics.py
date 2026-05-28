import torch
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
)


def _build_regression_metrics(metrics_list):
    out = {}
    if "mse" in metrics_list:
        out["MSE"] = MeanSquaredError()
    if "mae" in metrics_list:
        out["MAE"] = MeanAbsoluteError()
    return out
