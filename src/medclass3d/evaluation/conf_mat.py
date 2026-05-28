import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torchmetrics import Metric
from torchmetrics.utilities.data import _bincount


class ConfusionMatrix(Metric):
    """Multiclass confusion matrix that logs as image figures.

    ``save_state`` writes the raw + row-normalized matrices to whichever
    Lightning logger is configured (W&B, TensorBoard, or MLFlow). Reset must
    be called by the caller after ``save_state``.
    """

    full_state_update = False

    def __init__(self, num_classes: int, labels: list = None) -> None:
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        self.labels = labels if labels is not None else np.arange(num_classes).astype(str)
        self.add_state(
            "mat",
            default=torch.zeros((num_classes, num_classes), dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def compute(self):
        return self.mat

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        pred = pred.argmax(1).flatten()
        gt = gt.flatten()
        n = self.num_classes

        with torch.no_grad():
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]
            confmat = _bincount(inds, minlength=n**2).reshape(n, n)

        self.mat += confmat

    def save_state(self, trainer: pl.Trainer, split: str) -> None:
        def mat_to_figure(mat: np.ndarray, name: str, norm_colorbar: bool = False) -> Figure:
            figure = plt.figure(figsize=(8, 8))
            plt.imshow(mat, interpolation="nearest", cmap=plt.cm.viridis)
            plt.title(name)
            if norm_colorbar:
                plt.clim(0, 1)
            plt.colorbar()
            labels = getattr(self, "class_names", np.arange(self.num_classes))
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=0)
            plt.yticks(tick_marks, labels)
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.close(figure)
            return figure

        confmat = self.mat.detach().cpu().numpy()
        figure = mat_to_figure(confmat, "Confusion Matrix")

        row_sums = confmat.sum(axis=1)[:, np.newaxis]
        # Avoid divide-by-zero for empty rows
        with np.errstate(divide="ignore", invalid="ignore"):
            confmat_norm = np.around(
                np.divide(confmat.astype("float"), row_sums, where=row_sums != 0),
                decimals=2,
            )
        figure_norm = mat_to_figure(
            confmat_norm, "Confusion Matrix (normalized)", norm_colorbar=True
        )

        loggers = trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]
        for logger in loggers:
            if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                logger.experiment.add_figure(
                    f"{split}_ConfusionMatrix_normalized/ConfusionMatrix",
                    figure_norm,
                    trainer.current_epoch,
                )
                logger.experiment.add_figure(
                    f"{split}_ConfusionMatrix_absolute/ConfusionMatrix",
                    figure,
                    trainer.current_epoch,
                )
            elif isinstance(logger, pl.loggers.mlflow.MLFlowLogger):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=figure_norm,
                    artifact_file=f"{split}_ConfusionMatrix_normalized.png",
                )
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=figure,
                    artifact_file=f"{split}_ConfusionMatrix_absolute.png",
                )
            elif isinstance(logger, pl.loggers.wandb.WandbLogger):
                logger.log_image(
                    key=f"{split}_ConfusionMatrix_normalized",
                    images=[figure_norm],
                    step=trainer.current_epoch,
                )
                logger.log_image(
                    key=f"{split}_ConfusionMatrix_absolute",
                    images=[figure],
                    step=trainer.current_epoch,
                )
