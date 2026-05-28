import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_TASKS = ("Classification",)

_VALID_LOSS_FNS = {
    "Classification": (
        None, "focal", "weighted_focal", "weighted_ce", "topk10",
    ),
}

# Classification loss_fn values that require class_weights from the datamodule
# and therefore must be instantiated in trainer.setup() rather than __init__.
CLASSIFICATION_LOSSES_NEEDING_WEIGHTS = ("weighted_focal", "weighted_ce")


def _build_criterion(task, loss_fn, label_smoothing, subtask=None):
    """Return the loss callable for a given (task, loss_fn) pair.

    Returns ``None`` for classification variants that need ``class_weights``
    from the datamodule — those are finalized in :meth:`BaseModel.setup`.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task!r}. Expected one of {VALID_TASKS}.")

    valid = _VALID_LOSS_FNS[task]
    if loss_fn not in valid:
        raise ValueError(
            f"Unknown loss_fn={loss_fn!r} for task={task!r}. "
            f"Valid options are: {valid}."
        )

    if loss_fn in CLASSIFICATION_LOSSES_NEEDING_WEIGHTS:
        return None
    if loss_fn is None:
        if subtask == "multilabel":
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if loss_fn == "focal":
        print("Using Focal Loss for Classification")
        return FocalLoss(alpha=None, gamma=2.0)
    if loss_fn == "topk10":
        print("Using TopK10 Loss for Classification")
        return TopKLoss(k=10)

    raise ValueError(f"Unhandled (task, loss_fn) pair: ({task!r}, {loss_fn!r}).")


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class alpha weighting.

    ``alpha`` may be ``None`` (uniform), a scalar (uniform weight across all
    classes), or a list/tensor of length ``num_classes`` (per-class weight).
    The per-class form is the proper multiclass focal weighting and is what
    ``BaseModel.setup`` uses when class_weights are computed from the train
    split.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is None:
            loss = focal_weight * ce_loss
        elif isinstance(self.alpha, (int, float)):
            loss = self.alpha * focal_weight * ce_loss
        else:
            alpha = (
                self.alpha
                if isinstance(self.alpha, torch.Tensor)
                else torch.tensor(self.alpha)
            )
            alpha_t = alpha.to(inputs.device).to(inputs.dtype).gather(0, targets)
            loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """Wraps ``F.cross_entropy`` with per-class weights and label smoothing."""

    def __init__(self, weight=None, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if weight is None:
            self.weight = None
        elif isinstance(weight, (list, tuple)):
            self.weight = torch.tensor(weight, dtype=torch.float32)
        elif isinstance(weight, torch.Tensor):
            self.weight = weight.float()
        else:
            raise TypeError(f"Invalid type for weight: {type(weight)}")

    def forward(self, inputs, targets):
        weight = self.weight.to(inputs.device) if self.weight is not None else None
        return F.cross_entropy(
            inputs,
            targets,
            weight=weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class TopKLoss(nn.Module):
    """Mean of the top-k% per-sample cross-entropy losses in a batch."""

    def __init__(self, k=10, reduction="mean"):
        super().__init__()
        self.k = k
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        k = max(1, int(len(ce_loss) * self.k / 100))
        topk_loss, _ = torch.topk(ce_loss, k)
        if self.reduction == "mean":
            return topk_loss.mean()
        return topk_loss.sum()
