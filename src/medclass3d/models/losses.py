from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_TASKS = ("Regression", "Ordinal_Regression", "Classification")

_VALID_LOSS_FNS = {
    "Regression": (None,),
    "Ordinal_Regression": (
        None, "focal", "topk10", "topk20",
        "bce_focal", "bce_topk10", "bce_topk20",
        "weighted_bce", "bce_mae",
    ),
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

    if task == "Classification":
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

    if task == "Regression":
        return nn.MSELoss()

    if task == "Ordinal_Regression":
        if loss_fn is None:
            return coral_loss
        if loss_fn == "focal":
            print("Using Coral Focal Loss (Gamma=3.0) for Ordinal Regression")
            return partial(coral_focal_loss, gamma=3.0)
        if loss_fn == "topk10":
            print("Using Coral TopK10 Loss for Ordinal Regression")
            return coral_topk_loss
        if loss_fn == "topk20":
            print("Using Coral TopK20 Loss for Ordinal Regression")
            return partial(coral_topk_loss, k=20)
        if loss_fn == "bce_focal":
            print("Using Combined BCE and Focal Loss (Gamma=3.0) for Ordinal Regression")
            return partial(combined_bce_focal_loss, gamma=3.0)
        if loss_fn == "bce_topk10":
            print("Using Combined BCE and TopK10 Loss for Ordinal Regression")
            return combined_bce_topk_loss
        if loss_fn == "bce_topk20":
            print("Using Combined BCE and TopK20 Loss for Ordinal Regression")
            return partial(combined_bce_topk_loss, topk=20)
        if loss_fn == "weighted_bce":
            print("Using Weighted BCE Loss for Ordinal Regression")
            return coral_loss
        if loss_fn == "bce_mae":
            print("Using BCE Loss and MAE (L1) Loss for Ordinal Regression")
            return combined_coral_mae_loss

    raise ValueError(f"Unhandled (task, loss_fn) pair: ({task!r}, {loss_fn!r}).")


def coral_loss(logits, levels, importance_weights=None):
    loss = F.binary_cross_entropy_with_logits(logits, levels, reduction='none')
    if importance_weights is not None:
        loss = loss * importance_weights.view(1,-1)
    return loss.mean()


def coral_focal_loss(logits, levels, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(logits)
    pt = torch.where(levels == 1, prob, 1 - prob)
    ce_loss = F.binary_cross_entropy_with_logits(logits, levels, reduction='none')
    focal_weight = (1 - pt) ** gamma
    if alpha is not None:
        alpha_factor = torch.where(levels == 1, alpha, 1 - alpha)
        ce_loss = ce_loss * alpha_factor
    return (focal_weight * ce_loss).mean()


def coral_topk_loss(logits, levels, k=20):
    bce = F.binary_cross_entropy_with_logits(logits, levels, reduction='none')
    topk_vals, _ = torch.topk(bce, k=min(k, bce.shape[1]), dim=1)
    return topk_vals.mean()


def combined_bce_focal_loss(logits, levels, alpha=0.25, gamma=2.0, focal_weight=0.5, importance_weights=None):
    bce_loss = F.binary_cross_entropy_with_logits(logits, levels, reduction='none')
    if importance_weights is not None:
        bce_loss = bce_loss * importance_weights.view(1, -1)
    prob = torch.sigmoid(logits)
    pt = torch.where(levels == 1, prob, 1 - prob)
    focal_term = (1 - pt) ** gamma
    if alpha is not None:
        alpha_factor = torch.where(levels == 1, alpha, 1 - alpha)
        focal_term = focal_term * alpha_factor
    focal_loss = focal_term * bce_loss
    total_loss = (1 - focal_weight) * bce_loss + focal_weight * focal_loss
    return total_loss.mean()


def combined_bce_topk_loss(logits, levels, topk=10, topk_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, levels, reduction='none')  # shape [B, K-1]
    topk_vals, _ = torch.topk(bce, k=min(topk, bce.shape[1]), dim=1)  # shape [B, topk]
    print(f"Using the following k value: {min(topk, bce.shape[1])}")  # Debugging line
    topk_loss = topk_vals.mean()
    full_bce_loss = bce.mean()
    return (1 - topk_weight) * full_bce_loss + topk_weight * topk_loss


def label_to_levels(labels, num_classes):
    batch_size = labels.size(0)
    levels = torch.zeros(batch_size, num_classes - 1, device=labels.device)
    for i, label in enumerate(labels):
        levels[i, :int(label)] = 1
    return levels


def combined_coral_mae_loss(logits, levels, labels, mae_weight=0.2, importance_weights=None):
    coral = coral_loss(logits, levels, importance_weights)
    pred_soft_label = torch.sigmoid(logits).sum(dim=1)  # shape: [B]
    mae = F.l1_loss(pred_soft_label, labels.float())
    return coral + mae_weight * mae


# ---------------------------------------------------------------------------
# Classification losses
# ---------------------------------------------------------------------------


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
