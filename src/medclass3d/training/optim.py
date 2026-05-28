import math
import warnings

from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR_Warmstart(_LRScheduler):
    """
    CosineAnnealingLR with a linear warmup phase. See
    https://arxiv.org/pdf/1706.02677.pdf.
    """

    def __init__(
        self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False, warmstart=0,
    ):
        self.T_max = T_max - warmstart  # warmup epochs not part of cosine period
        self.eta_min = eta_min
        self.warmstart = warmstart
        self.T = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        # Warmup
        if self.last_epoch < self.warmstart:
            addrates = [lr / (self.warmstart + 1) for lr in self.base_lrs]
            return [
                addrates[i] * (self.last_epoch + 1)
                for i, _ in enumerate(self.optimizer.param_groups)
            ]

        # Cosine annealing
        if self.T == 0:
            self.T += 1
            return self.base_lrs

        if (self.T - 1 - self.T_max) % (2 * self.T_max) == 0:
            updated_lr = [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / self.T_max))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
            self.T += 1
            return updated_lr

        updated_lr = [
            (1 + math.cos(math.pi * self.T / self.T_max))
            / (1 + math.cos(math.pi * (self.T - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
        self.T += 1
        return updated_lr


class CosineAnnealingLR_DoubleWarmstart(_LRScheduler):
    """
    Two consecutive linear warmup phases followed by cosine annealing.

    - Phase 1 (warmstart1 epochs): only the head warms up; encoder LR stays at 0.
    - Phase 2 (warmstart2 epochs): both head and encoder warm up.
    - Cosine annealing decays both groups.
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
        warmstart1=0,
        warmstart2=0,
    ):
        self.warmstart1 = warmstart1
        self.warmstart2 = warmstart2
        self.eta_min = eta_min
        self.T_max = T_max - (warmstart1 + warmstart2)  # cosine decay period
        self.T = 0  # internal counter (unused, kept for parity)

        # Locate parameter groups by name; either "cls_head" or "reg_head" works.
        self.head_group = None
        self.encoder_group = None
        for param_group in optimizer.param_groups:
            name = param_group.get("name")
            if name in ("cls_head", "reg_head"):
                self.head_group = param_group
            elif name == "encoder":
                self.encoder_group = param_group

        if self.head_group is None:
            raise ValueError(
                "Optimizer must have a parameter group named 'cls_head' or 'reg_head'."
            )
        if self.encoder_group is None:
            raise ValueError("Optimizer must have a parameter group named 'encoder'.")

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        warmup_total = self.warmstart1 + self.warmstart2

        # Phase 1: warm up the head only.
        if self.last_epoch < self.warmstart1:
            denom = max(self.warmstart1, 1)
            warmup_factor = (self.last_epoch + 1) / denom
            return [
                group["initial_lr"] * warmup_factor if group is self.head_group else 0
                for group in self.optimizer.param_groups
            ]

        # Phase 2: warm up both head and encoder.
        if self.last_epoch < warmup_total:
            denom = max(self.warmstart2, 1)
            warmup_factor = (self.last_epoch - self.warmstart1 + 1) / denom
            return [
                group["initial_lr"] * warmup_factor
                for group in self.optimizer.param_groups
            ]

        # Cosine annealing for both groups.
        epoch_cosine = self.last_epoch - warmup_total
        return [
            self.eta_min
            + (group["initial_lr"] - self.eta_min)
            * 0.5
            * (1 + math.cos(math.pi * epoch_cosine / self.T_max))
            for group in self.optimizer.param_groups
        ]
