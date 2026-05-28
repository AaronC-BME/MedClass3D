import torch
import torch.nn.functional as F
import wandb
import lightning as L
from madgrad import MADGRAD
from timm.optim import RMSpropTF
from torchmetrics import MetricCollection
from torchmetrics.aggregation import CatMetric

from medclass3d.data.mixup import mixup_criterion, mixup_data
from medclass3d.evaluation.conf_mat import ConfusionMatrix
from medclass3d.evaluation.metrics import _build_classification_metrics
from medclass3d.models.losses import (
    CLASSIFICATION_LOSSES_NEEDING_WEIGHTS,
    VALID_TASKS,
    FocalLoss,
    WeightedCrossEntropyLoss,
    _build_criterion,
)
from medclass3d.training.optim import (
    CosineAnnealingLR_DoubleWarmstart,
    CosineAnnealingLR_Warmstart,
)
from medclass3d.training.sam import SAM


class BaseModel(L.LightningModule):
    def __init__(
        self,
        task,
        loss_fn,
        metric_computation_mode,
        result_plot,
        metrics,
        num_classes,
        name,
        lr,
        weight_decay,
        optimizer,
        nesterov,
        sam,
        adaptive_sam,
        scheduler,
        T_max,
        warmstart,
        epochs,
        mixup,
        mixup_alpha,
        label_smoothing,
        stochastic_depth,
        resnet_dropout,
        squeeze_excitation,
        apply_shakedrop,
        undecay_norm,
        zero_init_residual,
        input_dim,
        input_channels,
        pretrained,
        *args,
        **kwargs,
    ):
        super().__init__()

        # --- Task / loss ------------------------------------------------------
        if task not in VALID_TASKS:
            raise ValueError(
                f"Unknown task: {task!r}. Expected one of {VALID_TASKS}."
            )
        self.task = task
        self.loss_fn = loss_fn  # None => CE / BCEWithLogits per subtask
        self.subtask = kwargs.get("subtask", "multiclass")
        if self.subtask not in ("multiclass", "multilabel"):
            raise ValueError(
                f"Unknown subtask: {self.subtask!r}. Expected 'multiclass' or 'multilabel'."
            )

        # --- Metrics ----------------------------------------------------------
        self.metric_computation_mode = metric_computation_mode
        self.result_plot_setting = result_plot

        metrics_dict = _build_classification_metrics(
            metrics, num_classes=num_classes, subtask=self.subtask,
        )

        # Confusion matrix bookkeeping (always on for classification).
        if self.result_plot_setting in ("val", "all"):
            self.val_conf_mat = ConfusionMatrix(num_classes=num_classes)
        if self.result_plot_setting == "all":
            self.train_conf_mat = ConfusionMatrix(num_classes=num_classes)

        self.save_preds = bool(kwargs["save_preds"])
        if self.save_preds:
            self.val_preds = CatMetric(dist_sync_on_step=False)
            self.val_labels = CatMetric(dist_sync_on_step=False)
            self.val_indices = CatMetric(dist_sync_on_step=False)

        metric_collection = MetricCollection(metrics_dict)
        self.train_metrics = metric_collection.clone(prefix="Train/")
        self.val_metrics = metric_collection.clone(prefix="Val/")

        # --- Training args ----------------------------------------------------
        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.nesterov = nesterov
        self.sam = sam
        self.adaptive_sam = adaptive_sam
        self.scheduler = scheduler
        self.T_max = T_max
        self.warmstart = warmstart
        self.warmstart2 = kwargs["warmstart2"]
        self.epochs = epochs
        self.pretrained = pretrained

        # --- Regularization ---------------------------------------------------
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.stochastic_depth = stochastic_depth
        self.resnet_dropout = resnet_dropout
        self.se = squeeze_excitation
        self.apply_shakedrop = apply_shakedrop
        self.undecay_norm = undecay_norm
        self.zero_init_residual = zero_init_residual

        # --- Finetuning -------------------------------------------------------
        self.finetuning_method = kwargs["finetune_method"]

        # --- Data -------------------------------------------------------------
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.num_classes = num_classes

        # SAM uses manual optimization
        if self.sam:
            self.automatic_optimization = False

        # --- Loss -------------------------------------------------------------
        # For classification variants that need class_weights (weighted_ce,
        # weighted_focal), _build_criterion returns None and setup() finalizes
        # the criterion using datamodule.class_weights.
        self.criterion = _build_criterion(
            self.task, self.loss_fn, self.label_smoothing, subtask=self.subtask,
        )

    # -----------------------------------------------------------------------
    # Forward / setup
    # -----------------------------------------------------------------------

    def forward(self, x):
        pass

    def setup(self, stage=None):
        # Classification losses that need class_weights are instantiated here,
        # after the datamodule has computed them on the train split.
        if self.loss_fn in CLASSIFICATION_LOSSES_NEEDING_WEIGHTS:
            class_weights = getattr(self.trainer.datamodule, "class_weights", None)
            if class_weights is None:
                raise RuntimeError(
                    f"loss_fn={self.loss_fn!r} requires datamodule.class_weights, "
                    "but the datamodule did not expose any."
                )
            class_weights = class_weights.to(self.device)
            if self.loss_fn == "weighted_ce":
                print(f"[setup] WeightedCrossEntropyLoss weights={class_weights.tolist()}")
                self.criterion = WeightedCrossEntropyLoss(
                    weight=class_weights, label_smoothing=self.label_smoothing,
                )
            elif self.loss_fn == "weighted_focal":
                print(f"[setup] FocalLoss per-class alpha={class_weights.tolist()}")
                self.criterion = FocalLoss(alpha=class_weights, gamma=1.5)

    # -----------------------------------------------------------------------
    # Step helpers
    # -----------------------------------------------------------------------

    def _compute_loss(self, y_hat, y):
        """Standard (non-mixup, non-SAM) loss given logits and labels."""
        if self.subtask == "multilabel":
            return self.criterion(y_hat, y.float())
        return self.criterion(y_hat, y.long())

    def _update_metrics(self, metrics_obj, y_hat, y):
        """Update an epochwise MetricCollection with predictions in the right form."""
        if self.subtask == "multilabel":
            metrics_obj.update(torch.sigmoid(y_hat.detach()), y)
        else:
            metrics_obj.update(F.softmax(y_hat.detach(), dim=-1), y)

    @staticmethod
    def _explode_f1_per_class(metrics_res, prefix):
        """Explode F1_per_class tensor into one scalar entry per class.

        torchmetrics' macro-F1 with ``average=None`` returns a per-class
        tensor that ``log_dict`` can't accept; explode into
        ``{prefix}F1_class_0/1/...`` keys and sanitize NaN → 0.0 (which
        happens for classes with no samples in the batch/epoch).
        """
        key = f"{prefix}F1_per_class"
        if key not in metrics_res:
            return metrics_res
        per_class = metrics_res.pop(key)
        for i, value in enumerate(per_class):
            metrics_res[f"{prefix}F1_class_{i}"] = (
                value if not torch.isnan(value) else 0.0
            )
        return metrics_res

    def _log_metrics(self, metrics_res, prefix):
        metrics_res = self._explode_f1_per_class(metrics_res, prefix)
        self.log_dict(
            metrics_res,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Forward + (mixup) target prep
        if self.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(x, y, alpha=self.mixup_alpha)
            y_hat = self(inputs)
        else:
            y_hat = self(x)
            if self.num_classes == 1:
                y_hat = y_hat.view(-1)

        # Edge case: batch size 1 with squeezed batch dim
        if x.shape[0] == 1 and len(y_hat.shape) == 1:
            y_hat = y_hat.unsqueeze(0)

        # SAM uses manual optimization with two forward/backward passes
        if self.sam:
            opt = self.optimizers()

            if self.mixup:
                loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
            else:
                loss = self.criterion(y_hat, y)
            self.manual_backward(loss)
            opt.first_step(zero_grad=True)

            if self.mixup:
                self.manual_backward(
                    mixup_criterion(
                        self.criterion, self(inputs), targets_a, targets_b, lam
                    )
                )
            else:
                second = self(x)
                if self.num_classes == 1:
                    second = second.view(-1)
                self.manual_backward(self.criterion(second, y))
            opt.second_step(zero_grad=True)
        else:
            if self.mixup:
                loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
            else:
                loss = self._compute_loss(y_hat, y)

        self.log(
            "Train/loss", loss,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )

        if torch.isnan(y_hat).any():
            print("######################################### Model predicts NaNs!")

        # Metrics
        if self.metric_computation_mode == "stepwise":
            metrics_res = self.train_metrics(y_hat, y)
            self._log_metrics(metrics_res, "Train/")
        elif self.metric_computation_mode == "epochwise":
            self._update_metrics(self.train_metrics, y_hat, y)

        if hasattr(self, "train_conf_mat"):
            self.train_conf_mat.update(y_hat, y)

        return loss

    # -----------------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)

        val_loss = self._compute_loss(y_hat, y)

        self.log(
            "Val/loss", val_loss,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )

        if self.metric_computation_mode == "stepwise":
            metrics_res = self.val_metrics(y_hat, y)
            self._log_metrics(metrics_res, "Val/")
        elif self.metric_computation_mode == "epochwise":
            self._update_metrics(self.val_metrics, y_hat, y)

        if hasattr(self, "val_conf_mat"):
            self.val_conf_mat.update(y_hat, y)
        if hasattr(self, "val_preds"):
            actual_batch_size = x.size(0)
            start_idx = batch_idx * self.trainer.val_dataloaders.batch_size
            idx = torch.arange(
                start_idx, start_idx + actual_batch_size, device=self.device
            )
            self.val_preds.update(y_hat.detach())
            self.val_labels.update(y.detach())
            self.val_indices.update(idx)

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)
        return y, y_hat

    # -----------------------------------------------------------------------
    # Epoch ends
    # -----------------------------------------------------------------------

    def _log_val_predictions_table(self, preds_all, labels_all):
        """Log a per-sample W&B prediction table.

        One GT column for multiclass / one per label for multilabel; per-class
        probability columns (softmax for multiclass, sigmoid for multilabel).
        """
        n_classes = preds_all.shape[-1]
        pred_cols = [f"Prob_{i}" for i in range(n_classes)]

        if self.subtask == "multilabel":
            gt_cols = [f"GT_{i}" for i in range(labels_all.shape[-1])]
            rows = [
                x.tolist() + torch.sigmoid(y).tolist()
                for x, y in zip(labels_all, preds_all)
            ]
        else:
            gt_cols = ["GT"]
            rows = [
                [int(x.item())] + F.softmax(y, dim=-1).tolist()
                for x, y in zip(labels_all, preds_all)
            ]

        table = wandb.Table(data=rows, columns=gt_cols + pred_cols)
        wandb.log({"Val Predictions": table})

    def on_validation_epoch_end(self) -> None:
        if self.metric_computation_mode == "epochwise":
            metrics_res = self.val_metrics.compute()
            self._log_metrics(metrics_res, "Val/")
            self.val_metrics.reset()

        if hasattr(self, "val_conf_mat"):
            self.val_conf_mat.save_state(self.trainer, "val")
            self.val_conf_mat.reset()

        if hasattr(self, "val_preds"):
            preds_all = self.val_preds.compute()
            labels_all = self.val_labels.compute()
            indices = self.val_indices.compute()

            if self.trainer.is_global_zero:
                # Sort by original index to preserve dataset order
                sorted_idx = torch.argsort(indices)
                preds_all = preds_all[sorted_idx]
                labels_all = labels_all[sorted_idx]

                if self.save_preds:
                    self._log_val_predictions_table(preds_all, labels_all)

            self.val_preds.reset()
            self.val_labels.reset()
            self.val_indices.reset()

        # SAM uses manual optimization and Lightning skips the ModelCheckpoint
        # callback's automatic save path; invoke it manually so checkpoints
        # actually land.
        if self.sam:
            for callback in self.trainer.callbacks:
                if isinstance(callback, L.pytorch.callbacks.ModelCheckpoint):
                    callback._save_topk_checkpoint(self.trainer, self.trainer.callback_metrics)
                    callback._save_last_checkpoint(self.trainer, self.trainer.callback_metrics)
                    print(f"[Checkpoint] Saved (SAM) for epoch {self.trainer.current_epoch}")

    def on_train_epoch_end(self) -> None:
        if self.metric_computation_mode == "epochwise":
            metrics_res = self.train_metrics.compute()
            self._log_metrics(metrics_res, "Train/")
            self.train_metrics.reset()

        if hasattr(self, "train_conf_mat"):
            self.train_conf_mat.save_state(self.trainer, "train")
            self.train_conf_mat.reset()

    # -----------------------------------------------------------------------
    # Init from scratch (when not pretrained)
    # -----------------------------------------------------------------------

    def on_train_start(self):
        if self.pretrained:
            return

        print("Initializing weights")
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.SyncBatchNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    # -----------------------------------------------------------------------
    # Optimizer / scheduler
    # -----------------------------------------------------------------------

    def _split_params_for_sawtooth(self):
        """Group named params into encoder vs cls_head."""
        encoder_params, cls_head_params = [], []
        for name, param in self.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            elif "cls_head" in name:
                cls_head_params.append(param)
        return encoder_params, cls_head_params, "cls_head"

    def _build_param_groups(self):
        """Param groups: plain iterable, or list of dicts for sawtooth fine-tuning."""
        if self.undecay_norm:
            model_params, norm_params = [], []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if "norm" in name or "bias" in name or "bn" in name:
                    norm_params.append(p)
                else:
                    model_params.append(p)
            base_params = [
                {"params": model_params},
                {"params": norm_params, "weight_decay": 0},
            ]
        else:
            base_params = self.parameters()

        if self.finetuning_method != "full_sawtooth":
            return base_params

        encoder_params, head_params, head_name = self._split_params_for_sawtooth()

        common_head = {
            "params": head_params,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "name": head_name,
        }
        common_enc = {
            "params": encoder_params,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "name": "encoder",
        }
        if self.optimizer in ("SGD", "Madgrad"):
            common_head["momentum"] = 0.9
            common_enc["momentum"] = 0.9
            if self.optimizer == "SGD":
                common_head["nesterov"] = self.nesterov
                common_enc["nesterov"] = self.nesterov

        return [common_head, common_enc]

    def _build_optimizer(self, params):
        """Construct optimizer (non-SAM path)."""
        if self.optimizer == "SGD":
            return torch.optim.SGD(
                params, lr=self.lr, momentum=0.9,
                weight_decay=self.weight_decay, nesterov=self.nesterov,
            )
        if self.optimizer == "Adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "AdamW":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "Rmsprop":
            return RMSpropTF(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "Madgrad":
            return MADGRAD(
                params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def _build_sam_optimizer(self, params):
        # ASAM paper suggests 10x larger rho for adaptive SAM than normal SAM
        rho = 0.5 if self.adaptive_sam else 0.05
        common = dict(
            adaptive=self.adaptive_sam, lr=self.lr,
            weight_decay=self.weight_decay, rho=rho,
        )

        if self.optimizer == "SGD":
            return SAM(
                params, torch.optim.SGD, momentum=0.9, nesterov=self.nesterov,
                **common,
            )
        if self.optimizer == "Madgrad":
            return SAM(params, MADGRAD, momentum=0.9, **common)
        if self.optimizer == "Adam":
            return SAM(params, torch.optim.Adam, **common)
        if self.optimizer == "AdamW":
            return SAM(params, torch.optim.AdamW, **common)
        if self.optimizer == "Rmsprop":
            return SAM(params, RMSpropTF, **common)
        raise ValueError(f"Unknown optimizer for SAM: {self.optimizer}")

    def _build_scheduler(self, optimizer):
        if not self.scheduler:
            return None

        if self.scheduler == "CosineAnneal":
            if self.warmstart == 0:
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.T_max,
                )
            if self.finetuning_method == "full_sawtooth":
                print(
                    f"[INFO] Using CosineAnnealingLR_DoubleWarmstart: "
                    f"warmstart1={self.warmstart}, warmstart2={self.warmstart2}, "
                    f"T_max={self.T_max}"
                )
                return CosineAnnealingLR_DoubleWarmstart(
                    optimizer, T_max=self.T_max,
                    warmstart1=self.warmstart, warmstart2=self.warmstart2,
                )
            print(
                f"[INFO] Using CosineAnnealingLR_Warmstart: "
                f"warmstart1={self.warmstart}, T_max={self.T_max}"
            )
            return CosineAnnealingLR_Warmstart(
                optimizer, T_max=self.T_max, warmstart=self.warmstart,
            )

        if self.scheduler == "Step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.epochs // 4, gamma=0.1,
            )
        if self.scheduler == "MultiStep":
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [self.epochs // 2, self.epochs * 3 // 4],
            )

        raise ValueError(f"Unknown scheduler: {self.scheduler}")

    def configure_optimizers(self):
        params = self._build_param_groups()

        if self.sam:
            optimizer = self._build_sam_optimizer(params)
        else:
            optimizer = self._build_optimizer(params)

        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return [optimizer]
        return [optimizer], [scheduler]


class ModelConstructor(BaseModel):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, x):
        return self.model(x)
