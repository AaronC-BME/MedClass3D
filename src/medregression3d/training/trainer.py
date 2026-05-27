import torch
import torch.nn.functional as F
import wandb
import lightning as L
from madgrad import MADGRAD
from timm.optim import RMSpropTF
from torchmetrics import MetricCollection
from torchmetrics.aggregation import CatMetric

from medregression3d.data.mixup import mixup_criterion, mixup_data
from medregression3d.evaluation.metrics import _build_regression_metrics
from medregression3d.models.losses import (
    VALID_TASKS,
    _build_criterion,
    coral_loss,
    label_to_levels,
)
from medregression3d.training.optim import (
    CosineAnnealingLR_DoubleWarmstart,
    CosineAnnealingLR_Warmstart,
)
from medregression3d.training.sam import SAM


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
        self.loss_fn = loss_fn  # None => task default

        # --- Metrics ----------------------------------------------------------
        self.metric_computation_mode = metric_computation_mode
        self.result_plot_setting = result_plot

        metrics_dict = _build_regression_metrics(metrics)

        # Result-plotting bookkeeping
        if self.result_plot_setting in ("val", "all"):
            self.val_pred_list = []
            self.val_label_list = []
        if self.result_plot_setting == "all":
            self.train_pred_list = []
            self.train_label_list = []

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
        self.criterion = _build_criterion(
            self.task, self.loss_fn, self.label_smoothing,
        )

    # -----------------------------------------------------------------------
    # Forward / setup
    # -----------------------------------------------------------------------

    def forward(self, x):
        pass

    def setup(self, stage=None):
        self.level_weights = None
        if self.loss_fn == "weighted_bce":
            print("Setting up level weights for Ordinal Regression Weighted BCE")
            self.level_weights = self.trainer.datamodule.level_weights.to(self.device)

    # -----------------------------------------------------------------------
    # Step helpers
    # -----------------------------------------------------------------------

    @property
    def _is_ordinal(self):
        return self.task == "Ordinal_Regression"

    def _forward_logits(self, x):
        """Run forward and return logits only (handles tuple-returning ordinal heads)."""
        out = self(x)
        if self._is_ordinal and isinstance(out, tuple):
            return out[0]
        return out

    def _compute_loss(self, y_hat, y):
        """Standard (non-mixup, non-SAM) loss given logits and labels."""
        if self._is_ordinal:
            levels = label_to_levels(y, self.num_classes)
            if self.loss_fn == "bce_mae":
                return self.criterion(
                    y_hat, levels, y, importance_weights=self.level_weights
                )
            if self.loss_fn == "weighted_bce":
                return self.criterion(
                    y_hat, levels, importance_weights=self.level_weights
                )
            return self.criterion(y_hat, levels)

        return self.criterion(y_hat, y.float())

    def _update_metrics(self, metrics_obj, y_hat, y):
        """Update an epochwise MetricCollection with predictions in the right form."""
        if self._is_ordinal:
            pred_classes = (torch.sigmoid(y_hat.detach()) > 0.5).int().sum(dim=1)
            metrics_obj.update(pred_classes, y)
        else:
            metrics_obj.update(y_hat.view(-1).detach(), y.view(-1))

    def _log_metrics(self, metrics_res, prefix):
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
            y_hat = self._forward_logits(x)
            if (not self._is_ordinal) and self.num_classes == 1:
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

        # Optional plot bookkeeping
        if hasattr(self, "train_pred_list"):
            self.train_pred_list.extend(y_hat)
            self.train_label_list.extend(y)

        return loss

    # -----------------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self._forward_logits(x)
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
        """Log per-sample prediction table to W&B based on task type."""
        if self.task == "Regression":
            data = [[x.item(), y.item()] for x, y in zip(labels_all, preds_all)]
            table = wandb.Table(data=data, columns=["GT", "Pred"])
            wandb.log({"Val Predictions": table})

        elif self._is_ordinal:
            binary_preds = (torch.sigmoid(preds_all) > 0.5).int()
            expected_ages = binary_preds.sum(dim=1)
            data = [[x.item(), y.item()] for x, y in zip(labels_all, expected_ages)]
            table = wandb.Table(data=data, columns=["GT", "Pred"])
            wandb.log({"Val Predictions": table})

        else:
            raise NotImplementedError

    def _log_val_scatterplot(self, preds_all, labels_all):
        """Scatterplot of GT vs prediction for Regression / Ordinal Regression."""
        if self.task == "Regression":
            data = [[x, y] for (x, y) in zip(labels_all, preds_all)]
        elif self._is_ordinal:
            binary_preds = (torch.sigmoid(preds_all) > 0.5).int()
            expected_ages = binary_preds.sum(dim=1)
            data = [[x, y] for (x, y) in zip(labels_all, expected_ages)]
        else:
            return

        table = wandb.Table(data=data, columns=["Ground Truth", "Prediction"])
        wandb.log({
            "Val Scatterplot": wandb.plot.scatter(
                table, "Ground Truth", "Prediction", "Validation Scatterplot",
            )
        })

    def on_validation_epoch_end(self) -> None:
        if self.metric_computation_mode == "epochwise":
            metrics_res = self.val_metrics.compute()
            self._log_metrics(metrics_res, "Val/")
            self.val_metrics.reset()

        if hasattr(self, "val_preds"):
            preds_all = self.val_preds.compute()
            labels_all = self.val_labels.compute()
            indices = self.val_indices.compute()

            if self.trainer.is_global_zero:
                # Sort by original index to preserve dataset order
                sorted_idx = torch.argsort(indices)
                preds_all = preds_all[sorted_idx]
                labels_all = labels_all[sorted_idx]

                self._log_val_scatterplot(preds_all, labels_all)

                if self.save_preds:
                    self._log_val_predictions_table(preds_all, labels_all)

            self.val_preds.reset()
            self.val_labels.reset()
            self.val_indices.reset()

    def on_train_epoch_end(self) -> None:
        if self.metric_computation_mode == "epochwise":
            metrics_res = self.train_metrics.compute()
            self._log_metrics(metrics_res, "Train/")
            self.train_metrics.reset()

        if hasattr(self, "train_pred_list"):
            data = [
                [x, y] for (x, y) in zip(self.train_label_list, self.train_pred_list)
            ]
            table = wandb.Table(data=data, columns=["Ground Truth", "Prediction"])
            wandb.log({
                "Train Scatterplot": wandb.plot.scatter(
                    table, "Ground Truth", "Prediction", "Train Scatterplot",
                )
            })
            self.train_pred_list = []
            self.train_label_list = []

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
        """Group named params into encoder vs head (cls_head or reg_head)."""
        encoder_params, cls_head_params, reg_head_params = [], [], []
        for name, param in self.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            elif "cls_head" in name:
                cls_head_params.append(param)
            elif "reg_head" in name:
                reg_head_params.append(param)

        if cls_head_params:
            return encoder_params, cls_head_params, "cls_head"
        return encoder_params, reg_head_params, "reg_head"

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
