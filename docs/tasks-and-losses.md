# Tasks and losses

The model block in each `configs/train_*.yaml` takes two related fields:

- `task`: one of `'Regression'`, `'Ordinal_Regression'`.
- `loss_fn`: name of the loss to use. When `null`, a sensible default is selected per task (see table below).

| `task`              | `loss_fn: null` (default) | Other valid `loss_fn` values                                                                                  |
|---------------------|---------------------------|----------------------------------------------------------------------------------------------------------------|
| `Regression`        | `MSELoss`                 | *(none)*                                                                                                       |
| `Ordinal_Regression`| `coral_loss`              | `focal`, `topk10`, `topk20`, `bce_focal`, `bce_topk10`, `bce_topk20`, `weighted_bce`, `bce_mae`                |

Example ordinal-regression model block:

```yaml
model:
  task: 'Ordinal_Regression'
  loss_fn: null   # uses CORAL loss by default
  # ...
```

For ordinal regression, set `num_classes` in the `data:` block to the number of ordinal levels (e.g. `100` for ages 0–99). The CORAL head emits `num_classes - 1` logits.

To switch loss on the CLI without editing the file:

```bash
python scripts/train.py --config-name=train_age_ord_reg model.loss_fn=focal
```
