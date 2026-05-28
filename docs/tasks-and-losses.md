# Tasks and losses

The model block in each `configs/train_*.yaml` takes three related fields:

- `task`: `'Classification'` — currently the only supported task.
- `subtask`: `'multiclass'` (default) or `'multilabel'`. Controls how labels are interpreted at loss + metric time.
- `loss_fn`: name of the loss to use. When `null`, a sensible default is selected per subtask.

## Loss options

| `loss_fn`        | What it is                                                                 | Class weights? |
|------------------|----------------------------------------------------------------------------|----------------|
| `null` (default) | `CrossEntropyLoss` (multiclass) or `BCEWithLogitsLoss` (multilabel)        | No             |
| `focal`          | Focal loss with uniform alpha and gamma=2.0                                | No             |
| `weighted_focal` | Focal loss with per-class alpha = train-split class weights, gamma=1.5     | Yes (from train) |
| `weighted_ce`    | `F.cross_entropy(..., weight=class_weights, label_smoothing=...)`          | Yes (from train) |
| `topk10`         | Mean of the top-10% per-sample CE losses (hard-example mining)             | No             |

`weighted_focal` / `weighted_ce` pull their per-class weights from `datamodule.class_weights`, computed on the train split via the standard balanced formula `n_samples / (n_classes * n_samples_per_class)`, normalized to sum to `n_classes`. The criterion is constructed in `BaseModel.setup()` once the datamodule has run.

## Subtask differences

**Multiclass** (single class per sample):
- CSV `label` column is the integer class index (`0`, `1`, ..., `num_classes - 1`).
- Labels emitted as `torch.long`.
- Loss expects logits `[B, num_classes]`, targets `[B]`.
- Metrics receive `softmax(logits)`.

**Multilabel** (multiple labels per sample):
- CSV `label` column should encode a binary vector per row (project-specific encoding — see [data-csv-format](data-csv-format.md)).
- Loss is BCEWithLogits over `[B, num_labels]`.
- Metrics receive `sigmoid(logits)`.

## Example

```yaml
model:
  task: 'Classification'
  subtask: 'multiclass'
  loss_fn: weighted_ce        # picks up datamodule.class_weights automatically
  label_smoothing: 0.05
  num_classes: ${data.num_classes}
```

Set `data.num_classes` to the number of classes in your dataset.

To switch loss on the CLI without editing the file:

```bash
python scripts/train.py --config-name=train_classification model.loss_fn=focal
```
