# Data CSV format

Splits, labels, and fold assignments are all driven by a single CSV file per dataset. The CSV must contain these columns:

| Column       | Type   | Description                                                                 |
|--------------|--------|-----------------------------------------------------------------------------|
| `image_name` | string | Subject/image identifier. Must match the `.b2nd` filename stem on disk (no extension). |
| `split`      | string | One of `train`, `val`, `test`.                                              |
| `fold`       | int    | Fold index (`0`, `1`, `2`, ...). Used for cross-validation.                 |
| `<label>`    | int    | Integer class index (column name configurable; default `label`). Must fall in `[0, num_classes - 1]`. |

Example:

```csv
image_name,split,label,fold
sub-001,train,1,0
sub-002,val,0,0
sub-003,test,2,0
sub-004,train,1,1
sub-005,val,0,1
```

Notes:
- The label column is read as `int`; labels must fall in `[0, num_classes - 1]`. Set `data.num_classes` in your config to match.
- Splits are **global** across folds — an image's `split` value applies regardless of which fold is being trained. If you need per-fold splits, you'll need to extend the CSV schema.
- The label column name can be customized via the `label_column` field in the data config (e.g., `label_column: pathology`).
- **For k-fold cross-validation**, include one row per `(image, fold)` combination, with the `split` column indicating that image's role in that particular fold. With `cv.k=5` in the config, training is launched 5 times, each time filtering the CSV to one fold.
- **Class weights** are computed automatically on the train split (per fold) using the balanced formula `n_samples / (n_classes * n_samples_per_class)`. They're exposed as `datamodule.class_weights` and picked up by `loss_fn: weighted_ce` / `loss_fn: weighted_focal` — you don't need to pre-compute them.

## Multilabel

For `subtask: 'multilabel'`, the framework expects the `label` column to be a per-row binary vector. The default `Class_Data` reads it via `pd.read_csv(...)[label_column].astype(int)`, which treats each row as a scalar — you'll need to either:
- Use one binary column per label and pass a sub-dataclass that stacks them, or
- Subclass `Class_Data` and override `__init__` to parse a serialized vector (e.g., `"1,0,1,0"`).

The multiclass path is the default and needs no schema changes.
