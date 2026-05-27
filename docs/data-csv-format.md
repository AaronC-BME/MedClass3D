# Data CSV format

Splits, labels, and fold assignments are all driven by a single CSV file per dataset. The CSV must contain these columns:

| Column       | Type   | Description                                                                 |
|--------------|--------|-----------------------------------------------------------------------------|
| `image_name` | string | Subject/image identifier. Must match the `.b2nd` filename stem on disk (no extension). |
| `split`      | string | One of `train`, `val`, `test`.                                              |
| `fold`       | int    | Fold index (`0`, `1`, `2`, ...). Used for cross-validation.                 |
| `<label>`    | float  | Target value. Column name is configurable (default `label`); rounded to int internally for ordinal regression. |

Example:

```csv
image_name,split,label,fold
sub-001,train,42.3,0
sub-002,val,67.8,0
sub-003,test,55.1,0
sub-004,train,29.5,1
sub-005,val,71.0,1
```

Notes:
- For ordinal regression, labels are rounded to the nearest integer (`int(round(x))`) and must fall in `[0, num_classes - 1]`.
- Splits are **global** across folds — an image's `split` value applies regardless of which fold is being trained. If you need per-fold splits, you'll need to extend the CSV schema.
- The label column name can be customized via the `label_column` field in the data config (e.g., `label_column: age`).
- **For k-fold cross-validation**, include one row per `(image, fold)` combination, with the `split` column indicating that image's role in that particular fold. With `cv.k=5` in the config, training is launched 5 times, each time filtering the CSV to one fold.
