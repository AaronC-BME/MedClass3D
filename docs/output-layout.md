# Output layout and run naming

A training run writes everything for one experiment to:

```
<output_dir>/<dataset_name>/<trainer.logger.name>/
├── Configs/
│   ├── config.yaml           <- Hydra config snapshot for the run
│   └── preprocessing.json    <- copied from the dataset's preprocessing.json
│                                (see docs/inference.md for how this is used)
└── folds/
    ├── 0/                    <- ModelCheckpoint files for fold 0
    ├── 1/                    <- fold 1, if running CV
    └── ...
```

Three fields in your `configs/train_*.yaml` drive this layout:

- **`output_dir`** — the output root. Defaults to `${oc.env:EXPERIMENT_LOCATION,"<fallback>"}`. Set `EXPERIMENT_LOCATION` in your shell (`export EXPERIMENT_LOCATION=/path/to/outputs`) and every config picks it up. Override per-run on the CLI with `output_dir=/some/path`.
- **`data.module.name`** — the dataset identifier (e.g. `Meningioma_T1c_3cls_5folds`). Used as the second path component and as the default W&B project name.
- **`trainer.logger.name`** — the experiment identifier. Used both as the W&B run name and as the on-disk run folder. Defaults to a `YYYY-MM-DD_HH-MM-SS` timestamp via `${make_group_name:}` if you set it that way; override it in the config or via CLI (`trainer.logger.name=MyExperiment1`) to get a stable, human-readable name.

For multi-fold runs, `cli.py` appends `_fold{k}` to the W&B run name per fold while keeping the on-disk folder name unchanged — so all folds share one parent directory but each appears as its own run in W&B.
