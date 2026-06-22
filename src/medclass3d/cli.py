import os
import shutil
import sys
from contextlib import suppress
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from medclass3d.utils.parsing import make_omegaconf_resolvers


def _prepare_cfg(cfg):
    """Top-level cfg mutations done once before the CV loop."""
    if cfg.seed:
        seed_everything(cfg.seed)
        cfg.trainer.benchmark = False
        cfg.trainer.deterministic = "warn"

    # Hydra auto-creates main.log; remove it (W&B already captures everything).
    with suppress(FileNotFoundError):
        Path("./main.log").unlink()

    log_path = Path(cfg.trainer.logger.save_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    if cfg.trainer.devices > 1 and cfg.trainer.accelerator == "gpu":
        cfg.trainer.sync_batchnorm = True

    cfg.trainer.callbacks = [c for c in cfg.trainer.callbacks.values() if c]
    if not cfg.trainer["enable_checkpointing"]:
        cfg.trainer.callbacks = [
            c for c in cfg.trainer.callbacks
            if c["_target_"] != "lightning.pytorch.callbacks.ModelCheckpoint"
        ]


def _copy_preprocessing_sidecar(cfg):
    """Copy `preprocessing.json` from the dataset dir into the run's `Configs/`.

    The preprocess scripts (`preprocess_ct.py`, `preprocess_mri.py`) write a
    `preprocessing.json` next to their `preprocessed_b2nd/` output folder.
    Snapshotting it alongside `Configs/config.yaml` lets `predict_external.py`
    later replay the exact same preprocessing on new NIfTI files.

    Warn (but do not fail) if the sidecar is missing — preprocessed data created
    before the sidecar landed still trains correctly; external prediction just
    won't work for that run.
    """
    img_dir = Path(str(cfg.data.module.img_dir))
    sidecar_src = img_dir.parent / "preprocessing.json"
    if not sidecar_src.is_file():
        print(
            f"[warn] no preprocessing.json found at {sidecar_src} — "
            "predict_external.py will not be able to replay this run's "
            "preprocessing on new NIfTI files. Re-run preprocess_ct.py / "
            "preprocess_mri.py to generate one."
        )
        return

    dest_dir = Path(str(cfg.output_subdir))
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sidecar_src, dest_dir / "preprocessing.json")
    print(f"[info] copied preprocessing sidecar: {sidecar_src} -> {dest_dir / 'preprocessing.json'}")


def _set_checkpoint_dir(cfg, base_name):
    """Point ModelCheckpoint at <output_dir>/<dataset>/<base_name>/folds/<fold>."""
    if not cfg.trainer["enable_checkpointing"]:
        return
    for cb in cfg.trainer.callbacks:
        if cb["_target_"] == "lightning.pytorch.callbacks.ModelCheckpoint":
            cb["dirpath"] = os.path.join(
                str(cfg.output_dir),
                str(cfg.data.module.name),
                str(base_name),
                "folds",
                str(cfg.data.module.fold),
            )


def _log_hyperparams(trainer, cfg):
    """Strip non-loggable fields and forward the rest to the logger."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    cfg_dict["model"].pop("_target_")
    cfg_dict["model"]["model"] = cfg_dict["model"].pop("name")
    trainer.logger.log_hyperparams(cfg_dict["model"])

    data_module = cfg_dict["data"]["module"]
    data_module.pop("_target_")
    for key in ("train_transforms", "test_transforms"):
        if data_module.get(key) is not None:
            data_module[key] = ".".join(
                data_module[key]["_target_"].split(".")[-2:]
            )
    data_module.pop("name")
    trainer.logger.log_hyperparams(data_module)

    trainer_cfg = cfg_dict["trainer"]
    for key in (
        "_target_", "callbacks", "enable_checkpointing",
        "enable_progress_bar", "logger", "num_sanity_val_steps",
    ):
        trainer_cfg.pop(key, None)
    trainer.logger.log_hyperparams(trainer_cfg)


CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


def _require_config_name():
    """Each training run is driven by one self-contained configs/train_*.yaml,
    selected with --config-name=<name>. Emit a friendly error listing the
    available choices if the user forgot the flag."""
    if any(
        arg == "--config-name" or arg.startswith("--config-name=")
        for arg in sys.argv[1:]
    ):
        return

    configs = sorted(p.stem for p in Path(CONFIG_DIR).glob("train_*.yaml"))
    choices = "\n  ".join(configs) if configs else "(none yet — create one in configs/)"
    print(
        "ERROR: --config-name is required.\n"
        "\n"
        "Launch with:\n"
        "  python scripts/train.py --config-name=<name>\n"
        "\n"
        f"Available configs in {CONFIG_DIR}:\n  {choices}\n",
        file=sys.stderr,
    )
    sys.exit(2)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=None)
def _hydra_main(cfg):
    _prepare_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))

    # Snapshot the dataset's preprocessing.json into this run's Configs/ so
    # predict_external.py can replay the same preprocessing on new NIfTI files.
    _copy_preprocessing_sidecar(cfg)

    # `trainer.logger.name` is the source of truth for both the W&B run name
    # and the on-disk run folder. Capture the base before the fold loop so
    # all folds share one parent dir even though their W&B names differ.
    base_name = cfg.trainer.logger.name

    for k in range(cfg.data.cv.k):
        if cfg.data.cv.k > 1:
            cfg.data.module.fold = k
        elif cfg.data.module.fold is None:
            cfg.data.module.fold = "0"

        # Append _fold<k> to the W&B run name for multi-fold CV, and also for a
        # single-fold rerun (cv.k=1 with an explicit data.module.fold) so reruns
        # are named consistently. base_name (captured before the loop) still drives
        # the on-disk checkpoint dir, so checkpoints land in <run>/folds/<k>.
        if cfg.data.cv.k > 1 or cfg.data.module.fold is not None:
            cfg.trainer.logger.name = f"{base_name}_fold{cfg.data.module.fold}"

        _set_checkpoint_dir(cfg, base_name)

        trainer = instantiate(cfg.trainer)
        model = instantiate(cfg.model)

        if cfg.model.compile:
            model = torch.compile(model, mode="default")
        dataset = instantiate(cfg.data).module

        _log_hyperparams(trainer, cfg)

        if cfg.val_only:
            trainer.validate(model, dataset)
        else:
            trainer.fit(model, dataset, ckpt_path=cfg.get("ckpt_path", None))

        wandb.finish()


def main():
    """Entry point used by both ``scripts/train.py`` and the ``medclass-train``
    console script. Sets up env vars + OmegaConf resolvers, checks for
    ``--config-name``, and delegates to the Hydra-wrapped runner."""
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    make_omegaconf_resolvers()
    _require_config_name()
    _hydra_main()


if __name__ == "__main__":
    main()
