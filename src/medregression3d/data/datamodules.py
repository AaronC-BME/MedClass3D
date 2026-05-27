import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule
from medregression3d.utils.io import Blosc2IO


class AgeReg_Data(Dataset):
    def __init__(
        self,
        img_dir,
        csv_file,
        split,
        fold,
        label_column="label",
        transform=None,
        train=True,
    ):
        super().__init__()
        """
        Age regression dataset, driven by a single CSV.

        The CSV must contain these columns:
            - image_name : subject/image identifier (matches folder name on disk)
            - split      : one of 'train', 'val', 'test'
            - fold       : integer fold index (0, 1, 2, ...)
            - <label>    : float label; rounded to int for ordinal regression
                           (column name configurable via `label_column`)
        """
        self.img_dir = Path(img_dir)
        self.train = train
        self.transform = transform

        df = pd.read_csv(csv_file)

        required = {"image_name", "split", "fold", label_column}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"CSV {csv_file} is missing required columns: {sorted(missing_cols)}. "
                f"Found columns: {list(df.columns)}"
            )

        # Normalize types so filtering is robust to int/str fold values
        df["fold"] = df["fold"].astype(int)
        df["split"] = df["split"].astype(str).str.lower()

        try:
            fold_int = int(fold)
        except (TypeError, ValueError):
            raise ValueError(f"Fold must be int-castable, got {fold!r}")

        split_norm = str(split).lower()
        valid_splits = {"train", "val", "test"}
        if split_norm not in valid_splits:
            raise ValueError(
                f"Unknown split '{split}'. Expected one of {sorted(valid_splits)}."
            )

        subset = df[(df["fold"] == fold_int) & (df["split"] == split_norm)]
        if len(subset) == 0:
            raise ValueError(
                f"No rows found in {csv_file} for fold={fold_int}, split='{split_norm}'. "
                f"Available folds: {sorted(df['fold'].unique())}, "
                f"available splits: {sorted(df['split'].unique())}."
            )

        self.img_files = subset["image_name"].astype(str).tolist()

        # Round float labels to ints for ordinal regression
        raw_labels = subset[label_column].astype(float).to_numpy()
        rounded = raw_labels.round().astype(int)
        self.labels = torch.tensor(rounded, dtype=torch.float)

        # Optional: verify files exist up front so failures surface at setup, not mid-epoch
        missing_files = [
            f for f in self.img_files
            if not (self.img_dir / f"{f}.b2nd").exists()
        ]
        if missing_files:
            missing_list = "\n  ".join(missing_files)
            raise FileNotFoundError(
                f"{len(missing_files)} image files not found under {self.img_dir}:\n  {missing_list}"
            )

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir,
            self.img_files[idx] + ".b2nd",
        )
        img, _ = Blosc2IO.load(img_path, mode="r")

        if self.train:
            img = self.transform(**{"image": torch.from_numpy(img[...])})["image"]
        else:
            img = self.transform.transforms[0](
                **{"image": torch.from_numpy(img[...])}
            )["image"]

        return img, self.labels[idx]

    def __len__(self):
        return len(self.img_files)


class AgeReg_DataModule(BaseDataModule):
    def __init__(self, img_dir, csv_file, label_column="label", **params):
        super().__init__(**params)
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.label_column = label_column

    def setup(self, stage: str):
        common = dict(
            img_dir=self.img_dir,
            csv_file=self.csv_file,
            label_column=self.label_column,
            fold=self.fold,
        )

        # Peek at the CSV once to find which splits actually exist for this
        # fold. Missing splits (e.g. no test set) are skipped silently rather
        # than raising during setup.
        df = pd.read_csv(self.csv_file)
        df["fold"] = df["fold"].astype(int)
        df["split"] = df["split"].astype(str).str.lower()
        available = set(df[df["fold"] == int(self.fold)]["split"].unique())

        if "train" in available:
            self.train_dataset = AgeReg_Data(
                **common,
                split="train",
                transform=self.train_transforms,
            )
        if "val" in available:
            self.val_dataset = AgeReg_Data(
                **common,
                split="val",
                transform=self.test_transforms,
                train=False,
            )
        if "test" in available:
            self.test_dataset = AgeReg_Data(
                **common,
                split="test",
                transform=self.test_transforms,
                train=False,
            )