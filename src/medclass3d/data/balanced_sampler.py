import math
import random

import torch
from torch.utils.data import DataLoader, Sampler


class KClassBalancedBatchSampler(Sampler):
    """Each batch has (as close as possible) equal samples from each class.

    Works with any K >= 2. Samples with replacement, so minority classes are
    naturally oversampled. Labels must be ints in ``[0..K-1]``; the sampler
    pulls them from ``dataset[i][1]`` once at construction time.
    """

    def __init__(self, dataset, batch_size, drop_last=False, generator=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.generator = generator

        real_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
        labels = [int(real_dataset[i][1]) for i in range(len(real_dataset))]

        self.idx_by_class = {}
        for i, y in enumerate(labels):
            self.idx_by_class.setdefault(y, []).append(i)

        self.classes = sorted(self.idx_by_class.keys())
        self.K = len(self.classes)
        if self.K < 2:
            raise ValueError("KClassBalancedBatchSampler requires at least 2 classes.")
        for c in self.classes:
            if len(self.idx_by_class[c]) == 0:
                raise ValueError(f"Class {c} has no samples.")

        n = len(real_dataset)
        self.num_batches = n // self.batch_size if drop_last else math.ceil(n / self.batch_size)

        self.base = self.batch_size // self.K
        self.rem = self.batch_size % self.K

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random()
        if self.generator is not None:
            seed = int(torch.empty((), dtype=torch.int64).random_(generator=self.generator).item())
            rng.seed(seed)

        for _ in range(self.num_batches):
            extra_classes = rng.sample(self.classes, self.rem) if self.rem > 0 else []
            batch_idx = []
            for c in self.classes:
                need = self.base + (1 if c in extra_classes else 0)
                src = self.idx_by_class[c]
                picks = [rng.choice(src) for _ in range(need)]
                batch_idx.extend(picks)
            rng.shuffle(batch_idx)
            yield batch_idx


def make_k_class_balanced_trainloader(
    dataset,
    batch_size,
    num_workers=4,
    pin_memory=True,
    worker_init_fn=None,
    persistent_workers=True,
    drop_last=False,
):
    g = torch.Generator()
    g.manual_seed(torch.initial_seed() % (2**31))
    batch_sampler = KClassBalancedBatchSampler(
        dataset, batch_size=batch_size, drop_last=drop_last, generator=g,
    )
    return DataLoader(
        dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        batch_sampler=batch_sampler,
    )
