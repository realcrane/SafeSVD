"""Seeded calibration-dataset helpers for VICON MySVD."""

from __future__ import annotations

import random

from torch.utils.data import DataLoader, Subset


def calibration_cache_tag(
    calibration_num: int,
    random_calibration: bool = False,
    calibration_seed: int = 0,
) -> str:
    """Keep caches produced from different calibration samples isolated."""
    tag = f"cali{calibration_num}"
    if random_calibration:
        tag += f"_random_seed{calibration_seed}"
    return tag


def build_calibration_loaders(
    datasets: dict,
    *,
    calibration_num: int,
    batch_size: int,
    num_workers: int,
    random_calibration: bool = False,
    calibration_seed: int = 0,
) -> dict:
    """
    Build the original prefix loaders or fixed, seeded random-subset loaders.

    For map-style datasets, the subset is materialized once and iterated without
    shuffling. Consequently, profiling, Fisher collection, rank selection, and
    compression all see exactly the same calibration examples.

    VICON also exposes stream-style datasets without ``__len__``. Those cannot
    be wrapped by ``Subset``; their random calibration data is instead selected
    by the seed passed to ``all_datasets`` by the caller.
    """
    if calibration_num <= 0:
        raise ValueError("calibration_num must be positive")

    rng = random.Random(calibration_seed)
    loaders = {}
    for dataset_type, dataset in datasets.items():
        calibration_dataset = dataset
        if random_calibration:
            try:
                dataset_size = len(dataset)
            except (AttributeError, NotImplementedError, TypeError):
                dataset_size = None

            if dataset_size is not None:
                sample_count = min(calibration_num, dataset_size)
                indices = rng.sample(range(dataset_size), sample_count)
                calibration_dataset = Subset(dataset, indices)

        loaders[dataset_type] = DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
    return loaders


def describe_calibration_selection(
    datasets: dict,
    *,
    calibration_num: int,
    random_calibration: bool = False,
    calibration_seed: int = 0,
) -> str:
    counts = {}
    stream_datasets = []
    for dataset_type, dataset in datasets.items():
        try:
            counts[dataset_type] = min(calibration_num, len(dataset))
        except (AttributeError, NotImplementedError, TypeError):
            counts[dataset_type] = "stream"
            stream_datasets.append(dataset_type)

    if random_calibration:
        if stream_datasets:
            return (
                "seeded random VICON stream datasets "
                f"(seed={calibration_seed}, stream={stream_datasets}, "
                f"map-style counts={counts})"
            )
        return f"seeded random subsets (seed={calibration_seed}, counts={counts})"
    return f"fixed dataset prefixes (counts={counts})"
