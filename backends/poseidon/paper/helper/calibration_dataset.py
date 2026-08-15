"""Reproducible calibration-dataset selection for MySVD_origin."""

import random

from torch.utils.data import Dataset


class TrajectorySubset(Dataset):
    """Select complete trajectories while preserving the source metadata."""

    def __init__(self, dataset, trajectory_indices):
        self.source_dataset = dataset
        self.selected_trajectory_indices = tuple(trajectory_indices)
        self.num_trajectories = len(self.selected_trajectory_indices)

        total_trajectories = int(dataset.num_trajectories)
        if total_trajectories <= 0:
            raise ValueError("The source dataset must contain at least one trajectory.")
        if len(set(self.selected_trajectory_indices)) != self.num_trajectories:
            raise ValueError("trajectory_indices must not contain duplicates.")
        if any(idx < 0 or idx >= total_trajectories for idx in self.selected_trajectory_indices):
            raise IndexError(
                f"trajectory_indices must be in [0, {total_trajectories})."
            )
        if len(dataset) % total_trajectories != 0:
            raise ValueError(
                f"Dataset length {len(dataset)} is not divisible by its "
                f"{total_trajectories} trajectories."
            )

        self.samples_per_trajectory = len(dataset) // total_trajectories
        self.sample_indices = tuple(
            trajectory_index * self.samples_per_trajectory + sample_offset
            for trajectory_index in self.selected_trajectory_indices
            for sample_offset in range(self.samples_per_trajectory)
        )

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        return self.source_dataset[self.sample_indices[index]]

    def __getattr__(self, name):
        # Fisher collection and the trainer need source metadata such as
        # constants, transpose, resolution, and channel information.
        if name == "source_dataset":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "source_dataset"), name)


def calibration_cache_tag(
    num_trajectories,
    random_calibration=False,
    calibration_seed=0,
    final_time=None,
):
    """Return a cache tag that prevents mixing selections, seeds, and times."""
    tag = f"traj{num_trajectories}"
    if random_calibration:
        tag += f"_random_seed{calibration_seed}"
    if final_time is not None:
        tag += f"_ftime{final_time}"
    return tag


def build_calibration_dataset(
    dataset_factory,
    *,
    dataset_name,
    num_trajectories,
    random_calibration=False,
    calibration_seed=0,
    **dataset_kwargs,
):
    """Build a fixed-prefix set or a seeded random complete-trajectory subset."""
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive.")

    # The full train split is required before we can sample trajectory IDs.
    source_num_trajectories = -1 if random_calibration else num_trajectories
    dataset = dataset_factory(
        dataset=dataset_name,
        which="train",
        num_trajectories=source_num_trajectories,
        **dataset_kwargs,
    )
    if not random_calibration:
        return dataset

    total_trajectories = int(dataset.num_trajectories)
    if num_trajectories > total_trajectories:
        raise ValueError(
            f"Cannot sample {num_trajectories} calibration trajectories from "
            f"{total_trajectories} training trajectories."
        )
    selected = random.Random(calibration_seed).sample(
        range(total_trajectories), num_trajectories
    )
    return TrajectorySubset(dataset, selected)


def describe_calibration_selection(dataset, random_calibration=False):
    if random_calibration:
        return (
            "random training trajectory indices: "
            f"{list(dataset.selected_trajectory_indices)}"
        )
    return f"fixed training trajectory prefix: [0, {dataset.num_trajectories})"
