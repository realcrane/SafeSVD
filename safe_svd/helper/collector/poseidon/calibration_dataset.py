import random

from torch.utils.data import Dataset


class TrajectorySubset(Dataset):
    """Select complete trajectories while preserving the source dataset metadata."""

    def __init__(self, dataset, trajectory_indices):
        self.source_dataset = dataset
        self.selected_trajectory_indices = tuple(trajectory_indices)
        self.num_trajectories = len(self.selected_trajectory_indices)

        total_trajectories = int(dataset.num_trajectories)
        if total_trajectories <= 0:
            raise ValueError("The source dataset must contain at least one trajectory.")
        if len(set(self.selected_trajectory_indices)) != self.num_trajectories:
            raise ValueError("trajectory_indices must not contain duplicates.")
        if any(
            idx < 0 or idx >= total_trajectories
            for idx in self.selected_trajectory_indices
        ):
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
            trajectory_idx * self.samples_per_trajectory + sample_offset
            for trajectory_idx in self.selected_trajectory_indices
            for sample_offset in range(self.samples_per_trajectory)
        )

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self.source_dataset[self.sample_indices[idx]]

    def __getattr__(self, name):
        # Trainer and Fisher collection consume metadata such as resolution,
        # normalization constants, and transpose from the original dataset.
        if name == "source_dataset":
            raise AttributeError(name)
        source_dataset = object.__getattribute__(self, "source_dataset")
        return getattr(source_dataset, name)


def calibration_cache_tag(
    num_trajectories,
    random_calibration=False,
    calibration_seed=0,
    final_time=None,
):
    """Return a cache/output tag that isolates trajectory, seed, and time settings."""
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
    """
    Build the fixed prefix calibration set or a seeded random trajectory subset.

    Random sampling loads the complete training split first, then samples trajectory
    indices without replacement. All samples/time pairs belonging to a selected
    trajectory remain together.
    """
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive.")

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

    rng = random.Random(calibration_seed)
    selected = rng.sample(range(total_trajectories), num_trajectories)
    return TrajectorySubset(dataset, selected)


def describe_calibration_selection(dataset, random_calibration=False):
    if random_calibration:
        selected = list(dataset.selected_trajectory_indices)
        return f"random training trajectory indices: {selected}"
    return f"fixed training trajectory prefix: [0, {dataset.num_trajectories})"

