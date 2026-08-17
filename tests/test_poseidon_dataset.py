from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for Poseidon dataset tests")
class PoseidonDatasetTest(unittest.TestCase):
    def test_seeded_selection_keeps_complete_trajectories(self) -> None:
        from safesvd.adapters.poseidon_runtime import build_calibration_dataset

        class Source:
            num_trajectories = 4

            def __len__(self):
                return 12

            def __getitem__(self, index):
                return index

        def factory(**kwargs):
            self.assertEqual(kwargs["which"], "train")
            self.assertEqual(kwargs["num_trajectories"], -1)
            return Source()

        subset = build_calibration_dataset(
            factory,
            dataset_name="dummy",
            num_trajectories=2,
            random_calibration=True,
            calibration_seed=42,
        )
        self.assertEqual(len(subset), 6)
        self.assertEqual(len(subset.selected_trajectory_indices), 2)
        self.assertEqual(len({item // 3 for item in subset.sample_indices}), 2)
