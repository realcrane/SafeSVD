"""Calibration replay must survive process RNG changes and prior passes."""
import importlib.util
import os
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import IterDataPipe, TensorDataset, get_worker_info

from safe_svd.helper.collector import vicon_calibration_dataset as safe
from safe_svd.helper.collector import vicon_rng


class Stream(IterDataPipe):
    """Exercise upstream's mutable paths and two independent dataset RNGs."""
    def __init__(self, category="NS2D", root="/one", reverse=False):
        self.category = category
        self.cfg = SimpleNamespace(types={category: SimpleNamespace(folder=root)})
        self.dp = [f"{root}/{i}.h5" for i in range(12)]
        if reverse:
            self.dp.reverse()
        self.rng = self.tc_rng = None
        self.num_workers = 0

    def __iter__(self):
        if self.rng is None:
            seed = int(np.random.randint(10**9))
            self.rng = np.random.default_rng(seed)
            self.tc_rng = torch.Generator().manual_seed(seed)
        worker = get_worker_info()
        index = worker.id if worker else 0
        paths = self.dp if self.num_workers <= 1 else np.array_split(self.dp, self.num_workers)[index]
        self.rng.shuffle(paths)
        for path in paths:
            yield torch.tensor([int(Path(path).stem), self.rng.random(),
                                torch.rand((), generator=self.tc_rng).item()], dtype=torch.float64)


def build(module=safe, *, seed=42, workers=0, root="/one", reverse=False):
    source = Stream(root=root, reverse=reverse)
    loaders = module.build_calibration_loaders({"NS2D": source}, calibration_num=12,
                                              batch_size=3, num_workers=workers, calibration_seed=seed)
    return source, loaders


def flatten(loaders):
    return torch.cat(list(loaders["NS2D"]))


class CalibrationSamplingTests(unittest.TestCase):
    def test_partial_and_complete_passes_replay_without_changing_source(self):
        source, loaders = build()
        paths = source.dp[:]
        expected = flatten(loaders)
        next(iter(loaders["NS2D"]))  # Interrupted profiling/Fisher traversal.
        random.seed(991); np.random.seed(881); torch.manual_seed(771)
        torch.testing.assert_close(flatten(loaders), expected, rtol=0, atol=0)
        torch.testing.assert_close(flatten(loaders), expected, rtol=0, atol=0)
        self.assertEqual(source.dp, paths)
        self.assertIsNone(source.rng)
        self.assertIsNone(source.tc_rng)

    def test_root_order_and_global_rng_do_not_change_calibration(self):
        _, a = build(root="/first", reverse=True)
        expected = flatten(a)
        np.random.seed(314); torch.manual_seed(159)
        _, b = build(root="/second", reverse=False)
        torch.testing.assert_close(flatten(b), expected, rtol=0, atol=0)
        _, c = build(seed=43)
        self.assertFalse(torch.equal(flatten(c), expected))

    def test_sampler_does_not_advance_global_rngs(self):
        random.seed(7); np.random.seed(8); torch.manual_seed(9)
        before = random.getstate(), np.random.get_state(), torch.get_rng_state()
        _, loaders = build()
        flatten(loaders)
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        self.assertEqual(np.random.get_state()[2:], before[1][2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_worker_replay_and_disjoint_file_shards(self):
        _, loaders = build(workers=2)
        expected = flatten(loaders)
        torch.testing.assert_close(flatten(loaders), expected, rtol=0, atol=0)
        self.assertEqual(sorted(expected[:, 0].tolist()), list(range(12)))

    def test_resume_capture_restores_loader_with_repeatable_stream(self):
        _, loaders = build()
        state = vicon_rng.capture(loaders)
        expected = flatten(loaders)
        next(iter(loaders["NS2D"]))
        vicon_rng.restore(loaders, state)
        torch.testing.assert_close(flatten(loaders), expected, rtol=0, atol=0)

    def test_map_style_subsets_remain_repeatable(self):
        source = TensorDataset(torch.arange(100))
        def rows(seed):
            loader = safe.build_calibration_loaders({"map": source}, calibration_num=12, batch_size=3,
                                                    num_workers=0, random_calibration=True, calibration_seed=seed)["map"]
            return torch.cat([batch[0] for batch in loader])
        self.assertTrue(torch.equal(rows(42), rows(42)))
        self.assertFalse(torch.equal(rows(42), rows(43)))

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Reference checkout required")
    def test_reference_helper_matches_safe_across_stages(self):
        path = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]).parent / "vicon/MySVD/helper/calibration_dataset.py"
        spec = importlib.util.spec_from_file_location("reference_calibration_sampling", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        for seed in (42, 137):
            _, a = build(seed=seed, reverse=True)
            _, b = build(reference, seed=seed, root="/reference")
            expected = flatten(a)
            for _stage in ("profile", "fisher", "compression"):
                torch.testing.assert_close(flatten(b), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
