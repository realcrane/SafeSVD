"""MPP sampling and compression checks against a read-only research snapshot.

Set MPP_REFERENCE_SCRIPT to the original mpp/MySVD/mysvd.py to enable the
reference comparisons. No original modules are imported by the runtime.
"""

import copy
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from safe_svd.compress import compress
from safe_svd.helper.collector.mpp_collector import iter_mpp_batches
from safe_svd.rank import allocate_ranks
from test_unified_profiling import source_function


class MPPFileFixture(torch.utils.data.Dataset):
    def __init__(self, category, seed, scale):
        generator = torch.Generator().manual_seed(seed)
        self.category = category
        self.values = torch.randn(23, 2, 4, 3, 3, generator=generator) * scale
        self.targets = torch.randn(23, 4, 3, 3, generator=generator)

    def get_name(self):
        return self.category

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], torch.tensor([index]), self.targets[index]


class MPPSequentialFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 7)
        self.middle = nn.Linear(7, 6)
        self.last = nn.Linear(6, 4)

    def forward(self, inp, labels, bcs):
        value = (inp[-1] + 0.15 * inp[0]).permute(0, 2, 3, 1)
        value = torch.tanh(self.first(value))
        value = torch.tanh(self.middle(value))
        return self.last(value).permute(0, 3, 1, 2)


def mpp_case(batch_size=2):
    torch.manual_seed(183)
    model = MPPSequentialFixture().eval()
    categories = ("swe", "incompNS", "compNS", "diffre2d")
    files = [MPPFileFixture(name, 30 + index, scale) for index, (name, scale) in
             enumerate(zip(categories, (0.1, 1.0, 2.0, 8.0)))]
    dataset = SimpleNamespace(
        sub_dsets=[SimpleNamespace(get_per_file_dsets=lambda: files)],
        subset_dict={name: list(range(4)) for name in categories},
    )
    params = SimpleNamespace(batch_size=batch_size, num_data_workers=0)
    options = {"params": params, "calibration_seed": 42,
               "batches_per_dset": 16 / batch_size}
    left = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            factor = torch.diag(torch.linspace(0.7, 1.8, layer.out_features))
            left[name] = factor, torch.linalg.inv(factor)
    return model, dataset, options, left


def reference_path():
    direct = os.environ.get("MPP_REFERENCE_SCRIPT")
    if direct:
        return Path(direct)
    shared = os.environ.get("SAFE_SVD_REFERENCE_ROOT")
    return Path(shared).parent / "mpp/MySVD/mysvd.py" if shared else None


class MPPStageEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = reference_path()
        if path is None or not path.is_file():
            raise unittest.SkipTest("Set MPP_REFERENCE_SCRIPT to the original MySVD script")
        cls.reference = {
            "torch": torch, "nn": nn, "copy": copy, "os": os,
            "DataLoader": torch.utils.data.DataLoader,
            "tqdm": lambda values: values,
            "StopForwardException": type("StopForwardException", (Exception,), {}),
            "rearrange": lambda value, _: value.permute(1, 0, 2, 3, 4),
        }
        for name in ("get_module_by_name", "set_module_by_name", "svd_whitener",
                     "iter_mpp_batches", "profile_input_gram",
                     "get_global_greedy_rank_dict_mpp", "mysvd_compress_mpp"):
            source_function(path, name, cls.reference)

    def assert_equivalent(self, actual, expected, dataset):
        for name in ("first", "middle", "last"):
            actual_layer, expected_layer = actual.get_submodule(name), expected.get_submodule(name)
            actual_weight = actual_layer[1].weight @ actual_layer[0].weight
            expected_weight = expected_layer[1].weight @ expected_layer[0].weight
            torch.testing.assert_close(actual_weight, expected_weight, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(actual_layer[1].bias, expected_layer[1].bias, atol=0, rtol=0)
        for subset in dataset.sub_dsets[0].get_per_file_dsets():
            inputs = subset.values[:3].permute(1, 0, 2, 3, 4)
            labels, bcs = torch.arange(4).expand(3, -1), torch.zeros(3, 1)
            torch.testing.assert_close(actual(inputs, labels, bcs), expected(inputs, labels, bcs),
                                       atol=2e-6, rtol=2e-5)

    def test_calibration_16_preserves_exact_per_file_sample_sequence(self):
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                _, dataset, options, _ = mpp_case(batch_size)
                expected = list(self.reference["iter_mpp_batches"](
                    dataset, options["params"], "cpu", options["batches_per_dset"], 42))
                actual = list(iter_mpp_batches(dataset, "cpu", options))
                self.assertEqual(len(actual), 4 * 16 // batch_size)
                self.assertEqual(len(actual), len(expected))
                counts = {}
                for tagged, (inp, labels, bcs, target, subset) in zip(actual, expected):
                    self.assertEqual(tagged.category, subset.get_name())
                    counts[tagged.category] = counts.get(tagged.category, 0) + target.shape[0]
                    for first, second in zip(tagged.payload, (inp, labels, bcs, target)):
                        torch.testing.assert_close(first, second, atol=0, rtol=0)
                self.assertEqual(set(counts.values()), {16})

    def test_fractional_batch_limit_matches_research_ceiling(self):
        # The original compares an integer batch counter with cali / batch.
        # It therefore includes a whole batch even when that ratio is < 1.
        for batch_size in (3, 20):
            with self.subTest(batch_size=batch_size):
                _, dataset, options, _ = mpp_case(batch_size)
                limit = options["batches_per_dset"]
                expected = list(self.reference["iter_mpp_batches"](
                    dataset, options["params"], "cpu", limit, 42))
                actual = list(iter_mpp_batches(dataset, "cpu", options))
                self.assertEqual(len(actual), 4 * math.ceil(limit))
                self.assertEqual(len(actual), len(expected))
                for tagged, (inp, labels, bcs, target, subset) in zip(actual, expected):
                    self.assertEqual(tagged.category, subset.get_name())
                    for first, second in zip(tagged.payload, (inp, labels, bcs, target)):
                        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_legacy_budget_boundary_preserves_research_score_tie(self):
        model = nn.ModuleDict({"first": nn.Linear(8, 8), "second": nn.Linear(10, 8)})
        right = {name: (torch.eye(layer.in_features), torch.eye(layer.in_features))
                 for name, layer in model.items()}
        # Python pow() and Tensor.square() need not have the same last bit,
        # and pow() itself differs across system math libraries. Find a
        # boundary on this runtime rather than hard-code one platform's
        # winner. SVD is stubbed identically so only score arithmetic varies.
        candidates = torch.linspace(0.1, 3., 16384, dtype=torch.float64)
        neighbors = candidates * math.sqrt(18 / 16)
        old_first = (candidates.square() / 16).tolist()
        boundary = None
        for direction in (-math.inf, math.inf):
            adjusted = torch.nextafter(neighbors, torch.full_like(neighbors, direction))
            old_second = (adjusted.square() / 18).tolist()
            for x, y, first, second in zip(candidates.tolist(), adjusted.tolist(), old_first, old_second):
                if (x ** 2 / 16 >= y ** 2 / 18) != (first >= second):
                    boundary = x, y, first >= second
                    break
            if boundary is not None:
                break
        # Some runtimes may round both operations identically for the search
        # grid. Still compare the two allocators on an identical near tie.
        x, y = boundary[:2] if boundary else (1.4421575637362891, 1.529639089286101)
        singular_values = [
            torch.tensor([10., x, 0., 0., 0., 0., 0., 0.], dtype=torch.float64),
            torch.tensor([10., y, 0., 0., 0., 0., 0., 0.], dtype=torch.float64),
        ]
        with patch.object(torch.linalg, "svdvals", side_effect=singular_values):
            expected = self.reference["get_global_greedy_rank_dict_mpp"](
                copy.deepcopy(model), None, right, 52 / 144, "cpu")
        with patch.object(torch.linalg, "svdvals", side_effect=singular_values):
            actual = allocate_ranks(model, None, right, 52 / 144, "cpu", selector="legacy")
        self.assertEqual(actual, expected)
        if boundary is not None:
            old_ranks = ({"first": 2, "second": 1} if boundary[2]
                         else {"first": 1, "second": 2})
            self.assertNotEqual(actual, old_ranks)

    def test_shared_statistics_legacy_ranks_and_sequential_compression(self):
        for trace_balance in (False, True):
            with self.subTest(trace_balance=trace_balance):
                model, dataset, options, left = mpp_case()
                right = self.reference["profile_input_gram"](
                    copy.deepcopy(model), dataset, options["params"], "cpu",
                    batches_per_dset=8, calibration_seed=42, use_trace_balance=trace_balance)
                for ratio in (0.2, 0.4, 0.8, 1.0):
                    expected_ranks = self.reference["get_global_greedy_rank_dict_mpp"](
                        copy.deepcopy(model), left, right, ratio, "cpu")
                    actual_ranks = allocate_ranks(copy.deepcopy(model), left, right, ratio, "cpu",
                                                  selector="legacy")
                    self.assertEqual(actual_ranks, expected_ranks)
                ranks = {"first": 1, "middle": 2, "last": 1}
                expected = self.reference["mysvd_compress_mpp"](
                    copy.deepcopy(model), dataset, options["params"], "cpu", ranks,
                    Ldict=left, alpha=0.7, batches_per_dset=8, calibration_seed=42,
                    use_trace_balance=trace_balance)
                actual = compress(copy.deepcopy(model), dataset, ranks, model_type="mpp", device="cpu",
                                  route_options=options, left_metrics=left, alpha=0.7,
                                  trace_balance=trace_balance, progress=False)
                self.assert_equivalent(actual, expected, dataset)

    def test_fresh_and_interrupted_resume_match_reference(self):
        class Interrupted(RuntimeError):
            pass

        model, dataset, options, left = mpp_case()
        ranks = {"first": 1, "middle": 2, "last": 1}
        kwargs = dict(Ldict=left, alpha=0.7, batches_per_dset=8,
                      calibration_seed=42, use_trace_balance=True)
        expected = self.reference["mysvd_compress_mpp"](
            copy.deepcopy(model), dataset, options["params"], "cpu", ranks, **kwargs)
        real_save = torch.save
        with tempfile.TemporaryDirectory() as directory:
            def save_then_interrupt(value, path):
                real_save(value, path)
                raise Interrupted("after the first saved layer")

            with patch.object(torch, "save", side_effect=save_then_interrupt):
                with self.assertRaises(Interrupted):
                    self.reference["mysvd_compress_mpp"](
                        copy.deepcopy(model), dataset, options["params"], "cpu", ranks,
                        save_path=directory, **kwargs)
            checkpoint = torch.load(Path(directory) / "compressing_model.pt", weights_only=False)
            reference_resumed = self.reference["mysvd_compress_mpp"](
                copy.deepcopy(model), dataset, options["params"], "cpu", ranks,
                save_path=directory, **kwargs)
            actual = compress(copy.deepcopy(model), dataset, ranks, model_type="mpp", device="cpu",
                              route_options=options, left_metrics=left, alpha=0.7, trace_balance=True,
                              initial_student=checkpoint["model"],
                              completed_layers=checkpoint["compressed_layers"], progress=False)
            self.assert_equivalent(reference_resumed, expected, dataset)
            self.assert_equivalent(actual, expected, dataset)


if __name__ == "__main__":
    unittest.main()
