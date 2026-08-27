from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for runtime adapter tests")
class SharedRuntimeTest(unittest.TestCase):
    def test_category_statistics_callback_uses_shared_sequential_compressor(self) -> None:
        from safesvd.adapters.common_runtime import get_module_by_name, set_module_by_name
        from safesvd.core import CompressionCallbacks, compress_sequentially

        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))

        def statistics(_teacher, _student, name):
            layer = get_module_by_name(model, name)
            return {
                "hh": torch.eye(layer.in_features),
                "hph": torch.eye(layer.in_features),
                "hphp": torch.eye(layer.in_features),
            }

        callbacks = CompressionCallbacks(
            capture_inputs=lambda _model, _name: (),
            get_module=get_module_by_name,
            set_module=set_module_by_name,
            capture_statistics=statistics,
        )
        compressed = compress_sequentially(
            model,
            {"0": 1, "2": 1},
            callbacks,
            device="cpu",
            alpha=0.7,
        )
        self.assertIsInstance(compressed[0], torch.nn.Sequential)
        self.assertIsInstance(compressed[2], torch.nn.Sequential)

    def test_portable_checkpoint_rebuilds_low_rank_linear_layers(self) -> None:
        from safesvd.adapters.common_runtime import load_compressed_state, save_portable_checkpoint

        model = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False), torch.nn.Linear(2, 3)))
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.pt"
            save_portable_checkpoint(path, model, model_name="toy", ranks={"0": 2})
            restored = torch.nn.Sequential(torch.nn.Linear(4, 3))
            load_compressed_state(restored, path)
            self.assertIsInstance(restored[0], torch.nn.Sequential)
            self.assertTrue(torch.allclose(model(torch.ones(1, 4)), restored(torch.ones(1, 4))))

    def test_cached_batches_replay_a_single_fixed_calibration_set(self) -> None:
        from safesvd.adapters.common_runtime import TaggedBatch, cache_tagged_batches

        calls = 0

        def source():
            nonlocal calls
            calls += 1
            return (TaggedBatch("toy", torch.tensor([float(calls)])),)

        replay = cache_tagged_batches(source)
        first = tuple(replay())
        second = tuple(replay())
        self.assertEqual(calls, 1)
        self.assertTrue(torch.equal(first[0].payload, second[0].payload))

    def test_optional_layers_get_identity_profile_and_no_fisher_metric(self) -> None:
        from safesvd.adapters.common_runtime import TaggedBatch, collect_fisher_whiteners, collect_input_whiteners

        class OptionalBranch(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.used = torch.nn.Linear(2, 2)
                self.unused = torch.nn.Linear(2, 2)

            def forward(self, value):
                return self.used(value)

        model = OptionalBranch()
        batches = lambda: (TaggedBatch("toy", torch.ones(2, 2)),)
        profile = collect_input_whiteners(
            model, batches, lambda candidate, payload: candidate(payload), device="cpu", trace_balance=True,
        )
        self.assertEqual(set(profile), {"used", "unused"})
        fisher = collect_fisher_whiteners(
            model,
            batches,
            lambda candidate, tagged: candidate(tagged.payload).square().mean(),
            device="cpu",
            trace_balance=True,
        )
        self.assertIn("used", fisher)
        self.assertNotIn("unused", fisher)

    def test_category_aware_statistics_streams_and_trace_balances_batches(self) -> None:
        from safesvd.adapters.common_runtime import CategoryAwareStatistics, TaggedBatch

        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(2))
        values = (
            TaggedBatch("small", torch.tensor([[1.0, 0.0]])),
            TaggedBatch("large", torch.tensor([[0.0, 2.0]])),
        )
        statistics = CategoryAwareStatistics(
            lambda: values,
            lambda candidate, payload: candidate(payload),
            device="cpu",
            trace_balance=True,
        )(model, model, "0")
        # Each category is first normalised by its own trace; the two diagonal
        # contributions therefore receive equal energy before being averaged.
        self.assertEqual(set(statistics), {"hh", "hph", "hphp"})
        self.assertTrue(torch.allclose(statistics["hh"], torch.diag(torch.tensor([2.0, 2.0], dtype=torch.float64))))

    def test_category_statistics_without_trace_balance_is_category_weighted(self) -> None:
        from safesvd.adapters.common_runtime import CategoryAwareStatistics, TaggedBatch

        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(2))
        values = (
            TaggedBatch("small", torch.tensor([[1.0, 0.0]])),
            TaggedBatch("large", torch.tensor([[0.0, 2.0], [0.0, 2.0]])),
        )
        statistics = CategoryAwareStatistics(
            lambda: values,
            lambda candidate, payload: candidate(payload),
            device="cpu",
            trace_balance=False,
        )(model, model, "0")
        # Paper collectors average within each physical category before the
        # final merge, even when trace alignment is disabled.
        self.assertTrue(torch.allclose(
            statistics["hh"], torch.diag(torch.tensor([0.5, 2.0], dtype=torch.float64)),
        ))

    def test_trace_balance_uses_per_category_mean_teacher_trace(self) -> None:
        from safesvd.adapters.common_runtime import CategoryAwareStatistics, TaggedBatch

        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(2))
        values = (
            TaggedBatch("small", torch.tensor([[1.0, 0.0]])),
            TaggedBatch("large", torch.tensor([[0.0, 2.0], [0.0, 2.0]])),
        )
        statistics = CategoryAwareStatistics(
            lambda: values,
            lambda candidate, payload: candidate(payload),
            device="cpu",
            trace_balance=True,
        )(model, model, "0")
        # Category means are diag(1, 0) and diag(0, 4); trace balancing to
        # four therefore gives diag(2, 2), irrespective of sample counts.
        self.assertTrue(torch.allclose(
            statistics["hh"], torch.diag(torch.tensor([2.0, 2.0], dtype=torch.float64)),
        ))

    def test_category_statistics_supports_mpp_global_no_trace_mode(self) -> None:
        from safesvd.adapters.common_runtime import CategoryAwareStatistics, TaggedBatch

        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(2))
        values = (
            TaggedBatch("small", torch.tensor([[1.0, 0.0]])),
            TaggedBatch("large", torch.tensor([[0.0, 2.0], [0.0, 2.0]])),
        )
        statistics = CategoryAwareStatistics(
            lambda: values,
            lambda candidate, payload: candidate(payload),
            device="cpu",
            trace_balance=False,
            category_average_without_trace=False,
        )(model, model, "0")
        self.assertTrue(torch.allclose(
            statistics["hh"], torch.diag(torch.tensor([1 / 3, 8 / 3], dtype=torch.float64)),
        ))

    def test_explicit_attention_matches_native_attention_for_boolean_masks(self) -> None:
        from safesvd.adapters.vicon_attention import ExplicitMultiheadAttention

        torch.manual_seed(3)
        for batch_first in (False, True):
            native = torch.nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=batch_first).eval()
            explicit = ExplicitMultiheadAttention.from_native(native).eval()
            value = torch.randn((2, 3, 8) if batch_first else (3, 2, 8))
            attention_mask = torch.tensor([
                [False, True, False],
                [False, False, False],
                [False, False, False],
            ])
            padding_mask = torch.tensor([[False, False, True], [False, True, False]])
            expected, _ = native(
                value, value, value,
                attn_mask=attention_mask,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            actual, _ = explicit(
                value, value, value,
                attn_mask=attention_mask,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-5))
            additive_padding = torch.tensor([[0.0, float("-inf"), 0.0], [0.0, 0.0, float("-inf")]])
            expected, _ = native(value, value, value, key_padding_mask=additive_padding, need_weights=False)
            actual, _ = explicit(value, value, value, key_padding_mask=additive_padding, need_weights=False)
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-5))

    def test_vicon_fixed_prefix_uses_train_seed_and_source_worker_contract(self) -> None:
        from safesvd.adapters.vicon_runtime import ViconBundle, _vicon_batch_factory

        calls = []

        def all_datasets(_config, workers, seed, split):
            calls.append((workers, seed, split))
            return {"toy": torch.utils.data.TensorDataset(torch.arange(8.0).view(4, 2))}

        bundle = ViconBundle(
            Path("."),
            SimpleNamespace(datasets={}, dataset_workers=0, train_seed=17),
            None,
            all_datasets,
            Path("."),
            torch.device("cpu"),
        )
        batches = _vicon_batch_factory(
            bundle,
            split="train",
            batch_size=2,
            calibration_size=4,
            seed=42,
            random_subset=False,
            source_dataset_workers=0,
        )
        self.assertEqual(len(tuple(batches())), 2)
        self.assertEqual(calls, [(0, 17, "train")])
