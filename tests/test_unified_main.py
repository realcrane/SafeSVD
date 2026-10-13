"""End-to-end checks for safe_svd.main's unified Python entry point."""

from __future__ import annotations

import copy
import functools
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None

from test_unified_profiling import make_case


if torch is not None:
    class SaveableImageModel(nn.Module):
        """Top-level class so full-model saving can be checked like real models."""
        def __init__(self, model_type):
            super().__init__()
            self.model_type = model_type
            self.config = SimpleNamespace(name_or_path="toy")
            self.first, self.act, self.second = nn.Linear(4, 8), nn.ReLU(inplace=True), nn.Linear(8, 4)

        def forward(self, pixel_values, labels=None, bcs=None):
            values = pixel_values[-1] if self.model_type == "mpp" else pixel_values
            prediction = self.second(self.act(self.first(values.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
            if self.model_type == "poseidon":
                return SimpleNamespace(output=prediction, loss=(prediction - labels).square().mean())
            return prediction


@unittest.skipIf(torch is None, "PyTorch is required")
class UnifiedMainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        self.entry = importlib.import_module("safe_svd.main")

    def arguments(self, options):
        return dict(
            params=options.get("params"),
            trainer=options.get("trainer"),
            rollout_fn=options.get("rollout_fn"),
            dataset_name=options.get("dataset_name"),
            calibration_seed=options.get("calibration_seed", 42),
            batches_per_dset=options.get("batches_per_dset", 16),
            max_calib_batches=options.get("max_calib_batches"),
            train_seed=options.get("train_seed"),
            fisher_batch_size=options.get("fisher_batch_size", 16),
            capture_batch_size=options.get("capture_batch_size", 64),
            balance_ratio=options.get("balance_ratio", 10.0),
            order_control=options.get("order_control", 0),
            offload_cache=options.get("offload_cache", False),
        )

    def test_three_models_return_compressed_modules(self):
        for route in ("mpp", "vicon", "poseidon"):
            with self.subTest(route=route):
                model, data, options, args = make_case(route)
                before = copy.deepcopy(model.state_dict())
                with patch.object(self.entry, "allocate_ranks", wraps=self.entry.allocate_ranks) as select:
                    student = self.entry.main(model, data, model_type=route, rank_ratio=0.5,
                                              device="cpu", verbose=False, **self.arguments(options))
                output = student(*args)
                output = output.output if route == "poseidon" else output
                self.assertIsInstance(student, nn.Module)
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(select.call_args.kwargs["selector"], "safe")
                self.assertLess(sum(p.numel() for p in student.parameters()),
                                sum(p.numel() for p in model.parameters()))
                for name, values in model.state_dict().items():
                    torch.testing.assert_close(values, before[name])
                if route != "mpp":
                    self.assertIs(options["trainer"].model, model)

    def test_stage_order_and_explicit_parameters(self):
        model, data, options, _args = make_case("poseidon")
        calls, patches = [], []
        for name in ("collect_inputs", "collect_fisher", "allocate_ranks", "compress"):
            original = getattr(self.entry, name)
            def record(*args, _function=original, _name=name, **kwargs):
                calls.append((_name, kwargs.copy()))
                return _function(*args, **kwargs)
            patches.append(patch.object(self.entry, name, functools.wraps(original)(record)))
        for active in patches:
            active.start()
        try:
            self.entry.main(model, data, model_type="poseidon", rank_ratio=0.5, device="cpu",
                whitening_mode="new", eps_whiten=1e-5, eps_whiten_mode="relative", use_trace_balance="n",
                input_accumulation_dtype=torch.float64, verbose=False, **self.arguments(options))
        finally:
            for active in reversed(patches):
                active.stop()
        self.assertEqual([name for name, _ in calls], ["collect_inputs", "collect_fisher", "allocate_ranks", "compress"])
        for _name, kwargs in (calls[0], calls[-1]):
            self.assertEqual(kwargs["whitening"], "spectral")
            self.assertEqual(kwargs["eps"], 1e-5)
            self.assertEqual(kwargs["eps_mode"], "relative")
            self.assertFalse(kwargs["trace_balance"])
        self.assertEqual(calls[0][1]["accumulation_dtype"], torch.float64)

    def test_no_fisher_and_clean_input(self):
        model, data, options, _ = make_case("mpp")
        with patch.object(self.entry, "collect_fisher", side_effect=AssertionError("Fisher was disabled")):
            with patch.object(self.entry, "compress", wraps=self.entry.compress) as compress:
                student = self.entry.main(model, data, model_type="mpp", rank_ratio=0.5, device="cpu",
                    use_fisher="n", compress_mode="clean_input", alpha=0, verbose=False, **self.arguments(options))
        self.assertIsNone(compress.call_args.kwargs["left_metrics"])
        self.assertEqual(compress.call_args.kwargs["mode"], "clean")
        self.assertIsInstance(student.first, nn.Sequential)

    def test_poseidon_fisher_loader_is_preserved(self):
        model, data, options, args = make_case("poseidon")
        loader = torch.utils.data.DataLoader(data, batch_size=3, shuffle=False)
        with patch.object(self.entry, "collect_fisher", wraps=self.entry.collect_fisher) as fisher:
            student = self.entry.main(model, loader, model_type="poseidon", rank_ratio=0.5,
                                      device="cpu", verbose=False, **self.arguments(options))
        self.assertIs(fisher.call_args.args[1], data)
        self.assertIs(fisher.call_args.kwargs["route_options"]["fisher_loader"], loader)
        self.assertTrue(torch.isfinite(student(*args).output).all())

    def test_explicit_stage_caches_can_be_reused(self):
        model, data, options, args = make_case("mpp")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = dict(profile_cache=directory / "profiling_mat.pt", fisher_cache=directory / "Ldict.pt",
                         rank_cache=directory / "rank_dict.json")
            first = self.entry.main(model, data, model_type="mpp", rank_ratio=0.5, device="cpu",
                                    verbose=False, **self.arguments(options), **paths)
            for filename in ("profiling_mat.pt", "Ldict.pt", "rank_dict.json"):
                self.assertTrue((directory / filename).is_file())
            with patch.object(self.entry, "collect_inputs", side_effect=AssertionError("Use cached input")), \
                 patch.object(self.entry, "collect_fisher", side_effect=AssertionError("Use cached Fisher")), \
                 patch.object(self.entry, "allocate_ranks", side_effect=AssertionError("Use cached ranks")):
                second = self.entry.main(model, data, model_type="mpp", rank_ratio=0.5, device="cpu",
                                         verbose=False, **self.arguments(options), **paths)
            torch.testing.assert_close(second(*args), first(*args))

    def test_vicon_checkpoint_uses_original_state_dict_format(self):
        model, data, options, args = make_case("vicon")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            student = self.entry.main(model, data, model_type="vicon", rank_ratio=0.5, device="cpu",
                                      save_path=directory, verbose=False, **self.arguments(options))
            state = torch.load(directory / "compressed_model.pth", weights_only=True)
            ranks = json.loads((directory / "rank_dict.json").read_text())
            self.assertNotIn("state_dict", state)
            rebuilt = copy.deepcopy(model)
            for name, rank in ranks.items():
                if rank is None:
                    continue
                original = rebuilt.get_submodule(name)
                replacement = nn.Sequential(nn.Linear(original.in_features, rank, bias=False),
                                            nn.Linear(rank, original.out_features, bias=original.bias is not None))
                parent, _, leaf = name.rpartition(".")
                setattr(rebuilt.get_submodule(parent) if parent else rebuilt, leaf, replacement)
            rebuilt.load_state_dict(state)
            torch.testing.assert_close(rebuilt(*args), student(*args))
            with patch.object(self.entry, "collect_inputs", wraps=self.entry.collect_inputs) as collect:
                with self.assertRaises(FileExistsError):
                    self.entry.main(model, data, model_type="vicon", rank_ratio=0.5, device="cpu",
                                     save_path=directory, verbose=False, **self.arguments(options))
                collect.assert_not_called()

    def test_mpp_and_poseidon_save_complete_models(self):
        for route in ("mpp", "poseidon"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as temporary:
                _model, data, options, args = make_case(route)
                model = SaveableImageModel(route).eval()
                if route == "poseidon":
                    options["trainer"].model = model
                directory = Path(temporary)
                student = self.entry.main(model, data, model_type=route, rank_ratio=0.5, device="cpu",
                                          save_path=directory, verbose=False, **self.arguments(options))
                loaded = torch.load(directory / "compressed_model.pt", map_location="cpu", weights_only=False)
                self.assertIsInstance(loaded, nn.Module)
                expected, actual = student(*args), loaded(*args)
                if route == "poseidon":
                    expected, actual = expected.output, actual.output
                torch.testing.assert_close(actual, expected)

    def test_invalid_configuration_fails_before_collection(self):
        model, data, options, _ = make_case("mpp")
        for override in ({"whitening_mode": "invalid"}, {"compress_mode": "invalid"}):
            with patch.object(self.entry, "collect_inputs", wraps=self.entry.collect_inputs) as collect:
                with self.assertRaises(ValueError):
                    self.entry.main(model, data, model_type="mpp", rank_ratio=0.5, device="cpu",
                                     verbose=False, **self.arguments(options), **override)
                collect.assert_not_called()

    def test_native_vicon_attention_is_prepared_and_pipeline_runs(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
                self.head = nn.Linear(8, 4)
            def forward(self, x):
                return self.head(self.attn(x, x, x, need_weights=False)[0])
        model = Model().eval()
        trainer = SimpleNamespace(model=model, _move_to_device=lambda x: x,
                                  _data_preprocess=lambda x: (x, None, None))
        trainer._model_forward = lambda x: trainer.model(x)
        values = torch.randn(2, 4, 8)
        with torch.no_grad():
            before = model(values)
        def loss(network, tagged):
            return network(tagged.payload[1]).square().mean()
        student = self.entry.main(model, {"toy": [(0, values, None, None, None)]}, model_type="vicon",
                                  rank_ratio=0.5, device="cpu", trainer=trainer, fisher_loss_fn=loss, verbose=False)
        with torch.no_grad():
            torch.testing.assert_close(model(values), before, atol=1e-6, rtol=1e-5)
            self.assertTrue(torch.isfinite(student(values)).all())
        self.assertTrue(all(not isinstance(module, nn.MultiheadAttention) for module in model.modules()))

    def test_attention_conversion_preserves_masks_and_float64(self):
        from safe_svd.helper.collector.vicon_attention import ExplicitMultiheadAttention
        original = nn.MultiheadAttention(8, 2, batch_first=True).double().eval()
        converted = ExplicitMultiheadAttention.from_native(original)
        values = torch.randn(2, 4, 8, dtype=torch.float64)
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[:, -1] = True
        padding = torch.tensor([[False, True, False, False], [False, False, True, False]])
        with torch.no_grad():
            expected = original(values, values, values, attn_mask=mask, key_padding_mask=padding, need_weights=False)[0]
            actual = converted(values, values, values, attn_mask=mask, key_padding_mask=padding)[0]
        self.assertEqual(converted.q_proj.weight.dtype, torch.float64)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
