"""Persistent compression checkpoints: interruption, atomic writes and restart."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import test_unified_main as main_tests
from test_unified_profiling import make_case


class AttentionProjections(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query, self.key, self.value = (torch.nn.Linear(4, 4) for _ in range(3))

    def forward(self, values):
        return self.query(values) + self.key(values) + self.value(values)


class SaveableAttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(name_or_path="toy_attention")
        self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.encoder.layers[0].blocks = torch.nn.ModuleList([torch.nn.Module()])
        attention = torch.nn.Module()
        setattr(attention, "self", AttentionProjections())
        attention.output = torch.nn.Module()
        attention.output.dense = torch.nn.Linear(4, 4)
        self.encoder.layers[0].blocks[0].attention = attention

    def forward(self, pixel_values, labels):
        attention = self.encoder.layers[0].blocks[0].attention
        values = getattr(attention, "self")(pixel_values.permute(0, 2, 3, 1))
        output = attention.output.dense(values).permute(0, 3, 1, 2)
        return SimpleNamespace(output=output, loss=(output - labels).square().mean())


class CompressionCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(102)
        self.entry = importlib.import_module("safe_svd.main")
        self.compression = importlib.import_module("safe_svd.compress")

    def case(self, route, directory):
        model, data, options, args = make_case(route)
        if route != "vicon":
            model = main_tests.SaveableImageModel(route).eval()
            if route == "poseidon":
                options["trainer"].model = model
        rank_path = directory / "ranks.json"
        rank_path.write_text(json.dumps({"first": 1, "second": 1}))
        settings = dict(model_type=route, rank_ratio=0.5, device="cpu", verbose=False,
                        profile_cache=directory / "inputs.pt", fisher_cache=directory / "fisher.pt",
                        rank_cache=rank_path, save_path=directory / "out",
                        **main_tests.UnifiedMainTests.arguments(None, options))
        return model, data, options, args, settings

    def interrupt_on_second_layer(self, visited):
        route_collectors = {"mpp": self.compression.collect_mpp, "vicon": self.compression.collect_vicon,
                            "poseidon": self.compression.collect_poseidon}
        def interrupted(model, data, **kwargs):
            collect = route_collectors[kwargs["model_type"]]
            def capture(context, name):
                visited.append(name)
                if name == "second":
                    raise RuntimeError("simulated interruption")
                yield from collect(context, name)
            return self.compression.compress(model, data, **kwargs, capture_fn=capture)
        return interrupted

    def test_three_models_resume_and_match_uninterrupted_compression(self):
        for route in ("mpp", "vicon", "poseidon"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                model, data, options, args, settings = self.case(route, directory)
                visited = []
                with patch.object(self.entry, "compress", side_effect=self.interrupt_on_second_layer(visited)):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        self.entry.main(model, data, **settings)
                checkpoint_path = settings["save_path"] / "compressing_model.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["compressed_layers"], ["first"])
                self.assertIn("model_state_dict" if route == "vicon" else "model", checkpoint)
                if route != "vicon":
                    self.assertTrue(all(p.device.type == "cpu" for p in checkpoint["model"].parameters()))
                self.assertEqual(visited, ["first", "second"])
                resumed_layers = []
                collect = getattr(self.compression, f"collect_{route}")
                def recording_capture(context, name):
                    resumed_layers.append(name)
                    yield from collect(context, name)
                def resumed(model, data, **kwargs):
                    self.assertEqual(kwargs["completed_layers"], {"first"})
                    self.assertIsNotNone(kwargs["initial_student"])
                    return self.compression.compress(model, data, **kwargs, capture_fn=recording_capture)
                with patch.object(self.entry, "compress", side_effect=resumed), \
                     patch.object(self.entry, "collect_inputs", side_effect=AssertionError("Reuse input cache")), \
                     patch.object(self.entry, "collect_fisher", side_effect=AssertionError("Reuse Fisher cache")), \
                     patch.object(self.entry, "allocate_ranks", side_effect=AssertionError("Reuse ranks")):
                    student = self.entry.main(model, data, **settings)
                self.assertEqual(resumed_layers, ["second"])
                self.assertFalse(checkpoint_path.exists())
                fresh = self.entry.main(model, data, **{**settings, "save_path": None})
                for name, value in fresh.state_dict().items():
                    torch.testing.assert_close(student.state_dict()[name], value)
                expected, actual = fresh(*args), student(*args)
                if route == "poseidon":
                    expected, actual = expected.output, actual.output
                torch.testing.assert_close(actual, expected)

    def test_failed_checkpoint_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, data, options, args, settings = self.case("poseidon", Path(temporary))
            real_save, writes = torch.save, []
            def failing_save(value, path, *args, **kwargs):
                if Path(path).name == "compressing_model.pt.tmp":
                    writes.append(path)
                    if len(writes) == 2:
                        Path(path).write_bytes(b"incomplete checkpoint")
                        raise OSError("simulated disk error")
                return real_save(value, path, *args, **kwargs)
            with patch.object(self.entry.torch, "save", side_effect=failing_save):
                with self.assertRaisesRegex(OSError, "simulated disk error"):
                    self.entry.main(model, data, **settings)
            checkpoint_path = settings["save_path"] / "compressing_model.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.assertEqual(checkpoint["compressed_layers"], ["first"])
            self.assertFalse(checkpoint_path.with_suffix(".pt.tmp").exists())
            student = self.entry.main(model, data, **settings)
            self.assertTrue(torch.isfinite(student(*args).output).all())

    def test_poseidon_resume_after_query_rebuilds_missing_attention_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            _model, data, options, args, settings = self.case("poseidon", Path(temporary))
            model = SaveableAttentionModel().eval()
            options["trainer"].model = model
            names = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
            settings["rank_cache"].write_text(json.dumps(dict.fromkeys(names, 1)))
            def interrupted(model, data, **kwargs):
                def capture(context, name):
                    if name.endswith(".key"):
                        raise RuntimeError("interrupted after query")
                    yield from self.compression.collect_poseidon(context, name)
                return self.compression.compress(model, data, **kwargs, capture_fn=capture)
            with patch.object(self.entry, "compress", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "interrupted after query"):
                    self.entry.main(model, data, **settings)
            resumed = self.entry.main(model, data, **settings)
            fresh = self.entry.main(model, data, **{**settings, "save_path": None})
            for name, value in fresh.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[name], value)
            torch.testing.assert_close(resumed(*args).output, fresh(*args).output)

    def test_final_write_failure_can_resume_fully_processed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, data, options, args, settings = self.case("poseidon", Path(temporary))
            real_save = torch.save
            def failing_final(value, path, *args, **kwargs):
                if Path(path).name == "compressed_model.pt.tmp":
                    Path(path).write_bytes(b"incomplete final model")
                    raise OSError("simulated final write failure")
                return real_save(value, path, *args, **kwargs)
            with patch.object(self.entry.torch, "save", side_effect=failing_final):
                with self.assertRaisesRegex(OSError, "simulated final write failure"):
                    self.entry.main(model, data, **settings)
            output = settings["save_path"]
            checkpoint = torch.load(output / "compressing_model.pt", weights_only=False)
            self.assertEqual(checkpoint["compressed_layers"], ["first", "second"])
            self.assertFalse((output / "compressed_model.pt").exists())
            with patch.object(self.compression, "collect_poseidon", side_effect=AssertionError("All layers completed")):
                self.entry.main(model, data, **settings)
            self.assertTrue((output / "compressed_model.pt").is_file())
            self.assertFalse((output / "compressing_model.pt").exists())

    def test_changed_ranks_or_settings_are_rejected_and_resume_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, data, options, args, settings = self.case("poseidon", Path(temporary))
            with patch.object(self.entry, "compress", side_effect=self.interrupt_on_second_layer([])):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    self.entry.main(model, data, **settings)
            with self.assertRaisesRegex(ValueError, "settings differ.*alpha"):
                self.entry.main(model, data, **{**settings, "alpha": 0.4})
            settings["rank_cache"].write_text(json.dumps({"first": 2, "second": 1}))
            with self.assertRaisesRegex(ValueError, "ranks differ"):
                self.entry.main(model, data, **settings)
            with patch.object(self.entry, "compress", wraps=self.compression.compress) as compress:
                self.entry.main(model, data, **settings, overwrite_cache=True)
            self.assertIsNone(compress.call_args.kwargs["initial_student"])
            self.assertEqual(compress.call_args.kwargs["completed_layers"], set())

    def test_cli_checkpoint_settings_and_validation(self):
        args = self.entry.parse_args(["--model", "poseidon"])
        self.assertEqual(args.checkpoint_percent, 0.1)
        self.assertFalse(args.overwrite_cache)
        self.assertFalse(hasattr(args, "no_resume"))
        args = self.entry.parse_args(["--model", "mpp", "--checkpoint_percent", "0.05", "--overwrite_cache"])
        self.assertEqual(args.checkpoint_percent, 0.05)
        self.assertTrue(args.overwrite_cache)
        for invalid in ("0", "-1", "1.1", "nan"):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                self.entry.parse_args(["--model", "poseidon", "--checkpoint_percent", invalid])
