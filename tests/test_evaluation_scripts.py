"""Exercise standalone evaluation commands and their shared runtime routes."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def entry(model):
    spec = importlib.util.spec_from_file_location(f"{model}_evaluation_script", ROOT / f"evaluation/{model}/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MppFile(torch.utils.data.Dataset):
    title = "small-ns"

    def __len__(self):
        return 4

    def get_name(self):
        return "incompNS"

    def __getitem__(self, index):
        return torch.ones(2, 2, 4, 4), torch.zeros(1), torch.full((2, 4, 4), 2.)


class MppModel(torch.nn.Module):
    def forward(self, history, labels, boundary):
        return history[-1] * .5


class EvaluationScriptTests(unittest.TestCase):
    def test_direct_script_help_works_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            for model in ("vicon", "mpp"):
                result = subprocess.run([sys.executable, "-B", str(ROOT / f"evaluation/{model}/evaluate.py"), "--help"],
                                        cwd=directory, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                for option in ("--checkpoint", "--model", "--data-root", "--output", "--device"):
                    self.assertIn(option, result.stdout)

    def test_default_and_custom_options_reach_runtime(self):
        required = ["--checkpoint", "/weights/original", "--data-root", "/data", "--device", "cpu"]
        for model, batch_size, max_batches in (("vicon", 128, None), ("mpp", 1, 64)):
            module = entry(model)
            with patch(f"safesvd.adapters.{model}_runtime.evaluate_{model}") as evaluate:
                self.assertEqual(module.main(required), 0)
                options = evaluate.call_args.args[0]
                self.assertEqual((options.batch_size, options.max_batches), (batch_size, max_batches))
                self.assertIsNone(options.compressed_path)
                self.assertEqual(options.output_path, Path(f"/weights/evaluation/{model}/metrics.json"))
                if model == "vicon":
                    self.assertEqual(options.protocol, "research")
                module.main(required + ["--model", "/run/compressed/model.pt", "--batch-size", "2", "--max-batches", "3"])
                options = evaluate.call_args.args[0]
                self.assertEqual((options.batch_size, options.max_batches), (2, 3))
                self.assertEqual(options.output_path, Path("/run/metrics.json"))

    def test_model_specific_configuration_is_forwarded(self):
        required = ["--checkpoint", "/weights/original", "--data-root", "/data", "--output", "/out/metrics.json"]
        with patch("safesvd.adapters.vicon_runtime.evaluate_vicon") as evaluate:
            entry("vicon").main(required + ["--vicon-root", "/vicon", "--config-dir", "configs",
                                           "--config-override", "test_seed=42", "--protocol", "paired"])
            options = evaluate.call_args.args[0]
            self.assertEqual(options.config_dir, Path("/vicon/configs"))
            self.assertEqual(options.config_overrides, ("test_seed=42",))
            self.assertEqual(options.protocol, "paired")
            self.assertEqual(options.output_path, Path("/out/metrics.json"))
        with patch("safesvd.adapters.mpp_runtime.evaluate_mpp") as evaluate:
            entry("mpp").main(required + ["--mpp-root", "/mpp", "--yaml-config", "config/model.yaml", "--config-name", "custom"])
            options = evaluate.call_args.args[0]
            self.assertEqual(options.yaml_config, Path("/mpp/config/model.yaml"))
            self.assertEqual(options.config_name, "custom")

    def test_vicon_entry_evaluates_four_passes_and_writes_json(self):
        from test_evaluation_alignment import TimePairDataset, TinyTrainer
        from safesvd.adapters.vicon_runtime import ViconBundle

        events = []
        datasets = {name: TimePairDataset(name, events) for name in ("NS2D", "EULER2D")}
        bundle = ViconBundle(ROOT / "VICON", SimpleNamespace(datasets={}, dataset_workers=0, test_seed=84),
                             TinyTrainer(), lambda *_: datasets, ROOT / "data", torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vicon.json"
            with patch("safesvd.adapters.vicon_runtime.load_vicon_bundle", return_value=bundle), \
                 patch("safesvd.adapters.vicon_runtime.load_compressed_state", side_effect=lambda model, _: model):
                entry("vicon").main(["--checkpoint", "/weights/original", "--model", "/weights/compressed",
                                     "--data-root", "/data", "--device", "cpu", "--batch-size", "1", "--output", str(output)])
            payload = json.loads(output.read_text())
        self.assertEqual(payload["protocol"], "research")
        self.assertEqual(events, [(name, step) for step in range(1, 5) for name in datasets])
        self.assertEqual(payload["compressed"]["metrics"]["losses"]["NS2D"], 1.)

    def test_mpp_entry_evaluates_both_models_and_writes_json(self):
        from safesvd.adapters.mpp_runtime import MPPBundle

        dataset = SimpleNamespace(sub_dsets=[SimpleNamespace(get_per_file_dsets=lambda: [MppFile()])],
                                  subset_dict={"incompNS": [0, 1]})
        bundle = MPPBundle(ROOT, SimpleNamespace(batch_size=1, num_data_workers=0), MppModel(),
                           dataset, dataset, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mpp.json"
            with patch("safesvd.adapters.mpp_runtime.load_mpp_bundle", return_value=bundle), \
                 patch("safesvd.adapters.mpp_runtime.load_compressed_state", side_effect=lambda model, _: model):
                entry("mpp").main(["--checkpoint", "/weights/original", "--model", "/weights/compressed",
                                   "--data-root", "/data", "--device", "cpu", "--max-batches", "2", "--output", str(output)])
            payload = json.loads(output.read_text())
        for section in ("origin", "compressed"):
            metrics = payload[section]["metrics"]["datasets"]["small-ns"]
            self.assertEqual(metrics["mse"], 2.25)
            self.assertEqual(metrics["sobolev"], 1.5)
