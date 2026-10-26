"""MPP stage isolation, configuration provenance and incompatible-artifact guards."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


class MPPPipelineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.entry = importlib.import_module("safe_svd.main")

    def project(self, root):
        package = root / "multiple_physics_pretraining"
        (package / "models").mkdir(parents=True)
        (package / "models/avit.py").write_text("# Local MPP runtime fixture\n")
        config = root / "safe_svd/configs/mpp_avit_L_config.yaml"
        config.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "safe_svd/configs/mpp_avit_L_config.yaml", config)
        checkpoint = root / "weights.pt"
        checkpoint.write_bytes(b"checkpoint identity fixture")
        return ["--model", "mpp", "--ckpt_path", str(checkpoint),
                "--data_root", str(root / "data"), "--calibration_num", "16",
                "--calibration_seed", "42", "--batch_size", "1", "--rank_ratio", "0.8",
                "--rank_cache", str(root / "ranks.json"), "--save_path", str(root / "out")]

    def test_cli_dispatches_two_children_before_loading_parent_gpu_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            argv = self.project(root) + ["--DEV", "cuda", "--overwrite_cache"]
            children = []

            def child(command, *, env, check):
                self.assertEqual(command[1:4], ["-u", "-m", "safe_svd.main"])
                args = self.entry.parse_args(command[4:])
                children.append(args)
                self.assertEqual((args.calibration_num, args.calibration_seed, args.batch_size), (16, 42, 1))
                self.assertEqual(args.DEV, "cuda")
                self.assertIn(str(root), env["PYTHONPATH"].split(":"))
                self.assertFalse(check)
                if args.pipeline_stage == "rank_selection":
                    args.rank_cache.write_text('{"layer": 1}\n')
                return SimpleNamespace(returncode=0)

            with patch.object(self.entry, "PROJECT_ROOT", root), \
                 patch.object(self.entry, "load_model_and_data", side_effect=AssertionError("Parent must not allocate AViT")), \
                 patch.object(self.entry.subprocess, "run", side_effect=child), \
                 patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(self.entry.cli_main(argv), 0)
            self.assertEqual([args.pipeline_stage for args in children], ["rank_selection", "compression"])
            self.assertTrue(children[0].overwrite_cache)
            self.assertFalse(children[1].overwrite_cache)
            self.assertTrue(children[1]._overwrite_output)
            for field in ("profile_cache", "fisher_cache", "rank_cache", "save_path", "yaml_config"):
                self.assertEqual(getattr(children[0], field), getattr(children[1], field))
            self.assertFalse((root / "out").exists())

    def test_rank_failure_or_missing_rank_never_launches_compression(self):
        for returncode in (23, 0):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                argv = self.project(root)
                with patch.object(self.entry, "PROJECT_ROOT", root), \
                     patch.object(self.entry, "load_model_and_data", side_effect=AssertionError("Parent must not load model")), \
                     patch.object(self.entry.subprocess, "run", return_value=SimpleNamespace(returncode=returncode)) as run, \
                     patch("sys.stdout", new_callable=io.StringIO):
                    if returncode:
                        self.assertEqual(self.entry.cli_main(argv), returncode)
                    else:
                        with self.assertRaisesRegex(FileNotFoundError, "Rank selection did not produce"):
                            self.entry.cli_main(argv)
                self.assertEqual(run.call_count, 1)

    def test_default_config_preserves_reference_memory_and_input_settings(self):
        try:
            from ruamel.yaml import YAML
        except ImportError:
            self.skipTest("ruamel.yaml is needed to inspect the actual MPP configuration")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            argv = self.project(root)
            with patch.object(self.entry, "PROJECT_ROOT", root):
                args = self.entry.resolve_cli_paths(self.entry.parse_args(argv))
            config = YAML().load(args.yaml_config.read_text())[args.config]
            self.assertIs(config["gradient_checkpointing"], True)
            self.assertIs(config["enable_amp"], False)
            self.assertEqual((config["batch_size"], config["n_steps"], config["embedding_offset"]), (1, 16, 0))
            self.assertEqual(config["train_val_test"], [0.8, 0.1, 0.1])
            self.assertEqual(len(config["train_data_paths"]), 6)
            self.assertTrue(all(spec[0].startswith("2D/") for spec in config["train_data_paths"]))

    def test_runtime_or_yaml_changes_cannot_select_old_default_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            argv = self.project(root)
            with patch.object(self.entry, "PROJECT_ROOT", root):
                before = self.entry.resolve_cli_paths(self.entry.parse_args(argv))
                source = root / "multiple_physics_pretraining/models/avit.py"
                source.write_text(source.read_text() + "# Changed checkpoint implementation\n")
                changed_source = self.entry.resolve_cli_paths(self.entry.parse_args(argv))
                config = changed_source.yaml_config
                config.write_text(config.read_text().replace("gradient_checkpointing: !!bool True", "gradient_checkpointing: !!bool False"))
                changed_config = self.entry.resolve_cli_paths(self.entry.parse_args(argv))
            self.assertEqual(len({args.profile_cache for args in (before, changed_source, changed_config)}), 3)
            self.assertEqual(len({args.fisher_cache for args in (before, changed_source, changed_config)}), 3)
            self.assertIn("mpp_research_v3_", str(before.profile_cache))
            self.assertEqual(changed_config.mpp_run_provenance["config_sha256"], hashlib.sha256(config.read_bytes()).hexdigest())
            self.assertIn(("models/avit.py", hashlib.sha256(source.read_bytes()).hexdigest()),
                          changed_config.mpp_run_provenance["source_hashes"])

    def test_failed_child_records_effective_settings_and_passes_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            argv = self.project(root) + ["--DEV", "cpu", "--pipeline_stage", "rank_selection"]
            rebased = [[str(root / "data/2D/shallow-water"), "swe", ""]]
            params = SimpleNamespace(params={"gradient_checkpointing": True, "batch_size": 8,
                                            "n_steps": 16, "train_data_paths": [["2D/shallow-water", "swe", ""]]},
                                     gradient_checkpointing=True, batch_size=1, n_steps=16,
                                     train_data_paths=rebased)

            def load(args):
                args.train_seed, args.batches_per_dset, args.max_calib_batches = None, 16, None
                return torch.nn.Linear(2, 2), [], params, None, None

            with patch.object(self.entry, "PROJECT_ROOT", root), \
                 patch.object(self.entry, "load_model_and_data", side_effect=load), \
                 patch.object(self.entry, "main", side_effect=RuntimeError("Simulated Fisher failure")) as main, \
                 patch("sys.stdout", new_callable=io.StringIO):
                with self.assertRaisesRegex(RuntimeError, "Simulated Fisher failure"):
                    self.entry.cli_main(argv)
            report = json.loads((root / "out/rank_run_config.json").read_text())
            self.assertEqual(report["mpp_alignment_version"], 3)
            self.assertEqual(report["effective_mpp_config"]["batch_size"], 1)
            self.assertEqual(report["effective_mpp_config"]["train_data_paths"], rebased)
            self.assertIs(report["effective_mpp_config"]["gradient_checkpointing"], True)
            self.assertEqual(main.call_args.kwargs["run_provenance"]["data_root"], str(root / "data"))
            self.assertNotIn("completed_stage", report)

    def cached_case(self, root):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        profile, fisher, ranks = root / "input.pt", root / "fisher.pt", root / "ranks.json"
        factors = {"0": (torch.eye(2), torch.eye(2))}
        torch.save({"profile_mat": factors, "sample_counts": {"0": 1}, "mpp_alignment_version": 3}, profile)
        torch.save({"fisher_mat": factors, "mpp_alignment_version": 3}, fisher)
        ranks.write_text('{"0": 1}\n')
        return model, factors, dict(model_type="mpp", params=SimpleNamespace(batch_size=1, num_data_workers=0),
                                   device="cpu", rank_ratio=0.8, profile_cache=profile, fisher_cache=fisher,
                                   rank_cache=ranks, verbose=False)

    def test_old_structured_caches_fail_before_collection_or_compression(self):
        for field, payload_key in (("profile_cache", "profile_mat"), ("fisher_cache", "fisher_mat")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model, factors, settings = self.cached_case(root)
                torch.save({payload_key: factors, "sample_counts": {"0": 1}, "mpp_alignment_version": 2}, settings[field])
                with patch.object(self.entry, "collect_inputs", side_effect=AssertionError("Must reject old cache")), \
                     patch.object(self.entry, "collect_fisher", side_effect=AssertionError("Must reject old cache")), \
                     patch.object(self.entry, "compress", side_effect=AssertionError("Must reject old cache")):
                    with self.assertRaisesRegex(ValueError, "MPP statistics cache predates research alignment v3"):
                        self.entry.main(model, [], pipeline_stage="compression", **settings)

    def test_explicit_raw_reference_statistics_are_still_usable(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, factors, settings = self.cached_case(Path(temporary))
            torch.save(factors, settings["profile_cache"])
            torch.save(factors, settings["fisher_cache"])
            with patch.object(self.entry, "collect_inputs", side_effect=AssertionError("Must reuse reference cache")), \
                 patch.object(self.entry, "collect_fisher", side_effect=AssertionError("Must reuse reference cache")), \
                 patch.object(self.entry, "allocate_ranks", side_effect=AssertionError("Must reuse reference ranks")):
                self.assertIsNone(self.entry.main(model, [], pipeline_stage="rank_selection", **settings))

    def test_old_resume_checkpoint_is_rejected_without_overwriting_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, _factors, settings = self.cached_case(root)
            output = root / "out"
            output.mkdir()
            checkpoint = output / "compressing_model.pt"
            torch.save({"compressed_layers": ["0"], "compression_config": {"mpp_alignment_version": 2}}, checkpoint)
            before = checkpoint.read_bytes()
            with patch.object(self.entry, "compress", side_effect=AssertionError("Must reject old model before compression")):
                with self.assertRaisesRegex(ValueError, "MPP checkpoint predates research alignment v3"):
                    self.entry.main(model, [], pipeline_stage="compression", save_path=output, **settings)
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertFalse((output / "compressed_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
