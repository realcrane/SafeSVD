"""CLI loading tests with tiny model runtimes, plus real four-stage execution."""

from __future__ import annotations

import csv
import importlib
import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_unified_main import SaveableImageModel
from test_unified_profiling import make_case, source_function


ROOT = Path(__file__).resolve().parents[1]


def runtime_modules(names):
    modules = {name: types.ModuleType(name) for name in names}
    for module in modules.values():
        module.__path__ = []
    return modules


class UnifiedCLITests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(62)
        self.entry = importlib.import_module("safe_svd.main")

    def test_original_argument_aliases_and_model_defaults(self):
        mpp = self.entry.parse_args(["--model", "mpp", "--fisher_loss", "mse", "--DEV", "cpu"])
        self.assertEqual((mpp.rank_ratio, mpp.calibration_num, mpp.calibration_seed), (0.4, 16, 42))
        self.assertEqual(mpp.gradient_loss, "mse")
        self.assertEqual(mpp.rank_selector, "legacy")
        vicon = self.entry.parse_args(["--model_type", "vicon", "--use_trace_align", "n"])
        self.assertEqual((vicon.rank_ratio, vicon.calibration_num, vicon.calibration_seed), (0.8, 2048, 42))
        self.assertEqual(vicon.batch_size, 64)
        self.assertEqual(vicon.use_trace_balance, "n")
        poseidon = self.entry.parse_args(["--model", "poseidon", "--calibration_seed", "default", "--model_path", "weights"])
        self.assertIsNone(poseidon.calibration_seed)
        self.assertEqual((poseidon.rank_ratio, poseidon.traj_num, poseidon.final_time), (0.2, 64, 20))
        self.assertEqual(poseidon.rank_selector, "safe")
        self.assertEqual((poseidon.whitening_mode, poseidon.eps_whiten_mode, poseidon.eps_whiten),
                         ("new", "sum_equivalent", 1e-6))
        self.assertEqual((mpp.whitening_mode, mpp.eps_whiten_mode), ("new", "absolute"))
        self.assertEqual(poseidon.ckpt_path, Path("weights"))

    def test_poseidon_derivative_only_cache_has_a_distinct_path(self):
        args = self.entry.resolve_cli_paths(self.entry.parse_args(["--model", "poseidon"]))
        self.assertIn("_derivonly_", str(args.fisher_cache))
        self.assertIn("_derivonly_", str(args.rank_cache))
        self.assertIn("new_epssum_equivalent1e-06", str(args.profile_cache))

    def test_cli_dry_run_and_help_without_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for model in ("mpp", "vicon", "poseidon"):
                with patch.object(self.entry, "load_model_and_data", side_effect=AssertionError("dry_run must not load assets")), \
                     patch.object(self.entry, "PROJECT_ROOT", root), \
                     patch("sys.stdout", new_callable=io.StringIO) as output:
                    code = self.entry.cli_main(["--model", model, "--dry_run"])
                self.assertEqual(code, 0)
                self.assertIn(f"Model: {model}", output.getvalue())
                self.assertFalse((root / "artifacts").exists())
                self.assertNotIn("physic_model_comp", output.getvalue())
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                with self.assertRaises(SystemExit) as exit:
                    self.entry.cli_main(["--help"])
            self.assertEqual(exit.exception.code, 0)
            self.assertIn("--model {mpp,vicon,poseidon}", output.getvalue())
            self.assertNotIn("source_root", output.getvalue())

    def test_repository_defaults_and_invocation_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repository, invocation = root / "new_repo", root / "invocation"
            invocation.mkdir()
            previous_cwd = Path.cwd()
            try:
                os.chdir(invocation)
                with patch.object(self.entry, "PROJECT_ROOT", repository):
                    mpp = self.entry.resolve_cli_paths(self.entry.parse_args(["--model", "mpp"]))
                    self.assertEqual(mpp.runtime_root, repository / "mpp/multiple_physics_pretraining")
                    self.assertEqual(mpp.ckpt_path, repository / "models/MPP/MPP_AViT_L")
                    self.assertTrue(mpp.cache_dir.is_relative_to(repository / "artifacts"))
                    vicon = self.entry.resolve_cli_paths(self.entry.parse_args([
                        "--model", "vicon", "--ckpt_path", "weights.pth", "--data_root", "data",
                        "--save_path", "out"]))
                    self.assertEqual(vicon.ckpt_path, invocation / "weights.pth")
                    self.assertEqual(vicon.runtime_root, repository / "VICON")
                    self.assertIn("vicon_research_v3", str(vicon.profile_cache))
                    self.assertEqual(vicon.data_root, invocation / "data")
                    self.assertEqual(vicon.save_path, invocation / "out")
                    poseidon = self.entry.resolve_cli_paths(self.entry.parse_args(["--model", "poseidon"]))
                    self.assertEqual(poseidon.runtime_root, repository / "poseidon")
                    self.assertEqual(poseidon.ckpt_path, repository / "models/poseidon" / poseidon.dataset)
            finally:
                os.chdir(previous_cwd)

    def test_model_source_symlink_cannot_route_outside_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            repository, external = root / "new_repo", root / "external_source"
            repository.mkdir()
            external.mkdir()
            (repository / "poseidon").symlink_to(external, target_is_directory=True)
            with patch.object(self.entry, "PROJECT_ROOT", repository):
                args = self.entry.resolve_cli_paths(self.entry.parse_args(["--model", "poseidon", "--DEV", "cpu"]))
                with self.assertRaisesRegex(ValueError, "inside this repository"):
                    self.entry.load_model_and_data(args)

    def test_vicon_data_root_prefers_aligned_tree_over_prepared_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            prepared = root / "data/processed_data/2D/CFD/converted"
            prepared.mkdir(parents=True)
            with patch.object(self.entry, "PROJECT_ROOT", root):
                args = self.entry.resolve_cli_paths(self.entry.parse_args(["--model", "vicon"]))
                self.assertEqual(args.data_root, prepared)
                (root / "data/VICON").mkdir()
                args = self.entry.resolve_cli_paths(self.entry.parse_args(["--model", "vicon"]))
                self.assertEqual(args.data_root, root / "data/VICON")
                explicit = self.entry.resolve_cli_paths(self.entry.parse_args([
                    "--model", "vicon", "--data_root", str(prepared)]))
                self.assertEqual(explicit.data_root, prepared)

    def test_poseidon_fisher_batch_size_is_part_of_cache_identity(self):
        first = self.entry.resolve_cli_paths(self.entry.parse_args([
            "--model", "poseidon", "--fisher_batch_size", "8"]))
        second = self.entry.resolve_cli_paths(self.entry.parse_args([
            "--model", "poseidon", "--fisher_batch_size", "16"]))
        self.assertEqual(first.profile_cache, second.profile_cache)
        self.assertNotEqual(first.fisher_cache, second.fisher_cache)
        self.assertNotEqual(first.rank_cache, second.rank_cache)

    def test_poseidon_requires_official_source_and_rollout(self):
        from safe_svd.helper.collector.poseidon.input_collector import run_rollout
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            # An old research checkout must never be considered as a fallback.
            (root / "poseidon_main/scOT").mkdir(parents=True)
            (root / "weights").mkdir()
            with patch.object(self.entry, "PROJECT_ROOT", root), patch.object(sys, "path", sys.path[:]):
                args = self.entry.resolve_cli_paths(self.entry.parse_args([
                    "--model", "poseidon", "--model_path", str(root / "weights"), "--DEV", "cpu"]))
                with self.assertRaisesRegex(FileNotFoundError, "Official Poseidon source is missing"):
                    self.entry.load_model_and_data(args)
                self.assertNotIn(str(root / "poseidon_main"), sys.path)
            modules = runtime_modules(("scOT", "scOT.inference", "poseidon_main", "poseidon_main.scOT",
                                       "poseidon_main.scOT.inference"))
            calls = []
            modules["scOT.inference"].rollout = lambda *args, **kwargs: calls.append((args, kwargs))
            def forbidden_rollout(*args, **kwargs):
                raise AssertionError("Research runtime must never be called")
            modules["poseidon_main.scOT.inference"].rollout = forbidden_rollout
            trainer = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(name_or_path="weights")))
            with patch.dict(sys.modules, modules):
                run_rollout(trainer, "data")
            self.assertEqual(calls, [((trainer, "data"), {})])

    def test_mpp_loader_and_full_cli(self):
        _model, dataset, options, _args = make_case("mpp")
        model = SaveableImageModel("mpp")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            package = root / "mpp/multiple_physics_pretraining"
            (package / "models").mkdir(parents=True)
            (package / "models/avit.py").write_text("# Test official runtime\n")
            (package / "config").mkdir()
            (package / "config/mpp_avit_L_config.yaml").write_text("basic_config: {}\n")
            checkpoint = root / "weights.pt"
            torch.save({"model_state": {"module." + key: value for key, value in model.state_dict().items()}}, checkpoint)
            modules = runtime_modules(("multiple_physics_pretraining", "multiple_physics_pretraining.utils",
                "multiple_physics_pretraining.utils.YParams", "multiple_physics_pretraining.models",
                "multiple_physics_pretraining.models.avit", "multiple_physics_pretraining.data_utils",
                "multiple_physics_pretraining.data_utils.datasets"))
            params = options["params"]
            params.train_data_paths, params.valid_data_paths = [["train.h5", "swe"]], [["val.h5", "swe"]]
            params.embedding_offset = 0
            loaded = []
            modules["multiple_physics_pretraining.utils.YParams"].YParams = lambda path, config: params
            modules["multiple_physics_pretraining.models.avit"].build_avit = lambda parameters: model
            def data_loader(parameters, paths, **kwargs):
                loaded.append((paths, kwargs))
                return None, dataset, None
            modules["multiple_physics_pretraining.data_utils.datasets"].get_data_loader = data_loader
            with patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path[:]), patch.object(self.entry, "PROJECT_ROOT", root):
                arguments = ["--model", "mpp", "--ckpt_path", str(checkpoint),
                    "--data_root", str(root / "data"), "--calibration_num", "4", "--rank_ratio", "0.5", "--DEV", "cpu",
                    "--cache_dir", str(root / "cache"), "--rank_root", str(root / "ranks"), "--save_path", str(root / "out")]
                self.assertEqual(self.entry.cli_main(arguments + ["--pipeline_stage", "rank_selection"]), 0)
                code = self.entry.cli_main(arguments + ["--pipeline_stage", "compression"])
            self.assertEqual(code, 0)
            self.assertEqual(loaded[0][0][0][0], str(root / "data/train.h5"))
            self.assertEqual(loaded[0][1]["split"], "train")
            self.assertFalse(params.use_ddp)
            self.assertFalse(params.log_to_wandb)
            compressed = torch.load(root / "out/compressed_model.pt", weights_only=False)
            self.assertIsInstance(compressed.first, torch.nn.Sequential)

    def test_vicon_loader_handles_current_dataset_signature(self):
        model, loaders, options, _args = make_case("vicon")
        trainer = options["trainer"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "VICON/src").mkdir(parents=True)
            (root / "VICON/configs").mkdir()
            checkpoint = root / "weights.pth"
            torch.save({"model_state_dict": model.state_dict()}, checkpoint)
            cfg = SimpleNamespace(model=SimpleNamespace(type="nocrop"), opt=SimpleNamespace(), loss=trainer.loss_cfg,
                                  trainable_mode="all", train_seed=84, dataset_workers=0, rollout=SimpleNamespace(),
                                  datasets=SimpleNamespace(types={"NS2D": SimpleNamespace(folder="../icon-data/NS"),
                                                                 "EULER2D": SimpleNamespace(folder="../icon-data/Euler")}))
            calls = []
            modules = runtime_modules(("hydra", "omegaconf", "VICON", "VICON.src", "VICON.src.models",
                                       "VICON.src.trainer", "VICON.src.dataset"))
            modules["hydra"].initialize_config_dir = lambda **kwargs: nullcontext()
            modules["hydra"].compose = lambda **kwargs: cfg
            modules["omegaconf"].open_dict = lambda config: nullcontext()
            modules["VICON.src.models"].ICON_UNCROPPED = lambda config: model
            modules["VICON.src.models"].ICON_CROPPED = lambda config: model
            modules["VICON.src.trainer"].Trainer = lambda model, *args, **kwargs: trainer
            def all_datasets(config, workers, seed, mode, rollout_cfg):
                calls.append((seed, mode, rollout_cfg))
                return {name: loader.dataset for name, loader in loaders.items()}
            modules["VICON.src.dataset"].all_datasets = all_datasets
            with patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path[:]), patch.object(self.entry, "PROJECT_ROOT", root):
                arguments = ["--model", "vicon", "--ckpt_path", str(checkpoint),
                    "--data_root", str(root / "data"), "--batch_size", "2", "--calibration_num", "4", "--random_calibration", "y",
                    "--calibration_seed", "19", "--rank_ratio", "0.5", "--DEV", "cpu",
                    "--vicon_file_order", "filesystem", "--cache_dir", str(root / "cache"),
                    "--rank_root", str(root / "ranks"), "--save_path", str(root / "out")]
                self.assertEqual(self.entry.cli_main(arguments + ["--pipeline_stage", "rank_selection"]), 0)
                code = self.entry.cli_main(arguments + ["--pipeline_stage", "compression"])
            self.assertEqual(code, 0)
            self.assertEqual(calls[0][:2], (19, "train"))
            self.assertIs(calls[0][2], cfg.rollout)
            self.assertEqual(cfg.datasets.types["NS2D"].folder, str(root / "data/NS"))
            self.assertEqual(trainer.device, torch.device("cpu"))
            self.assertTrue((root / "out/compressed_model.pth").is_file())

    def test_poseidon_loader_random_trajectories_and_full_cli(self):
        _model, dataset, options, _args = make_case("poseidon")
        model, trainer = SaveableImageModel("poseidon"), options["trainer"]
        trainer.model = model
        dataset.num_trajectories = 3
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "poseidon/scOT").mkdir(parents=True)
            (root / "weights").mkdir()
            modules = runtime_modules(("scOT", "scOT.inference", "scOT.problems", "scOT.problems.base"))
            calls = {}
            def get_dataset(**kwargs):
                calls["dataset"] = kwargs
                return dataset
            def get_trainer(**kwargs):
                calls["trainer"] = kwargs
                trainer.args = SimpleNamespace(dataloader_num_workers=99, remove_unused_columns=True)
                return trainer
            modules["scOT.problems.base"].get_dataset = get_dataset
            modules["scOT.inference"].get_trainer = get_trainer
            modules["scOT.inference"].rollout = options["rollout_fn"]
            with patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path[:]), patch.object(self.entry, "PROJECT_ROOT", root), \
                 patch.object(self.entry, "compress", wraps=self.entry.compress) as compressor:
                code = self.entry.cli_main(["--model", "poseidon", "--model_path", str(root / "weights"),
                    "--data_path", str(root / "data"), "--dataset", "wave", "--traj_num", "2", "--calibration_seed", "0",
                    "--final_time", "3", "--rank_ratio", "0.5", "--whitening_mode", "new", "--DEV", "cpu",
                    "--fisher_batch_size", "2", "--gpu_profile", "--cache_dir", str(root / "cache"),
                    "--rank_root", str(root / "ranks"), "--save_path", str(root / "out")])
            self.assertEqual(code, 0)
            self.assertTrue(compressor.call_args.kwargs["route_options"]["offload_cache"])
            self.assertEqual(compressor.call_args.kwargs["compute_dtype"], torch.float64)
            self.assertEqual(calls["dataset"]["num_trajectories"], -1)
            self.assertEqual(calls["dataset"]["max_num_time_steps"], 3)
            selected = calls["trainer"]["dataset"]
            self.assertEqual(selected.num_trajectories, 2)
            self.assertEqual(len(selected), 4)
            self.assertEqual(calls["trainer"]["batch_size"], 256)
            self.assertEqual(calls["trainer"]["workers"], 1)
            self.assertEqual(trainer.args.dataloader_num_workers, 0)
            with (root / "out/gpu_profile.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["stage"] for row in rows], ["input_collection", "fisher_collection", "rank_selection", "compression"])
            self.assertTrue((root / "out/compressed_model.pt").is_file())
            import json
            manifest = json.loads((root / "out/run_config.json").read_text())
            self.assertEqual(manifest["parameters"]["fisher_batch_size"], 2)
            self.assertEqual(manifest["parameters"]["whitening_mode"], "new")
            self.assertEqual(manifest["parameters"]["rank_selector"], "safe")
            self.assertEqual(manifest["completed_stage"], "all")
            self.assertEqual(len(manifest["rank_cache_sha256"]), 64)

    def test_poseidon_scaled_rank_matches_original_selector(self):
        from safe_svd.rank import allocate_ranks
        import os
        if not os.environ.get("SAFE_SVD_REFERENCE_ROOT"):
            self.skipTest("Original checkout required")
        path = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]) / "SAFE_SVD/rank_selector.py"
        namespace = {"torch": torch, "nn": torch.nn, "heapq": __import__("heapq"),
                     "itertools": __import__("itertools"), "tqdm": lambda values: values}
        original = source_function(path, "_get_global_greedy_rank_dict", namespace)
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        profiles = {str(index): (torch.eye(8), torch.eye(8)) for index in range(2)}
        counts = {"0": 4, "1": 100}
        expected = original(SimpleNamespace(model=model), None, profiles, 0.6, energy_scales=counts, dev="cpu")
        actual = allocate_ranks(model, None, profiles, 0.6, "cpu", selector="safe", energy_scales=counts)
        self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            allocate_ranks(model, None, profiles, 0.6, "cpu", energy_scales={"0": 0, "1": 100})

    def test_rank_only_then_compression_only_use_saved_stage_outputs(self):
        model, data, options, forward_args = make_case("mpp")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = dict(model_type="mpp", rank_ratio=0.5, device="cpu", params=options["params"],
                          batches_per_dset=2, calibration_seed=17, verbose=False,
                          profile_cache=root / "R.pt", fisher_cache=root / "L.pt", rank_cache=root / "rank.json")
            result = self.entry.main(model, data, pipeline_stage="rank_selection", **common)
            self.assertIsNone(result)
            self.assertTrue((root / "rank.json").is_file())
            with patch.object(self.entry, "collect_inputs", side_effect=AssertionError("Read input cache")), \
                 patch.object(self.entry, "collect_fisher", side_effect=AssertionError("Read Fisher cache")), \
                 patch.object(self.entry, "allocate_ranks", side_effect=AssertionError("Read rank cache")):
                student = self.entry.main(model, data, pipeline_stage="compression", **common)
            self.assertTrue(torch.isfinite(student(*forward_args)).all())

    def test_old_poseidon_input_cache_rebuilds_counts_for_new_selector(self):
        from safe_svd.helper.collector.input_collector import collect_inputs
        model, data, options, _args = make_case("poseidon")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factors, _ = collect_inputs(model, data, model_type="poseidon", device="cpu", route_options=options, whitening="spectral")
            torch.save(factors, root / "R.pt")  # Original factors-only format.
            with patch.object(self.entry, "collect_inputs", wraps=self.entry.collect_inputs) as profile:
                self.entry.main(model, data, model_type="poseidon", rank_ratio=0.5, device="cpu", trainer=options["trainer"],
                    rollout_fn=options["rollout_fn"], dataset_name="wave", whitening_mode="new", use_fisher="n",
                    profile_cache=root / "R.pt", rank_cache=root / "rank.json", pipeline_stage="rank_selection", verbose=False)
                profile.assert_called_once()
            rebuilt = torch.load(root / "R.pt", weights_only=True)
            self.assertTrue(all(count > 0 for count in rebuilt["sample_counts"].values()))


if __name__ == "__main__":
    unittest.main()
