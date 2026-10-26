"""Regression checks against the MPP MySVD compression/evaluation source."""

import copy
import importlib
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from test_unified_profiling import make_case, source_function
from safesvd.adapters.mpp_paths import default_mpp_package
from safesvd.adapters.mpp_runtime import (
    MPPBundle, MPPOptions, _checkpoint_state_dict, _rebase_data_paths,
    _mpp_batch_factory, _evaluate_mpp_model, run_mpp,
)


class MPPAlignmentTests(unittest.TestCase):
    def test_checkout_discovery_supports_both_layouts_and_ignores_empty_clone(self):
        entry = importlib.import_module("safe_svd.main")
        from safesvd.adapters.mpp import MPPAdapter
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "mpp/multiple_physics_pretraining"
            flat = root / "multiple_physics_pretraining"
            nested.mkdir(parents=True)
            (flat / "models").mkdir(parents=True)
            (flat / "models/avit.py").touch()
            self.assertEqual(default_mpp_package(root), flat)
            with patch.object(entry, "PROJECT_ROOT", root):
                args = entry.resolve_cli_paths(entry.parse_args(["--model", "mpp"]))
            self.assertEqual(args.runtime_root, flat)
            self.assertEqual(args.yaml_config, root / "safe_svd/configs/mpp_avit_L_config.yaml")
            request = SimpleNamespace(repo_root=root, config=SimpleNamespace(source_root=None))
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(MPPAdapter().source_root(request), flat.resolve())
            (nested / "models").mkdir()
            (nested / "models/avit.py").touch()
            self.assertEqual(default_mpp_package(root), nested)

    def test_cache_and_output_paths_isolate_config_and_selector(self):
        entry = importlib.import_module("safe_svd.main")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text("basic_config: {batch_size: 1}\n")
            def resolve(selector):
                with patch.object(entry, "PROJECT_ROOT", root):
                    return entry.resolve_cli_paths(entry.parse_args([
                        "--model", "mpp", "--yaml_config", str(config), "--rank_selector", selector]))
            first, safe = resolve("legacy"), resolve("safe")
            self.assertIn("mpp_research_v3", str(first.profile_cache))
            self.assertEqual(first.profile_cache, safe.profile_cache)
            self.assertNotEqual(first.rank_cache, safe.rank_cache)
            self.assertNotEqual(first.save_path, safe.save_path)
            config.write_text("basic_config: {batch_size: 2}\n")
            self.assertNotEqual(first.profile_cache, resolve("legacy").profile_cache)

    def test_checkpoint_wrappers_and_expanded_data_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            weight = torch.randn(2, 3)
            for key in (None, "model_state", "model_state_dict", "state_dict"):
                state = {"module.layer.weight": weight}
                torch.save({key: state} if key else state, path)
                torch.testing.assert_close(_checkpoint_state_dict(path)["layer.weight"], weight)
        params = SimpleNamespace(train_data_paths=[["2D/shallow-water", "swe", ""]],
                                 valid_data_paths=[["~/PDEBench/2D/shallow-water", "swe", ""]])
        _rebase_data_paths(params, Path("/data/raw_data"))
        self.assertEqual(params.train_data_paths[0][0], "/data/raw_data/2D/shallow-water")
        self.assertEqual(params.valid_data_paths[0][0], str(Path("~/PDEBench/2D/shallow-water").expanduser()))

    def test_public_compression_uses_unified_pipeline_including_alpha_zero(self):
        model, data, options, _ = make_case("mpp")
        bundle = MPPBundle(Path("mpp"), options["params"], model, data, None, torch.device("cpu"))
        cfg = MPPOptions(Path("mpp"), Path("model"), Path("data"), Path("out"),
                         Path("profile"), Path("fisher"), Path("ranks"), .4, "legacy", 5, 42, 0., "cpu")
        with patch("safesvd.adapters.mpp_runtime._validate_options"), \
             patch("safesvd.adapters.mpp_runtime.load_mpp_bundle", return_value=bundle) as load, \
             patch("safe_svd.main.main") as run:
            self.assertEqual(run_mpp(cfg), Path("out/compressed_model.pt"))
        self.assertFalse(load.call_args.kwargs["include_validation"])
        self.assertEqual(run.call_args.kwargs["batches_per_dset"], 2.5)
        self.assertEqual(run.call_args.kwargs["compress_mode"], "comb")
        self.assertEqual(run.call_args.kwargs["rank_selector"], "legacy")

    def test_mpp_factorization_requests_research_float32_factors(self):
        module = importlib.import_module("safe_svd.compress")
        model, data, options, _ = make_case("mpp")
        with patch.object(module, "replace_linear", wraps=module.replace_linear) as replace:
            module.compress(model, data, {"first": 1, "second": 1}, model_type="mpp",
                            route_options=options, device="cpu", progress=False)
        self.assertEqual(len(replace.call_args_list), 2)
        self.assertTrue(all(call.kwargs["factor_dtype"] == torch.float32 for call in replace.call_args_list))

    def test_public_compress_checkpoint_reloads_and_evaluates(self):
        from test_unified_main import SaveableImageModel
        from safesvd.adapters.common_runtime import load_compressed_state, save_portable_checkpoint
        from safesvd.adapters.mpp_runtime import evaluate_mpp, MPPEvaluationOptions
        import json
        _, data, route, args = make_case("mpp")
        model = SaveableImageModel("mpp").eval()
        bundle = MPPBundle(Path("mpp"), route["params"], model, data, data, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = MPPOptions(Path("mpp"), Path("model"), Path("data"), root / "out",
                                 root / "profile.pt", root / "fisher.pt", root / "ranks.json",
                                 .5, "legacy", 4, 17, .7, "cpu")
            with patch("safesvd.adapters.mpp_runtime._validate_options"), \
                 patch("safesvd.adapters.mpp_runtime.load_mpp_bundle", return_value=bundle):
                checkpoint = run_mpp(options)
            expected = torch.load(checkpoint, weights_only=False).eval()
            self.assertIsInstance(expected, nn.Module)
            restored = load_compressed_state(copy.deepcopy(model), checkpoint).eval()
            torch.testing.assert_close(restored(*args), expected(*args))
            portable = save_portable_checkpoint(root / "portable.pt", expected, model_name="mpp", ranks={})
            old_format = load_compressed_state(copy.deepcopy(model), portable).eval()
            torch.testing.assert_close(old_format(*args), expected(*args))
            evaluation = MPPEvaluationOptions(Path("mpp"), Path("model"), Path("data"), checkpoint,
                                             root / "metrics.json", "cpu", batch_size=2, max_batches=2)
            def fresh_bundle(*_args, **_kwargs):
                return MPPBundle(Path("mpp"), route["params"], copy.deepcopy(model), data, data, torch.device("cpu"))
            with patch("safesvd.adapters.mpp_runtime.load_mpp_bundle", side_effect=fresh_bundle):
                evaluate_mpp(evaluation)
            result = json.loads(evaluation.output_path.read_text())
            self.assertEqual(result["protocol"], "research_per_file")
            self.assertEqual(result["origin"]["metrics"]["datasets"]["toy"]["batches"], 2)
            self.assertLess(result["compressed"]["metrics"]["parameters"], result["origin"]["metrics"]["parameters"])

    def test_old_mpp_partial_checkpoint_is_rejected(self):
        entry = importlib.import_module("safe_svd.main")
        model, data, options, _ = make_case("mpp")
        with tempfile.TemporaryDirectory() as directory:
            torch.save({"compressed_layers": ["first"], "model": nn.Linear(1, 1)},
                       Path(directory) / "compressing_model.pt")
            with patch.object(entry, "collect_inputs", return_value=({}, {})), \
                 patch.object(entry, "allocate_ranks", return_value={}):
                with self.assertRaisesRegex(ValueError, "predates research alignment"):
                    entry.main(model, data, model_type="mpp", params=options["params"], rank_ratio=.4,
                               use_fisher=False, device="cpu", save_path=directory, verbose=False)

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Original checkout required")
    def test_sequential_compression_and_legacy_ranks_match_reference(self):
        from safe_svd.compress import compress
        from safe_svd.rank import allocate_ranks
        path = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]).parent / "mpp/MySVD/mysvd.py"
        ns = {"torch": torch, "nn": nn, "copy": copy, "os": os,
              "DataLoader": torch.utils.data.DataLoader, "tqdm": lambda values: values,
              "StopForwardException": type("StopForwardException", (Exception,), {}),
              "rearrange": lambda value, _: value.permute(1, 0, 2, 3, 4)}
        for name in ("get_module_by_name", "set_module_by_name", "svd_whitener",
                     "mysvd_compress_mpp", "get_global_greedy_rank_dict_mpp"):
            source_function(path, name, ns)
        for alpha in (0., .7, 1.):
            for trace in (False, True):
                with self.subTest(alpha=alpha, trace=trace):
                    torch.manual_seed(42)
                    model, data, options, args = make_case("mpp")
                    left, right = {}, {}
                    for name, layer in model.named_modules():
                        if isinstance(layer, nn.Linear):
                            l = torch.diag(torch.linspace(.7, 1.8, layer.out_features))
                            r = torch.diag(torch.linspace(.8, 2., layer.in_features))
                            left[name], right[name] = (l, torch.linalg.inv(l)), (r, torch.linalg.inv(r))
                    for ratio in (.2, .4, .8, 1.):
                        expected_ranks = ns["get_global_greedy_rank_dict_mpp"](model, left, right, ratio, "cpu")
                        self.assertEqual(allocate_ranks(model, left, right, ratio, "cpu", selector="legacy"), expected_ranks)
                    ranks = {"first": 1, "second": 2}
                    expected = ns["mysvd_compress_mpp"](model, data, options["params"], "cpu", ranks,
                        Ldict=left, alpha=alpha, batches_per_dset=2, calibration_seed=17, use_trace_balance=trace)
                    actual = compress(model, data, ranks, model_type="mpp", device="cpu", route_options=options,
                                      left_metrics=left, alpha=alpha, trace_balance=trace, progress=False)
                    for name in ranks:
                        a, b = actual.get_submodule(name), expected.get_submodule(name)
                        torch.testing.assert_close(a[1].weight @ a[0].weight, b[1].weight @ b[0].weight,
                                                   atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(actual(*args), expected(*args), atol=2e-6, rtol=2e-5)

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Original checkout required")
    def test_full_evaluation_matches_reference_all_four_physics_categories(self):
        from safesvd.adapters.mpp_physics import SOBOLEV_ORDERS
        path = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]).parent / "mpp/eval/eval_all.py"
        ns = {"np": np, "torch": torch,
              "amp": SimpleNamespace(autocast=lambda *args, **kwargs: nullcontext()),
              "rearrange": lambda value, _: value.permute(1, 0, 2, 3, 4),
              "DSET_SOBOLEV_CONFIG": {key: {"order": value} for key, value in SOBOLEV_ORDERS.items()}}
        for name in ("_central_dx", "_central_dy", "calc_sobolev", "calc_incompNS_metrics",
                     "calc_compNS_metrics", "calc_swe_metrics", "calc_diffre2d_metrics", "evaluate_model"):
            source_function(path, name, ns)
        ns["DSET_METRIC_FN"] = {key: ns[f"calc_{key}_metrics"] for key in SOBOLEV_ORDERS}
        torch.manual_seed(42)
        model, data, options, _ = make_case("mpp")
        subset = data.sub_dsets[0].get_per_file_dsets()[0]
        files = []
        for category in SOBOLEV_ORDERS:
            current = copy.deepcopy(subset)
            current.title = category + "_file"
            current.get_name = lambda category=category: category
            files.append(current)
        dataset = SimpleNamespace(sub_dsets=[SimpleNamespace(get_per_file_dsets=lambda: files)],
                                  subset_dict={key: list(range(4)) for key in SOBOLEV_ORDERS})
        bundle = MPPBundle(Path("mpp"), options["params"], model, dataset, dataset, torch.device("cpu"))
        batches = _mpp_batch_factory(bundle, dataset, labels_from=dataset, calibration_size=None, seed=0, max_batches=2)
        actual = _evaluate_mpp_model(bundle, model, batches)["datasets"]
        mse, relative, physics = ns["evaluate_model"](options["params"], model, dataset, dataset, "cpu", max_batches=2)
        for title, metrics in actual.items():
            self.assertEqual(metrics["batches"], 2)
            self.assertAlmostEqual(metrics["mse"], mse[title], places=7)
            self.assertAlmostEqual(metrics["rel_mse"], relative[title], places=7)
            for key, value in physics[title].items():
                np.testing.assert_allclose(metrics[key], value, rtol=1e-6, atol=1e-6)
