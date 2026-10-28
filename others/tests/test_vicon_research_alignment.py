"""VICON process boundaries, cache compatibility and MySVD regressions."""

import copy
import importlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from test_unified_profiling import make_case, source_function
from safe_svd.helper.collector import vicon_rng


class Stream(IterableDataset):
    """Same RNG ownership and mutable path ordering as VICON datasetBase."""
    def __init__(self):
        self.rng = self.tc_rng = None
        self.dp = ["a", "b", "c"]
        self.epoch = 0

    def __iter__(self):
        if self.rng is None:
            seed = int(np.random.randint(1000000))
            self.rng = np.random.default_rng(seed)
            self.tc_rng = torch.Generator().manual_seed(seed)
        self.rng.shuffle(self.dp)
        for _ in self.dp:
            yield torch.tensor(self.rng.normal()) + torch.rand((), generator=self.tc_rng)


class ViconResearchAlignmentTests(unittest.TestCase):
    def test_stream_state_is_weights_only_loadable_and_replays_next_pass(self):
        loaders = {"NS2D": DataLoader(Stream(), batch_size=1)}
        list(loaders["NS2D"])
        buffer = io.BytesIO()
        torch.save(vicon_rng.capture(loaders), buffer)
        expected = list(loaders["NS2D"])
        expected_global = (random.random(), np.random.random(), torch.rand(()))
        buffer.seek(0)
        fresh = {"NS2D": DataLoader(Stream(), batch_size=1)}
        vicon_rng.restore(fresh, torch.load(buffer, weights_only=True))
        for actual, wanted in zip(fresh["NS2D"], expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        self.assertEqual(random.random(), expected_global[0])
        self.assertEqual(np.random.random(), expected_global[1])
        torch.testing.assert_close(torch.rand(()), expected_global[2], rtol=0, atol=0)
        self.assertEqual(fresh["NS2D"].dataset.dp, loaders["NS2D"].dataset.dp)

    def test_missing_compression_stream_state_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            vicon_rng.restore({}, None)

    def test_cache_rejects_changed_batch_size(self):
        state = vicon_rng.capture({"NS2D": DataLoader(Stream(), batch_size=1)})
        with self.assertRaisesRegex(ValueError, "batch size/worker"):
            vicon_rng.restore({"NS2D": DataLoader(Stream(), batch_size=2)}, state)

    def test_resume_rejects_changed_category_order(self):
        loaders = {name: DataLoader(Stream(), batch_size=1) for name in ("NS2D", "EULER2D")}
        state = vicon_rng.capture(loaders)
        with self.assertRaises(ValueError):
            vicon_rng.restore(dict(reversed(list(loaders.items()))), state)

    @staticmethod
    def _seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    @classmethod
    def _fresh_stream_case(cls, seed):
        cls._seed(seed)
        model = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 3))
        return model, {"NS2D": DataLoader(Stream(), batch_size=1)}

    @staticmethod
    def _consume(loaders):
        return torch.cat(list(loaders["NS2D"]))

    def test_reference_cache_hits_skip_sampling_and_compression_starts_fresh(self):
        entry = importlib.import_module("safe_svd.main")
        for input_hit, fisher_hit in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(input_hit=input_hit, fisher_hit=fisher_hit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                # The original caches contain factors only, with no RNG payload.
                if input_hit:
                    torch.save({}, root / "input.pt")
                if fisher_hit:
                    torch.save({}, root / "fisher.pt")

                # The rank process only iterates stages whose caches are absent.
                _, reference_loaders = self._fresh_stream_case(701)
                expected = {}
                if not input_hit:
                    self._seed(42)  # Original profile_svd_vicon seed boundary.
                    expected["profile"] = self._consume(reference_loaders)
                if not fisher_hit:
                    expected["fisher"] = self._consume(reference_loaders)
                # A separate compression process gets a new loader. The Python
                # API models that boundary for a fixed caller-supplied RNG state;
                # it does not claim to reproduce an old process's unknown entropy.
                _, reference_loaders = self._fresh_stream_case(701)
                expected["compress"] = self._consume(reference_loaders)
                seen = {}

                def profile(model, loaders, **kwargs):
                    self._seed(42)
                    seen["profile"] = self._consume(loaders)
                    return {}, {}

                def fisher(model, loaders, **kwargs):
                    seen["fisher"] = self._consume(loaders)
                    return {}

                def compress(model, loaders, *args, **kwargs):
                    seen["compress"] = self._consume(loaders)
                    return model

                model, loaders = self._fresh_stream_case(701)
                with patch.object(entry, "collect_inputs", side_effect=profile), \
                     patch.object(entry, "collect_fisher", side_effect=fisher), \
                     patch.object(entry, "allocate_ranks", return_value={}), \
                     patch.object(entry, "compress", side_effect=compress):
                    entry.main(model, loaders, trainer=SimpleNamespace(model=model), model_type="vicon",
                               rank_ratio=.5, device="cpu", profile_cache=root / "input.pt",
                               fisher_cache=root / "fisher.pt", rank_cache=root / "ranks.json", verbose=False)
                self.assertEqual(set(seen), set(expected))
                for stage in expected:
                    torch.testing.assert_close(seen[stage], expected[stage], rtol=0, atol=0)
                for path, cache_hit in ((root / "input.pt", input_hit), (root / "fisher.pt", fisher_hit)):
                    if not cache_hit:
                        payload = torch.load(path, weights_only=True)
                        self.assertEqual(payload["vicon_alignment_version"], 3)
                        self.assertNotIn("vicon_rng_state", payload)

    def test_old_structured_statistics_caches_are_rejected(self):
        entry = importlib.import_module("safe_svd.main")
        valid_input = {"profile_mat": {}, "sample_counts": {}, "vicon_alignment_version": 3}
        valid_fisher = {"fisher_mat": {}, "vicon_alignment_version": 3}
        old_caches = (
            ("input", {"profile_mat": {}, "sample_counts": {}}),
            ("input", {"profile_mat": {}, "sample_counts": {}, "vicon_alignment_version": 2}),
            ("input", {"profile_mat": {}, "sample_counts": {}, "vicon_rng_state": {}}),
            ("fisher", {"fisher_mat": {}}),
            ("fisher", {"fisher_mat": {}, "vicon_alignment_version": 2}),
            ("fisher", {"fisher_mat": {}, "vicon_rng_state": {}}),
        )
        for stage, payload in old_caches:
            with self.subTest(stage=stage, fields=list(payload)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                torch.save(payload if stage == "input" else valid_input, root / "input.pt")
                torch.save(payload if stage == "fisher" else valid_fisher, root / "fisher.pt")
                model, loaders = self._fresh_stream_case(701)
                with patch.object(entry, "collect_inputs", side_effect=AssertionError("unexpected input recomputation")), \
                     patch.object(entry, "collect_fisher", side_effect=AssertionError("unexpected Fisher recomputation")), \
                     patch.object(entry, "allocate_ranks", return_value={}), \
                     self.assertRaisesRegex(ValueError, "(?i)cache|alignment|version"):
                    entry.main(model, loaders, trainer=SimpleNamespace(model=model), model_type="vicon",
                               rank_ratio=.5, device="cpu", profile_cache=root / "input.pt",
                               fisher_cache=root / "fisher.pt", pipeline_stage="rank_selection", verbose=False)

    def test_disabling_compression_fisher_keeps_reference_fisher_rank_selection(self):
        entry = importlib.import_module("safe_svd.main")
        model, loaders = self._fresh_stream_case(701)
        fisher = {"0": (torch.eye(5), torch.eye(5))}
        ranks = {"0": 1, "1": 1}
        with tempfile.TemporaryDirectory() as directory:
            rank_path = Path(directory) / "rank.json"
            with patch.object(entry, "collect_inputs", return_value=({}, {})), \
                 patch.object(entry, "collect_fisher", return_value=fisher) as collect, \
                 patch.object(entry, "allocate_ranks", return_value=ranks) as allocate, \
                 patch.object(entry, "compress", return_value=model) as compress:
                entry.main(model, loaders, trainer=SimpleNamespace(model=model), model_type="vicon",
                           rank_ratio=.5, rank_selector="legacy", use_fisher="n", device="cpu",
                           rank_cache=rank_path, verbose=False)
            collect.assert_called_once()
            allocate.assert_called_once()
            left = allocate.call_args.kwargs["left_whiteners"]
            self.assertEqual(set(left), set(fisher))
            for actual, expected in zip(left["0"], fisher["0"]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertIsNone(compress.call_args.kwargs["left_metrics"])
            self.assertEqual(json.loads(rank_path.read_text()), ranks)

    def test_rank_stage_recomputes_an_existing_rank_file(self):
        entry = importlib.import_module("safe_svd.main")
        model, loaders = self._fresh_stream_case(701)
        ranks = {"0": 1, "1": 1}
        with tempfile.TemporaryDirectory() as directory:
            rank_path = Path(directory) / "rank.json"
            rank_path.write_text(json.dumps({"0": None, "1": None}))
            with patch.object(entry, "collect_inputs", return_value=({}, {})), \
                 patch.object(entry, "collect_fisher", return_value={}), \
                 patch.object(entry, "allocate_ranks", return_value=ranks) as allocate, \
                 patch.object(entry, "compress") as compress:
                entry.main(model, loaders, trainer=SimpleNamespace(model=model), model_type="vicon",
                           rank_ratio=.5, device="cpu", rank_cache=rank_path,
                           pipeline_stage="rank_selection", verbose=False)
            allocate.assert_called_once()
            compress.assert_not_called()
            self.assertEqual(json.loads(rank_path.read_text()), ranks)

    def test_resume_replays_original_prefetch_after_student_reconstruction(self):
        entry = importlib.import_module("safe_svd.main")
        seen = []
        ranks = {"0": 1, "1": 1}

        def profile(model, loaders, **kwargs):
            self._seed(42)
            self._consume(loaders)
            return {}, {}

        def fisher(model, loaders, **kwargs):
            self._consume(loaders)
            return {}

        def interrupted_compress(model, loaders, *args, **kwargs):
            seen.append((self._consume(loaders), torch.rand(4)))
            student = copy.deepcopy(model)
            student[0] = nn.Sequential(nn.Linear(3, 1, bias=False), nn.Linear(1, 5))
            kwargs["checkpoint_fn"](student, {"0"})
            raise RuntimeError("synthetic interruption after first compressed layer")

        def resumed_compress(model, loaders, *args, **kwargs):
            self.assertEqual(kwargs["completed_layers"], {"0"})
            self.assertIsInstance(kwargs["initial_student"][0], nn.Sequential)
            seen.append((self._consume(loaders), torch.rand(4)))
            return kwargs["initial_student"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = dict(model_type="vicon", rank_ratio=.5, device="cpu", profile_cache=root / "input.pt",
                           fisher_cache=root / "fisher.pt", rank_cache=root / "ranks.json", save_path=root / "out",
                           verbose=False)
            model, loaders = self._fresh_stream_case(701)
            original_weights = copy.deepcopy(model.state_dict())
            with patch.object(entry, "collect_inputs", side_effect=profile), \
                 patch.object(entry, "collect_fisher", side_effect=fisher), \
                 patch.object(entry, "allocate_ranks", return_value=ranks), \
                 patch.object(entry, "compress", side_effect=interrupted_compress):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    entry.main(model, loaders, trainer=SimpleNamespace(model=model), **options)
            self.assertTrue((root / "out/compressing_model.pt").is_file())

            # A restart has unrelated global RNG state. Reconstructing low-rank
            # nn.Linear modules also consumes Torch RNG before the resume hook.
            model, loaders = self._fresh_stream_case(983)
            model.load_state_dict(original_weights)
            with patch.object(entry, "collect_inputs", side_effect=AssertionError("unexpected input recomputation")), \
                 patch.object(entry, "collect_fisher", side_effect=AssertionError("unexpected Fisher recomputation")), \
                 patch.object(entry, "allocate_ranks", return_value=ranks), \
                 patch.object(entry, "compress", side_effect=resumed_compress):
                entry.main(model, loaders, trainer=SimpleNamespace(model=model), **options)
            self.assertFalse((root / "out/compressing_model.pt").exists())
            self.assertTrue((root / "out/compressed_model.pth").is_file())
        for actual, expected in zip(seen[1], seen[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_resume_rejects_changed_provenance_and_statistics(self):
        entry = importlib.import_module("safe_svd.main")

        def interrupted_compress(model, loaders, *args, **kwargs):
            kwargs["checkpoint_fn"](copy.deepcopy(model), set())
            raise RuntimeError("synthetic interruption")

        for changed in ("run_provenance", "statistics_files"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                options = dict(model_type="vicon", rank_ratio=.5, device="cpu", profile_cache=root / "input.pt",
                               fisher_cache=root / "fisher.pt", rank_cache=root / "ranks.json", save_path=root / "out",
                               run_provenance={"checkpoint": "checkpoint-a", "data": "data-a"}, verbose=False)
                model, loaders = self._fresh_stream_case(701)
                with patch.object(entry, "collect_inputs", return_value=({}, {})), \
                     patch.object(entry, "collect_fisher", return_value={}), \
                     patch.object(entry, "allocate_ranks", return_value={}), \
                     patch.object(entry, "compress", side_effect=interrupted_compress):
                    with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                        entry.main(model, loaders, trainer=SimpleNamespace(model=model), **options)
                checkpoint = root / "out/compressing_model.pt"
                saved = checkpoint.read_bytes()
                if changed == "run_provenance":
                    options["run_provenance"] = {"checkpoint": "checkpoint-b", "data": "data-a"}
                else:
                    # A valid new statistics file at the old path must not be
                    # combined silently with a partially compressed checkpoint.
                    torch.save({"fisher_mat": {}, "vicon_alignment_version": 3,
                                "replacement": torch.zeros(7)}, root / "fisher.pt")
                model, loaders = self._fresh_stream_case(701)
                with patch.object(entry, "compress") as compress, \
                     self.assertRaisesRegex(ValueError, changed):
                    entry.main(model, loaders, trainer=SimpleNamespace(model=model), **options)
                compress.assert_not_called()
                self.assertEqual(checkpoint.read_bytes(), saved)

    def test_cli_all_uses_separate_stage_processes_and_scopes_overwrite(self):
        entry = importlib.import_module("safe_svd.main")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            output.mkdir()
            (output / "compressed_model.pth").write_bytes(b"previous output")
            args = ["--model", "vicon", "--DEV", "cpu", "--ckpt_path", str(root / "model.pth"),
                    "--data_root", str(root / "data"), "--save_path", str(output),
                    "--profile_cache", str(root / "input.pt"), "--fisher_cache", str(root / "fisher.pt"),
                    "--rank_cache", str(root / "rank.json"), "--pipeline_stage", "all", "--overwrite_cache"]

            def run_child(command, **kwargs):
                if command[-1] == "rank_selection" or "rank_selection" in command:
                    (root / "rank.json").write_text("{}")
                    torch.save({"profile_mat": {}, "sample_counts": {}, "vicon_alignment_version": 3},
                               root / "input.pt")
                    torch.save({"fisher_mat": {}, "vicon_alignment_version": 3}, root / "fisher.pt")
                return subprocess.CompletedProcess(command, 0)

            with patch("subprocess.run", side_effect=run_child) as run, \
                 patch.object(entry, "load_model_and_data") as load:
                self.assertEqual(entry.cli_main(args), 0)
            load.assert_not_called()
            self.assertEqual(run.call_count, 2)
            commands = [call.args[0] for call in run.call_args_list]
            for command, expected_stage in zip(commands, ("rank_selection", "compression")):
                self.assertEqual(command[0], sys.executable)
                self.assertIn("safe_svd.main", command)
                stage = [command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--pipeline_stage"]
                self.assertEqual(stage[-1], expected_stage)
                self.assertIn(str(root / "model.pth"), command)
                self.assertIn(str(root / "data"), command)
            self.assertIn("--overwrite_cache", commands[0])
            self.assertNotIn("--overwrite_cache", commands[1])
            self.assertIn("--_overwrite_output", commands[1])

    def test_cli_failed_rank_process_does_not_start_compression(self):
        entry = importlib.import_module("safe_svd.main")
        with tempfile.TemporaryDirectory() as directory:
            args = ["--model", "vicon", "--DEV", "cpu", "--save_path", str(Path(directory) / "out")]
            with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 7)) as run, \
                 patch.object(entry, "load_model_and_data") as load:
                self.assertEqual(entry.cli_main(args), 7)
            run.assert_called_once()
            load.assert_not_called()

    def test_public_compressor_delegates_to_unified_comb_even_when_alpha_zero(self):
        from safesvd.adapters.vicon_runtime import ViconBundle, ViconOptions, run_vicon
        model, data, route, _ = make_case("vicon")
        config = SimpleNamespace(datasets={}, dataset_workers=0, train_seed=42)
        bundle = ViconBundle(Path("VICON"), config, route["trainer"],
                             lambda *_: {name: loader.dataset for name, loader in data.items()},
                             Path("data"), torch.device("cpu"))
        options = ViconOptions(Path("VICON"), Path("model"), Path("data"), Path("out"),
                               Path("input.pt"), Path("fisher.pt"), Path("rank.json"),
                               .8, "legacy", 4, 2026, 0., "cpu", batch_size=2)
        with patch("safesvd.adapters.vicon_runtime.load_vicon_bundle", return_value=bundle), \
             patch("safesvd.adapters.vicon_runtime._validate_options"), \
             patch("safe_svd.main.main") as run:
            self.assertEqual(run_vicon(options), Path("out/compressed_model.pth"))
        self.assertEqual(run.call_args.kwargs["compress_mode"], "comb")
        self.assertEqual(run.call_args.kwargs["max_calib_batches"], 2)
        self.assertEqual(run.call_args.kwargs["train_seed"], 42)
        self.assertEqual(run.call_args.kwargs["rank_selector"], "legacy")

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Original checkout required")
    def test_legacy_rank_selection_matches_original_vicon(self):
        from safe_svd.rank import allocate_ranks
        root = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]).parent / "vicon"
        original = source_function(root / "MySVD/helper/rank_selector.py", "get_global_greedy_rank_dict",
                                   {"torch": torch, "nn": nn})
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(7, 13), nn.GELU(), nn.Linear(13, 9))
        left, right = {}, {}
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                l = torch.diag(torch.linspace(.5, 2., layer.out_features))
                r = torch.diag(torch.linspace(.8, 3., layer.in_features))
                left[name], right[name] = (l, torch.linalg.inv(l)), (r, torch.linalg.inv(r))
        for ratio in (.05, .2, .5, .8, 1.):
            expected = original(SimpleNamespace(model=model), left, right, ratio, "cpu")
            actual = allocate_ranks(model, left, right, ratio, "cpu", selector="legacy")
            self.assertEqual(actual, expected)

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Original checkout required")
    def test_cropped_attention_matches_native_and_research_forward(self):
        from test_vendored_vicon import ICON_CROPPED, model_config
        from safe_svd.helper.collector.vicon_attention import convert_attn_to_linear as unified
        from safesvd.adapters.vicon_attention import convert_attn_to_linear as public
        from safesvd.adapters.vicon_attention import disable_transformer_fast_path
        root = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]).parent / "vicon"
        spec = importlib.util.spec_from_file_location("reference_attention", root / "MySVD/convert_attn.py")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        disable = source_function(root / "evaluation/eval_comp.py", "disable_fast_path",
                                  {"nn": nn, "types": __import__("types")})
        torch.manual_seed(42)
        native = ICON_CROPPED(model_config()).eval()
        data = (torch.randn(1, 2, 7, 4, 4), torch.randn(1, 2, 7, 4, 4), torch.ones(1, 7))
        with torch.inference_mode():
            expected = native(data)
            for convert, slow in ((reference.convert_attn_to_linear, disable),
                                  (unified, lambda model: model), (public, disable_transformer_fast_path)):
                model = copy.deepcopy(native)
                convert(model)
                slow(model)
                torch.testing.assert_close(model(data), expected, atol=2e-6, rtol=2e-5)

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Original checkout required")
    def test_evaluation_loss_matches_original_script(self):
        from test_evaluation_alignment import TimePairDataset, TinyTrainer
        from safesvd.adapters.vicon_runtime import ViconBundle, _vicon_batch_factory, _evaluate_vicon_model
        root = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]).parent / "vicon"
        original = source_function(root / "evaluation/eval_comp.py", "evaluate_loss",
                                   {"torch": torch, "tqdm": lambda values, **kwargs: values})
        trainer = TinyTrainer()
        expected, relative = original(trainer, {"NS2D": DataLoader(TimePairDataset("NS2D", []), batch_size=1)})
        bundle = ViconBundle(Path("VICON"), SimpleNamespace(datasets={}, dataset_workers=0), trainer,
                             lambda *_: {"NS2D": TimePairDataset("NS2D", [])}, Path("data"), torch.device("cpu"))
        factory = _vicon_batch_factory(bundle, split="test", batch_size=1, calibration_size=None,
                                       seed=84, random_subset=False)
        actual = _evaluate_vicon_model(bundle, trainer.model, factory)
        self.assertEqual(actual["losses"], expected)
        self.assertEqual(actual["relative_losses"], relative)

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Original checkout required")
    def test_sequential_compression_matches_original_mysvd(self):
        from safe_svd.compress import compress
        root = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]).parent / "vicon/MySVD"
        namespace = {"torch": torch, "nn": nn, "copy": copy, "os": os, "json": json,
                     "tqdm": lambda values: values,
                     "StopForwardException": type("StopForwardException", (Exception,), {})}
        source_function(root / "helper/whitener.py", "svd_whitener", namespace)
        source_function(root / "helper/input_collector.py", "collect_covariances_vicon", namespace)
        original = source_function(root / "compress/comb_compress.py", "mysvd_comb", namespace)
        for alpha, trace, use_fisher in product((0., .7, 1.), (False, True), (False, True)):
            with self.subTest(alpha=alpha, trace=trace, use_fisher=use_fisher), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(42)
                model, data, options, args = make_case("vicon")
                ranks = {"first": 1, "second": 2}
                # Nontrivial left transform exercises float32 inverse multiplication;
                # None exercises the original no-Fisher identity path.
                fisher = {} if use_fisher else None
                if use_fisher:
                    for name, layer in model.named_modules():
                        if isinstance(layer, nn.Linear):
                            left = torch.diag(torch.linspace(.7, 1.8, layer.out_features))
                            fisher[name] = (left, torch.linalg.inv(left))
                path = Path(directory) / "ranks.json"
                path.write_text(json.dumps(ranks))
                expected = original(options["trainer"], data, dev="cpu", ranks_json_path=str(path),
                                    Ldict=fisher, max_calib_batches=2, alpha=alpha, use_trace_align=trace)
                options["trainer"].model = model
                actual = compress(model, data, ranks, model_type="vicon", device="cpu",
                                  route_options=options, left_metrics=fisher, alpha=alpha,
                                  trace_balance=trace)
                # SVD signs need not agree; compare effective operators and predictions.
                for name in ranks:
                    a, b = actual.get_submodule(name), expected.get_submodule(name)
                    torch.testing.assert_close(a[1].weight @ a[0].weight,
                                               b[1].weight @ b[0].weight, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(actual(*args), expected(*args), atol=2e-6, rtol=2e-5)
