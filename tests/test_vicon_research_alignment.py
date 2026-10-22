"""VICON cold/warm cache and original MySVD numerical regressions."""

import copy
import importlib
import io
import json
import os
import random
import tempfile
import unittest
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

    def test_missing_stream_state_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "Recompute"):
            vicon_rng.restore({}, None)

    def test_cache_rejects_changed_batch_size(self):
        state = vicon_rng.capture({"NS2D": DataLoader(Stream(), batch_size=1)})
        with self.assertRaisesRegex(ValueError, "batch size/worker"):
            vicon_rng.restore({"NS2D": DataLoader(Stream(), batch_size=2)}, state)

    def test_partial_and_full_cache_hits_replay_stage_inputs(self):
        entry = importlib.import_module("safe_svd.main")
        seen = []
        def consume(loaders):
            return torch.cat(list(loaders["NS2D"]))
        def profile(model, loaders, **kwargs):
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            consume(loaders)
            return {}, {}
        def fisher(model, loaders, **kwargs):
            seen.append(("fisher", consume(loaders)))
            return {}
        def compress(model, loaders, *args, **kwargs):
            seen.append(("compress", consume(loaders)))
            return model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(entry, "collect_inputs", side_effect=profile), \
                 patch.object(entry, "collect_fisher", side_effect=fisher), \
                 patch.object(entry, "allocate_ranks", return_value={}), \
                 patch.object(entry, "compress", side_effect=compress):
                for fisher_path in (root / "first.pt", root / "second.pt", root / "first.pt"):
                    model = nn.Linear(2, 2)
                    entry.main(model, {"NS2D": DataLoader(Stream(), batch_size=1)},
                               trainer=SimpleNamespace(model=model), model_type="vicon", rank_ratio=.5,
                               device="cpu", profile_cache=root / "input.pt", fisher_cache=fisher_path,
                               rank_cache=root / "ranks.json", verbose=False)
        # cold, input cache only, both caches: Fisher and compression see identical streams.
        self.assertEqual([name for name, _ in seen], ["fisher", "compress", "fisher", "compress", "compress"])
        torch.testing.assert_close(seen[0][1], seen[2][1], rtol=0, atol=0)
        for index in (3, 4):
            torch.testing.assert_close(seen[1][1], seen[index][1], rtol=0, atol=0)

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
        for alpha in (0., .7, 1.):
            for trace in (False, True):
                with self.subTest(alpha=alpha, trace=trace), tempfile.TemporaryDirectory() as directory:
                    torch.manual_seed(42)
                    model, data, options, args = make_case("vicon")
                    ranks = {"first": 1, "second": 2}
                    # Nontrivial left transform exercises float32 inverse multiplication.
                    fisher = {}
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
