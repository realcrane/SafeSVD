"""Numerical comparisons to the reference SAFE_SVD, only when requested.

Set SAFE_SVD_REFERENCE_ROOT to the reference poseidon/ directory. Reference
functions run unchanged on CPU toy data; production has no reference imports.
"""

import __future__
import ast
import copy
import heapq
import io
import itertools
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

import test_unified_capture_routes as capture_reference
from test_unified_capture_routes import _load_source_helpers, _toy_rollout
from test_unified_profiling import make_case, source_function


@unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Reference source not requested")
class PoseidonReferenceConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]) / "SAFE_SVD"
        cls.whiteners = {"torch": torch}
        for name in ("fisher_cholesky_whitener", "resolve_whitening_eps", "svd_whitener"):
            source_function(cls.root / "helper/whitener.py", name, cls.whiteners)
        cls.rank_namespace = {"torch": torch, "nn": torch.nn, "heapq": heapq,
                              "itertools": itertools, "tqdm": lambda values, **kwargs: values}
        cls.reference_rank = staticmethod(source_function(cls.root / "rank_selector.py",
                                                         "_get_global_greedy_rank_dict", cls.rank_namespace))

    def test_safe_matches_reference_and_legacy_is_different(self):
        from safe_svd.rank import allocate_ranks
        model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False), torch.nn.Linear(8, 8, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.diag(torch.arange(1, 9).float() * 10))
            model[1].weight.copy_(torch.diag(torch.arange(1, 9).float()))
        profiles = {str(i): (torch.eye(8), torch.eye(8)) for i in range(2)}
        with redirect_stdout(io.StringIO()):
            expected = self.reference_rank(SimpleNamespace(model=model), None, profiles, 0.5, dev="cpu")
        self.assertEqual(expected, {"0": 3, "1": 1})
        self.assertEqual(allocate_ranks(model, None, profiles, 0.5, "cpu", selector="safe"), expected)
        self.assertNotEqual(allocate_ranks(model, None, profiles, 0.5, "cpu", selector="legacy"), expected)

    def test_origin_compression_whitener_matches_reference_precision(self):
        from safe_svd.helper.whitener import right_whitener
        torch.manual_seed(719)
        values = torch.randn(19, 8).double()
        covariance = values.T @ values / len(values)
        left, inverse = self.whiteners["fisher_cholesky_whitener"](covariance, 1)
        actual = right_whitener(covariance, method="fisher_cholesky", eps=1e-6,
                                eps_mode="absolute", sample_count=len(values))
        # The reference constructs/inverts its factor in float32, then casts
        # to compression compute_dtype. Exact precision policy matters here.
        torch.testing.assert_close(actual[0], left.T.double(), rtol=0, atol=0)
        torch.testing.assert_close(actual[1], inverse.double(), rtol=0, atol=0)

    def test_complete_compression_matches_reference_on_fixed_ranks_and_data(self):
        from safe_svd.compress import compress
        from safe_svd.helper.perf_logger import make_perf_logger
        helpers = _load_source_helpers(self.root.parent)
        namespace = dict(helpers, torch=torch, nn=torch.nn, copy=copy, os=os, Path=Path,
                         Subset=torch.utils.data.Subset, tqdm=lambda values, **kwargs: values,
                         make_perf_logger=make_perf_logger, **{k: v for k, v in self.whiteners.items() if k != "torch"})
        tree = ast.parse((self.root / "compress.py").read_text())
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(self.root / "compress.py"),
                     "exec", flags=__future__.annotations.compiler_flag), namespace)
        for whitening in ("origin", "new"):
            torch.manual_seed(818)
            model, data, options, args = make_case("poseidon")
            trainer = options["trainer"]
            trainer.rollouts = 0
            teacher = copy.deepcopy(model)
            with redirect_stdout(io.StringIO()):
                expected = namespace["compress"](trainer, data, {"first": 1, "second": 1},
                    device_name="cpu", whitening_mode=whitening, eps_whiten=1e-6, low_gpu=False)
            local_trainer = copy.deepcopy(trainer)
            local_trainer.model = teacher
            actual = compress(teacher, data, {"first": 1, "second": 1}, model_type="poseidon", device="cpu",
                              whitening="fisher_cholesky" if whitening == "origin" else "spectral",
                              eps_mode="absolute",
                              route_options={"trainer": local_trainer, "rollout_fn": _toy_rollout,
                                             "capture_batch_size": 64}, progress=False)
            with self.subTest(whitening=whitening):
                torch.testing.assert_close(actual(*args).output, expected(*args).output, rtol=2e-5, atol=2e-6)

    def test_full_attention_cache_reconstruction_matches_reference(self):
        from safe_svd.compress import compress
        from safe_svd.helper.perf_logger import make_perf_logger
        helpers = _load_source_helpers(self.root.parent)
        namespace = dict(helpers, torch=torch, nn=torch.nn, copy=copy, os=os, Path=Path,
                         Subset=torch.utils.data.Subset, tqdm=lambda values, **kwargs: values,
                         make_perf_logger=make_perf_logger, **{k: v for k, v in self.whiteners.items() if k != "torch"})
        definitions = [node for node in ast.parse((self.root / "compress.py").read_text()).body
                       if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(self.root / "compress.py"),
                     "exec", flags=__future__.annotations.compiler_flag), namespace)
        for skipped in (None, "query", "value"):
            torch.manual_seed(317)
            ctx = capture_reference.PoseidonRouteTests().attention_context()
            original = copy.deepcopy(ctx.teacher)
            prefix = "encoder.layers.0.blocks.0.attention"
            ranks = {"pre": 1, **{f"{prefix}.self.{name}": None if name == skipped else 1
                                   for name in ("query", "key", "value")}, f"{prefix}.output.dense": 1}
            with redirect_stdout(io.StringIO()):
                expected = namespace["compress"](ctx.options["trainer"], ctx.calibration_data, ranks,
                                                  device_name="cpu", whitening_mode="origin", eps_whiten=1e-6)
            trainer = copy.deepcopy(ctx.options["trainer"])
            trainer.model = original
            actual = compress(original, ctx.calibration_data, ranks, model_type="poseidon", device="cpu",
                              whitening="fisher_cholesky", route_options={"trainer": trainer,
                              "rollout_fn": _toy_rollout, "capture_batch_size": 64}, progress=False)
            with self.subTest(skipped=skipped), torch.no_grad():
                for example in torch.utils.data.DataLoader(ctx.calibration_data, batch_size=1):
                    torch.testing.assert_close(actual(**example), expected(**example), rtol=2e-5, atol=2e-6)
