"""Derivative-loss and raw-Gram compression regressions for MySVD_fast.

Optional source oracle: SAFE_SVD_REFERENCE_ROOT=.../physic_model_comp/poseidon.
Only tests read the reference; production code is independent of it.
"""

from __future__ import annotations

import __future__
import ast
import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from test_unified_profiling import make_case, source_function
from test_unified_capture_routes import _load_source_helpers, _toy_rollout


class PoseidonMySVDFastTests(unittest.TestCase):
    def test_constant_offset_has_no_derivative_loss(self):
        from safe_svd.helper.collector.poseidon.fisher_loss import sobolev_loss_calc_denorm
        for order in (0, 1, 2):
            prediction = torch.full((1, 4, 4, 4), 0.25, requires_grad=True)
            physical = sobolev_loss_calc_denorm(
                "fluids.incompressible.PiecewiseConstants", prediction, torch.zeros_like(prediction),
                order, 0.1, F.l1_loss, torch.device("cpu"),
                {"mean": [0.] * 4, "std": [1.] * 4}, balance_ratio=10,
            )
            self.assertEqual(float(physical), 0.)
            base = prediction.square().mean()
            actual_gradient = torch.autograd.grad(base + physical, prediction, retain_graph=True)[0]
            expected_gradient = torch.autograd.grad(base, prediction)[0]
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    def test_default_compression_decomposes_raw_grams(self):
        import importlib
        module = importlib.import_module("safe_svd.compress")
        model = nn.Sequential(nn.Linear(4, 6, bias=False)).double()
        torch.manual_seed(911)
        h = torch.randn(11, 4).double()
        hp = torch.randn(11, 4).double()
        def capture(_context, _name):
            yield "poseidon", h[:3], hp[:3]
            yield "poseidon", h[3:], hp[3:]
        with patch.object(module, "right_whitener", wraps=module.right_whitener) as whitening:
            module.compress(model, None, {"0": 1}, model_type="poseidon", device="cpu",
                            capture_fn=capture, progress=False)
        expected = 0.3 * (h.T @ h) + 0.7 * (hp.T @ hp)
        torch.testing.assert_close(whitening.call_args.args[0], expected)
        self.assertEqual(whitening.call_args.kwargs["method"], "spectral")
        self.assertEqual(whitening.call_args.kwargs["eps_mode"], "absolute")
        self.assertEqual(whitening.call_args.kwargs["eps"], 1e-6)

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Reference source not requested")
    def test_loss_values_and_gradients_match_mysvd_fast_source(self):
        from safe_svd.helper.collector.poseidon.fisher_loss import sobolev_loss_calc_denorm
        path = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]) / "MySVD_fast/helper/fisher_collector.py"
        reference = source_function(path, "sobolev_loss_calc_denorm", {"torch": torch})
        torch.manual_seed(913)
        for dataset in ("fluids.incompressible.PiecewiseConstants", "wave.Gaussians"):
            for loss_fn in (F.l1_loss, F.mse_loss):
                for order in (0, 1, 2):
                    actual_pred = torch.randn(2, 4, 6, 5, requires_grad=True)
                    expected_pred = actual_pred.detach().clone().requires_grad_()
                    labels = torch.randn_like(actual_pred)
                    args = (labels, order, 0.1, loss_fn, torch.device("cpu"),
                            {"mean": [0.1] * 4, "std": [0.5, 1., 2., 3.]})
                    expected = reference(dataset, expected_pred, *args, balance_ratio=10, decay_alpha=0.7)
                    actual = sobolev_loss_calc_denorm(dataset, actual_pred, *args, balance_ratio=10, decay_alpha=0.7)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    # At order zero both physical terms are constant zeros.
                    torch.testing.assert_close(
                        torch.autograd.grad(actual + actual_pred.square().mean(), actual_pred)[0],
                        torch.autograd.grad(expected + expected_pred.square().mean(), expected_pred)[0],
                        rtol=0, atol=0,
                    )

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Reference source not requested")
    def test_complete_compression_matches_mysvd_fast_source(self):
        from safe_svd.compress import compress
        from safe_svd.helper.perf_logger import make_perf_logger
        root = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"])
        helpers = _load_source_helpers(root, package="MySVD_fast")
        namespace = dict(helpers, torch=torch, nn=nn, copy=copy, os=os, json=json,
                         Subset=torch.utils.data.Subset, tqdm=lambda values, **kwargs: values,
                         make_perf_logger=make_perf_logger)
        source_function(root / "MySVD_fast/helper/whitener.py", "svd_whitener", namespace)
        path = root / "MySVD_fast/compression_method/compress_comb_input.py"
        definitions = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        torch.manual_seed(917)
        body, data, _options, args = make_case("poseidon")
        # Original set_module_by_name expects nested names.
        class Model(nn.Module):
            def __init__(self, body):
                super().__init__()
                self.body, self.config = body, body.config
            def forward(self, **inputs):
                return self.body(**inputs)
        model = Model(body)
        trainer = _options["trainer"]
        trainer.model, trainer.rollouts = model, 0
        ranks = {"body.first": 1, "body.second": 1}
        left = {}
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                factor = torch.diag(torch.linspace(0.5, 2., layer.out_features))
                left[name] = (factor, factor.inverse())
        original = copy.deepcopy(model)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            rank_path = Path(temporary) / "rank.json"
            rank_path.write_text(json.dumps(ranks))
            expected = namespace["mysvd_comb"](trainer, "cpu", str(rank_path), data,
                                                Ldict=left, alpha=0.7)
        trainer.model = original
        actual = compress(original, data, ranks, model_type="poseidon", device="cpu",
                          left_metrics=left, route_options={"trainer": trainer, "rollout_fn": _toy_rollout},
                          progress=False)
        for name in ranks:
            for index in (0, 1):
                torch.testing.assert_close(actual.get_submodule(name)[index].weight,
                                           expected.get_submodule(name)[index].weight, rtol=2e-5, atol=2e-6)
        inputs = {"pixel_values": args[0], "labels": args[1]}
        torch.testing.assert_close(actual(**inputs).output, expected(**inputs).output, rtol=2e-5, atol=2e-6)
