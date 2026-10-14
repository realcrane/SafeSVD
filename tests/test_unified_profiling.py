"""CPU profiling/Fisher/pipeline checks with optional original-code comparisons."""

from __future__ import annotations

import ast
import __future__
import copy
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None


def make_case(model_type):
    class MPPDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.values, self.targets = torch.randn(6, 2, 4, 4, 5), torch.randn(6, 4, 4, 5) + 0.5
            self.title = "toy"
        def get_name(self):
            return "swe"
        def __len__(self):
            return len(self.values)
        def __getitem__(self, index):
            return self.values[index], torch.zeros(1), self.targets[index]
    class MPPModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first, self.act, self.second = nn.Linear(4, 8), nn.ReLU(inplace=True), nn.Linear(8, 4)
        def forward(self, inp, labels, bcs):
            x = inp[-1].permute(0, 2, 3, 1)
            return self.second(self.act(self.first(x))).permute(0, 3, 1, 2)
    class ViconModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first, self.act, self.second = nn.Linear(5, 8), nn.ReLU(inplace=True), nn.Linear(8, 5)
        def forward(self, data):
            init, _end, _mask = data
            x = init.permute(0, 1, 3, 4, 2)
            return self.second(self.act(self.first(x))).permute(0, 1, 4, 2, 3)
    class ViconTrainer:
        def __init__(self, model):
            self.model = model
            self.loss_cfg = SimpleNamespace(min_ex=1)
        def _move_to_device(self, pairs):
            return [values.to(next(self.model.parameters()).device) for values in pairs]
        def _data_preprocess(self, pairs):
            init, end, mask = pairs
            return (init * 0.5, end * 0.5, mask[:, None, :, None, None]), 0., 2.
        def _model_forward(self, data):
            return self.model(data)
    class PoseidonDataset(torch.utils.data.Dataset):
        constants = {"mean_solution": [0.1] * 4, "std_solution": [1.1] * 4}
        transpose = True
        def __init__(self):
            self.values, self.targets = torch.randn(6, 4, 4, 5), torch.randn(6, 4, 4, 5)
        def __len__(self):
            return len(self.values)
        def __getitem__(self, index):
            return {"pixel_values": self.values[index], "labels": self.targets[index]}
    class PoseidonModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(name_or_path="toy")
            self.first, self.act, self.second = nn.Linear(4, 8), nn.ReLU(inplace=True), nn.Linear(8, 4)
        def forward(self, pixel_values, labels):
            prediction = self.second(self.act(self.first(pixel_values.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
            return SimpleNamespace(output=prediction, loss=(prediction - labels).square().mean())
    class PoseidonTrainer:
        def __init__(self, model):
            self.model, self.ar_steps, self.output_all_steps = model, 3, True
        def set_ar_steps(self, steps, output_all_steps=False):
            self.ar_steps, self.output_all_steps = steps, output_all_steps
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            return model(**inputs)
    def rollout(trainer, dataset, model_path=""):
        for batch in torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False):
            trainer.prediction_step(trainer.model, batch, False)
    if model_type == "mpp":
        subset = MPPDataset()
        data = SimpleNamespace(sub_dsets=[SimpleNamespace(get_per_file_dsets=lambda: [subset])],
                               subset_dict={"swe": [0, 1]})
        options = {"params": SimpleNamespace(batch_size=2, num_data_workers=0),
                   "batches_per_dset": 2, "calibration_seed": 17, "balance_ratio": 0.5}
        model = MPPModel()
        args = (subset.values[:2].permute(1, 0, 2, 3, 4), torch.tensor([[0, 1], [0, 1]]), torch.zeros(2, 1))
    elif model_type == "vicon":
        def loader():
            raw = []
            for index in range(4):
                init, end = torch.randn(3, 5, 4, 5), torch.randn(3, 5, 4, 5)
                end[:, -1] = 0
                end[:, -1, 0] = 1
                raw.append((index, (init, end, torch.tensor([1., 1., 0., 1., 1.])), 0., 1., 1.))
            return torch.utils.data.DataLoader(raw, batch_size=2, shuffle=False)
        model = ViconModel()
        trainer = ViconTrainer(model)
        data = {"NS2D": loader(), "EULER2D": loader()}
        options = {"trainer": trainer, "max_calib_batches": 2, "train_seed": 41}
        args = (trainer._data_preprocess(trainer._move_to_device(next(iter(data["NS2D"]))[1]))[0],)
    else:
        model, data = PoseidonModel(), PoseidonDataset()
        options = {"trainer": PoseidonTrainer(model), "rollout_fn": rollout,
                   "dataset_name": "wave", "order_control": 1, "fisher_batch_size": 2,
                   "capture_batch_size": 2, "offload_cache": True}
        args = (data.values[:2], data.targets[:2])
    model.eval()
    return model, data, options, args


def source_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


@unittest.skipIf(torch is None, "PyTorch is required")
class UnifiedProfilingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def test_three_default_routes_complete_input_fisher_rank_compress(self):
        from safe_svd.helper.collector import collect_inputs, collect_fisher
        from safe_svd.rank import allocate_ranks
        from safe_svd.compress import compress
        for route in ("mpp", "vicon", "poseidon"):
            with self.subTest(route=route):
                model, data, options, args = make_case(route)
                before = copy.deepcopy(model.state_dict())
                inputs, sample_counts, input_cov = collect_inputs(
                    model, data, model_type=route, device="cpu", route_options=options, return_statistics=True)
                left, gradient_counts, fisher_cov = collect_fisher(
                    model, data, model_type=route, device="cpu", route_options=options, return_statistics=True)
                self.assertEqual(set(inputs), {"first", "second"})
                self.assertEqual(set(left), set(inputs))
                self.assertTrue(all(count > 0 for count in sample_counts.values()))
                self.assertTrue(all(count > 0 for count in gradient_counts.values()))
                for name in inputs:
                    self.assertTrue(torch.isfinite(input_cov[name]).all())
                    self.assertTrue(torch.isfinite(fisher_cov[name]).all())
                ranks = allocate_ranks(model, left, inputs, 0.5, "cpu", selector="safe")
                student = compress(model, data, ranks, model_type=route, device="cpu", route_options=options, left_metrics=left)
                output = student(*args)
                output = output.output if route == "poseidon" else output
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(tuple(output.shape), tuple(args[0].shape) if route == "poseidon" else
                                 ((2, 4, 4, 5) if route == "mpp" else (2, 3, 5, 4, 5)))
                actual = sum(layer.weight.numel() if ranks[name] is None else
                             ranks[name] * (layer.in_features + layer.out_features)
                             for name, layer in model.named_modules() if isinstance(layer, nn.Linear))
                original = sum(layer.weight.numel() for layer in model.modules() if isinstance(layer, nn.Linear))
                self.assertLessEqual(actual, int(original * 0.5))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name])
                self.assertTrue(model.act.inplace)
                self.assertFalse(model.training)
                self.assertTrue(all(not layer._forward_hooks and not layer._backward_hooks for layer in model.modules()))
                if route != "mpp":
                    self.assertIs(options["trainer"].model, model)

    def test_raw_input_covariances_match_full_forward_capture(self):
        from safe_svd.helper.collector import collect_inputs
        from safe_svd.helper.collector.profiling_routes import run_input_profile
        from safe_svd.helper.collector.collector import merge_covariances
        for route in ("mpp", "vicon", "poseidon"):
            model, data, options, _args = make_case(route)
            sums, counts, handles = {}, {}, []
            category = "default"
            def set_category(value):
                nonlocal category
                category = value
            for name, layer in model.named_modules():
                if not isinstance(layer, nn.Linear):
                    continue
                def hook(_layer, args, name=name):
                    rows = args[0].detach().float().reshape(-1, args[0].shape[-1])
                    sums.setdefault(name, {}).setdefault(category, []).append(rows)
                handles.append(layer.register_forward_pre_hook(hook))
            try:
                with torch.no_grad():
                    run_input_profile(model, data, route, torch.device("cpu"), options, set_category)
            finally:
                for handle in handles:
                    handle.remove()
            _, actual_counts, covariances = collect_inputs(model, data, model_type=route, device="cpu",
                route_options=options, trace_balance=False, return_statistics=True)
            for name, categories in sums.items():
                gram = {cat: sum(rows.T @ rows for rows in chunks) for cat, chunks in categories.items()}
                per_category_counts = {cat: sum(len(rows) for rows in chunks) for cat, chunks in categories.items()}
                expected = merge_covariances(gram, per_category_counts,
                                             aggregation="category_mean" if route == "vicon" else "sample_mean")
                torch.testing.assert_close(covariances[name], expected, atol=1e-6, rtol=1e-5)
                self.assertEqual(actual_counts[name], sum(per_category_counts.values()))

    def test_fisher_matches_output_gradients_and_care_identity(self):
        from safe_svd.helper.collector import collect_fisher
        model, data, options, _ = make_case("mpp")
        expected_outputs = {name: [] for name in ("first", "second")}
        def loss(network, tagged):
            handles = []
            for name, layer in network.named_modules():
                if isinstance(layer, nn.Linear):
                    def save_output(_module, _inputs, output, name=name):
                        output.retain_grad()
                        expected_outputs[name].append(output)
                    handles.append(layer.register_forward_hook(save_output))
            try:
                inp, labels, bcs, _target = tagged.payload
                value = network(inp, labels, bcs).square().mean()
                return value
            finally:
                for handle in handles:
                    handle.remove()
        # Inplace is disabled by the collector before this callback executes.
        with torch.no_grad():
            metrics, counts, covariances = collect_fisher(model, data, model_type="mpp", device="cpu",
                route_options=options, loss_fn=loss, trace_balance=False, care_identity=True, return_statistics=True)
        for name, outputs in expected_outputs.items():
            rows = torch.cat([output.grad.detach().reshape(-1, output.shape[-1]) for output in outputs])
            reference = rows.T @ rows / len(rows)
            torch.testing.assert_close(covariances[name], reference.double(), atol=1e-6, rtol=1e-5)
            self.assertEqual(counts[name], len(rows))
            left, inverse = metrics[name]
            regularized = reference / reference.diagonal().mean() + torch.eye(reference.shape[0])
            torch.testing.assert_close(left.T @ left, regularized, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(inverse @ left, torch.eye(len(left)), atol=1e-5, rtol=1e-5)

    def test_trace_alignment_is_confined_to_groups(self):
        from safe_svd.helper.collector.collector import merge_covariances
        sums = {"a": torch.eye(2), "b": torch.eye(2) * 4, "c": torch.eye(2) * 9}
        merged = merge_covariances(sums, {name: 1 for name in sums}, trace_balance=True,
                                   trace_groups={"a": 1, "b": 1, "c": 2})
        torch.testing.assert_close(merged, torch.eye(2).double() * (4 + 4 + 9) / 3)

    def test_untouched_layers_and_sum_equivalent_regularizer(self):
        from safe_svd.helper.collector import collect_inputs, collect_fisher
        model, data, options, _ = make_case("mpp")
        model.unused = nn.Linear(3, 3)
        metrics, counts, covariance = collect_inputs(model, data, model_type="mpp", device="cpu",
            route_options=options, trace_balance=False, eps=0.01, eps_mode="sum_equivalent", return_statistics=True)
        self.assertEqual(counts["unused"], 0)
        for name in ("first", "second"):
            right, _inverse = metrics[name]
            expected = covariance[name] + torch.eye(len(right)) * (0.01 / counts[name])
            torch.testing.assert_close((right @ right.T).double(), expected, atol=1e-6, rtol=1e-5)
        fisher, counts, _covariance = collect_fisher(model, data, model_type="mpp", device="cpu",
                                                   route_options=options, return_statistics=True)
        self.assertNotIn("unused", fisher)
        self.assertEqual(counts["unused"], 0)
        with self.assertRaisesRegex(ValueError, "No input samples for unused"):
            collect_inputs(model, data, model_type="mpp", device="cpu", route_options=options, missing="raise")
        self.assertTrue(all(not layer._forward_hooks for layer in model.modules()))

    def test_failure_restores_hooks_modes_inplace_and_trainer(self):
        from safe_svd.helper.collector import collect_inputs, collect_fisher
        model, data, options, _ = make_case("poseidon")
        model.train()
        model.first.eval()
        original_modes = [module.training for module in model.modules()]
        def fail(*args, **kwargs):
            raise RuntimeError("synthetic failure")
        broken_options = {**options, "rollout_fn": fail}
        for function, kwargs in ((collect_inputs, {"route_options": broken_options}),
                                  (collect_fisher, {"route_options": options, "loss_fn": fail})):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                function(model, data, model_type="poseidon", device="cpu", **kwargs)
            self.assertEqual([module.training for module in model.modules()], original_modes)
            self.assertTrue(model.act.inplace)
            self.assertIs(options["trainer"].model, model)
            self.assertTrue(all(not layer._forward_hooks and not layer._backward_hooks for layer in model.modules()))

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Original checkout required")
    def test_input_factors_match_original_profilers(self):
        from safe_svd.helper.collector import collect_inputs
        root = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]).parent
        for route in ("mpp", "vicon", "poseidon"):
            for whitening in (("spectral", "fisher_cholesky") if route == "poseidon" else ("spectral",)):
                with self.subTest(route=route, whitening=whitening):
                    model, data, options, _ = make_case(route)
                    actual, counts = collect_inputs(model, data, model_type=route, device="cpu",
                                                    route_options=options, whitening=whitening, eps_mode="absolute")
                    namespace = {"torch": torch, "nn": nn, "tqdm": lambda values, **kwargs: values,
                                 "DataLoader": torch.utils.data.DataLoader}
                    if route == "mpp":
                        namespace["rearrange"] = lambda values, pattern: values.permute(1, 0, 2, 3, 4)
                        path = root / "mpp/MySVD/mysvd.py"
                        source_function(path, "svd_whitener", namespace)
                        source_function(path, "iter_mpp_batches", namespace)
                        original = source_function(path, "profile_input_gram", namespace)
                        reference = original(model, data, options["params"], "cpu", batches_per_dset=2, calibration_seed=17)
                    elif route == "vicon":
                        path = root / "vicon/MySVD/helper/input_collector.py"
                        source_function(root / "vicon/MySVD/helper/whitener.py", "svd_whitener", namespace)
                        source_function(path, "_trace_align_and_merge", namespace)
                        seed = source_function(root / "vicon/VICON/src/utils.py", "set_seed", {
                            "torch": torch, "random": __import__("random"), "np": __import__("numpy")})
                        namespace["utils"] = SimpleNamespace(set_seed=seed)
                        original = source_function(path, "profile_svd_vicon", namespace)
                        reference = original(options["trainer"], data, SimpleNamespace(train_seed=41), "cpu", max_calib_batches=2)
                    else:
                        path = root / "poseidon/SAFE_SVD/helper/whitener.py"
                        for name in ("fisher_cholesky_whitener", "resolve_whitening_eps", "svd_whitener"):
                            source_function(path, name, namespace)
                        namespace["rollout"] = options["rollout_fn"]
                        original = source_function(root / "poseidon/SAFE_SVD/helper/input_collector.py", "profile_inputs", namespace)
                        reference, reference_counts = original(options["trainer"], data, "cpu", return_sample_counts=True,
                                                               whitening_mode="new" if whitening == "spectral" else "origin")
                        self.assertEqual(counts, reference_counts)
                    self.assertEqual(set(actual), set(reference))
                    for name in actual:
                        torch.testing.assert_close(actual[name][0] @ actual[name][0].T,
                                                   reference[name][0] @ reference[name][0].T, atol=1e-6, rtol=1e-5)

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Original checkout required")
    def test_fisher_factors_match_original_collectors(self):
        from safe_svd.helper.collector import collect_fisher
        from safe_svd.helper.collector import mpp_fisher_loss as mpp_loss_module
        from safe_svd.helper.collector import vicon_fisher_loss as vicon_loss_module
        from safe_svd.helper.collector.poseidon.fisher_loss import sobolev_loss_calc_denorm
        from safe_svd.helper.collector.vicon_fisher_forward import _patched_forward
        root = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]).parent
        poseidon_whitener = source_function(root / "poseidon/SAFE_SVD/helper/whitener.py", "fisher_cholesky_whitener", {"torch": torch})
        for route in ("mpp", "vicon", "poseidon"):
            with self.subTest(route=route):
                model, data, options, _ = make_case(route)
                actual = collect_fisher(model, data, model_type=route, device="cpu", route_options=options)
                namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional,
                             "tqdm": lambda values, **kwargs: values}
                if route == "mpp":
                    namespace.update(DataLoader=torch.utils.data.DataLoader, DSET_SOBOLEV_CONFIG=mpp_loss_module.DSET_SOBOLEV_CONFIG,
                                     sobolev_loss_mpp=mpp_loss_module.sobolev_loss_mpp,
                                     rearrange=lambda values, pattern: values.permute(1, 0, 2, 3, 4))
                    original = source_function(root / "mpp/MySVD/mysvd.py", "collect_fisher_mpp", namespace)
                    reference = original(model, data, options["params"], "cpu", batches_per_dset=2, calibration_seed=17)
                elif route == "vicon":
                    namespace.update(gc=__import__("gc"), sobolev_loss_for_fisher=vicon_loss_module.sobolev_loss_for_fisher)
                    original = source_function(root / "vicon/MySVD/helper/fisher_collector.py", "collect_mysvd_fisher_vicon", namespace)
                    modules = {name: types.ModuleType(name) for name in ("MySVD", "MySVD.helper", "MySVD.helper.input_collector")}
                    modules["MySVD.helper.input_collector"]._patched_forward = _patched_forward
                    with patch.dict(sys.modules, modules):
                        reference = original(options["trainer"], data, dev="cpu", max_calib_batches=2)
                else:
                    namespace.update(sobolev_loss_calc_denorm=sobolev_loss_calc_denorm,
                                     fisher_cholesky_whitener=poseidon_whitener)
                    original = source_function(root / "poseidon/MySVD_fast/helper/fisher_collector.py", "collect_fisher_dict_linear_only", namespace)
                    reference = original(options["trainer"], "wave", torch.utils.data.DataLoader(data, batch_size=2),
                                         torch.device("cpu"), order_control=1, constants=data.constants, transpose=True)
                self.assertEqual(set(actual), set(reference))
                for name in actual:
                    # Factor signs/bases can differ; compare the induced metric.
                    torch.testing.assert_close(actual[name][0].T @ actual[name][0],
                                               reference[name][0].T @ reference[name][0], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
