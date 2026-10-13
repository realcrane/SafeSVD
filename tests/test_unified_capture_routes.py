"""Capture regressions, optionally checked against the original SAFE_SVD helpers.

Set SAFE_SVD_REFERENCE_ROOT to physic_model_comp/poseidon to enable the source
comparisons. The normal tests use the local helpers and a CPU toy trainer;
the original helpers are loaded only as optional regression references.
"""

from __future__ import annotations

import copy
import ast
import __future__
import importlib.util
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


@unittest.skipIf(torch is None, "PyTorch is required")
class BatchRouteTests(unittest.TestCase):
    def test_mpp_time_axis_labels_categories_seed_and_current_student(self):
        from safe_svd.compress import CaptureContext, collect_mpp

        class FileDataset(torch.utils.data.Dataset):
            def get_name(self):
                return "incompNS"

            def __len__(self):
                return 7

            def __getitem__(self, index):
                return torch.full((3, 4, 2, 2), float(index)), torch.tensor([index]), torch.zeros(3, 4, 2, 2)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pre = nn.Linear(4, 4, bias=False)
                self.target = nn.Linear(4, 4)
                self.observed = []

            def forward(self, inp, labels, bcs):
                self.observed.append((inp.clone(), labels.clone(), bcs.clone()))
                return self.target(self.pre(inp.permute(0, 1, 3, 4, 2)))

        dataset = SimpleNamespace(
            sub_dsets=[SimpleNamespace(get_per_file_dsets=lambda: [FileDataset()])],
            subset_dict={"incompNS": [0, 2]},
        )
        teacher = Model()
        student = copy.deepcopy(teacher)
        student.pre.weight.data.mul_(0.4)
        ctx = CaptureContext(teacher, student, dataset, torch.device("cpu"), {
            "params": SimpleNamespace(batch_size=2, num_data_workers=0),
            "calibration_seed": 23, "batches_per_dset": 2,
        })
        first = list(collect_mpp(ctx, "target"))
        second = list(collect_mpp(ctx, "target"))
        self.assertEqual(len(first), 2)
        for (category, h, hp), (_, repeated, _) in zip(first, second):
            self.assertEqual(category, "incompNS")
            torch.testing.assert_close(h, repeated)
            torch.testing.assert_close(hp, h * 0.4)
        inp, labels, bcs = teacher.observed[0]
        self.assertEqual(inp.shape, (3, 2, 4, 2, 2))
        torch.testing.assert_close(labels, torch.tensor([[0, 2], [0, 2]]))
        torch.testing.assert_close(inp[0, :, 0, 0, 0], bcs[:, 0].float())
        dataset.subset_dict = {}
        list(collect_mpp(ctx, "target"))
        torch.testing.assert_close(teacher.observed[-1][1], torch.full((2, 1), -1))
        self.assertFalse(teacher.target._forward_pre_hooks)
        self.assertFalse(student.target._forward_pre_hooks)

    def test_vicon_batches_replayed_once_and_trainer_restored_before_yield(self):
        from safe_svd.compress import CaptureContext, collect_vicon

        class Loader:
            calls = 0

            def __iter__(self):
                self.calls += 1
                for index in range(3):
                    yield (index, {"x": torch.ones(2, 4) * index}, None, None, None)

        class Trainer:
            def __init__(self, model):
                self.model = model
                self.preprocessed = 0

            def _move_to_device(self, pairs):
                return pairs

            def _data_preprocess(self, pairs):
                self.preprocessed += 1
                return pairs["x"] + 2, "ignored", "ignored"

            def _model_forward(self, data):
                return self.model(data)

        teacher = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        student = copy.deepcopy(teacher)
        student[0].weight.data.mul_(0.3)
        trainer = Trainer(teacher)
        loaders = {"NS": Loader(), "SWE": Loader()}
        ctx = CaptureContext(teacher, student, loaders, torch.device("cpu"), {
            "trainer": trainer, "max_calib_batches": 2,
        })
        route = iter(collect_vicon(ctx, "1"))
        category, h, hp = next(route)
        self.assertIs(trainer.model, teacher)
        route.close()
        torch.testing.assert_close(h, teacher[0](torch.full((2, 4), 2.0)))
        torch.testing.assert_close(hp, student[0](torch.full((2, 4), 2.0)))
        result = list(collect_vicon(ctx, "1"))
        self.assertEqual([entry[0] for entry in result], ["NS", "NS", "SWE", "SWE"])
        self.assertEqual([loader.calls for loader in loaders.values()], [1, 1])
        self.assertIs(trainer.model, teacher)
        self.assertFalse(student[1]._forward_pre_hooks)

    def test_vicon_failure_removes_hooks_and_restores_model(self):
        from safe_svd.compress import CaptureContext, collect_vicon

        teacher = nn.Sequential(nn.Linear(4, 4))
        student = copy.deepcopy(teacher)
        def fail(_data):
            raise ValueError("synthetic forward failure")
        trainer = SimpleNamespace(model=teacher, _move_to_device=lambda x: x,
                                  _data_preprocess=lambda x: (x, None, None), _model_forward=fail)
        ctx = CaptureContext(teacher, student, {"a": [(0, torch.ones(1, 4), None, None, None)]},
                             torch.device("cpu"), {"trainer": trainer})
        with self.assertRaisesRegex(ValueError, "synthetic"):
            list(collect_vicon(ctx, "0"))
        self.assertIs(trainer.model, teacher)
        self.assertFalse(teacher[0]._forward_pre_hooks)


def _toy_rollout(trainer, dataset, model_path=""):
    trainer.rollouts += 1
    for inputs in torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False):
        trainer.prediction_step(trainer.model, inputs, False)


def _load_source_helpers(root):
    modules = {}
    for name in ("SAFE_SVD", "SAFE_SVD.helper", "SAFE_SVD.helper.whitener", "transformers", "transformers.utils",
                 "poseidon_main", "poseidon_main.scOT", "poseidon_main.scOT.inference"):
        modules[name] = types.ModuleType(name)
        modules[name].__path__ = []
    modules["transformers.utils"].logging = SimpleNamespace(
        get_verbosity=lambda: 20, set_verbosity_error=lambda: None, set_verbosity=lambda level: None,
    )
    modules["poseidon_main.scOT.inference"].rollout = _toy_rollout
    helpers = {}
    with patch.dict(sys.modules, modules):
        # Whitening is upstream of capture and unused here. Keeping its module
        # stubbed also permits collector tests on the local Python 3.9 runtime.
        for filename in ("input_collector", "mlp_module_collector",
                         "attention_module_collector", "patch_unmerging_collector"):
            name = f"SAFE_SVD.helper.{filename}"
            spec = importlib.util.spec_from_file_location(name, root / "SAFE_SVD" / "helper" / f"{filename}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            helpers.update({key: getattr(module, key) for key in dir(module) if not key.startswith("__")})
    return helpers


def _source_function(path, name, namespace):
    """Execute the original function unchanged, without unrelated loader imports."""
    parsed = ast.parse(path.read_text())
    definition = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[definition], type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


@unittest.skipIf(torch is None, "PyTorch is required")
class PoseidonRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_helpers = (_load_source_helpers(Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]))
                                 if os.environ.get("SAFE_SVD_REFERENCE_ROOT") else None)

    def setUp(self):
        torch.manual_seed(12)

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Original checkout required")
    def test_mpp_matches_original_batch_iterator(self):
        from safe_svd.compress import CaptureContext, collect_mpp
        root = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]).parent
        original_iterator = _source_function(root / "mpp/MySVD/mysvd.py", "iter_mpp_batches", {
            "torch": torch, "DataLoader": torch.utils.data.DataLoader,
            "rearrange": lambda values, pattern: values.permute(1, 0, 2, 3, 4),
        })
        class Dataset(torch.utils.data.Dataset):
            def get_name(self):
                return "swe"
            def __len__(self):
                return 7
            def __getitem__(self, index):
                return torch.full((2, 4, 2, 2), float(index)), torch.tensor([index]), torch.zeros(2, 4, 2, 2)
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.target = nn.Linear(4, 4)
                self.seen = []
            def forward(self, inp, labels, bcs):
                self.seen.append((inp.clone(), labels.clone(), bcs.clone()))
                return self.target(inp.permute(0, 1, 3, 4, 2))
        dataset = SimpleNamespace(sub_dsets=[SimpleNamespace(get_per_file_dsets=lambda: [Dataset()])], subset_dict={"swe": [2, 3]})
        params = SimpleNamespace(batch_size=2, num_data_workers=0)
        model = Model()
        ctx = CaptureContext(model, copy.deepcopy(model), dataset, torch.device("cpu"), {
            "params": params, "batches_per_dset": 2, "calibration_seed": 39,
        })
        captured = list(collect_mpp(ctx, "target"))
        original = list(original_iterator(dataset, params, torch.device("cpu"), batches_per_dset=2, seed=39))
        self.assertEqual(len(captured), len(original))
        for (inp, labels, bcs), reference in zip(model.seen, original):
            for actual, expected in zip((inp, labels, bcs), reference[:3]):
                torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Original checkout required")
    def test_vicon_covariances_match_original_collector(self):
        from safe_svd.compress import CaptureContext, collect_vicon
        from safe_svd.helper.collector import collect_statistics
        root = Path(os.environ["SAFE_SVD_REFERENCE_ROOT"]).parent
        original_collect = _source_function(root / "vicon/MySVD/helper/input_collector.py", "collect_covariances_vicon", {
            "torch": torch, "StopForwardException": type("StopForwardException", (Exception,), {}),
        })
        teacher = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))
        student = copy.deepcopy(teacher)
        student[0].weight.data.mul_(0.3)
        trainer = SimpleNamespace(model=teacher, _move_to_device=lambda data: data,
                                  _data_preprocess=lambda data: (data * 2 + 1, None, None))
        trainer._model_forward = lambda data: trainer.model(data)
        batches = [(category, (0, torch.randn(size, 4), None, None, None))
                   for category, size in (("NS", 2), ("NS", 3), ("SWE", 7))]
        loaders = {category: [batch for cat, batch in batches if cat == category] for category in ("NS", "SWE")}
        ctx = CaptureContext(teacher, student, loaders, torch.device("cpu"), {"trainer": trainer})
        for trace_balance in (True, False):
            actual = collect_statistics(collect_vicon(ctx, "2"), 4, device=torch.device("cpu"),
                                        dtype=torch.float64, aggregation="category_mean", trace_balance=trace_balance)
            reference = original_collect(trainer, teacher, student, "2", batches, torch.device("cpu"), use_trace_align=trace_balance)
            for values, expected in zip(actual[:3], reference):
                torch.testing.assert_close(values, expected)

    def make_context(self, branch, path, features=8):
        from safe_svd.compress import CaptureContext
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(name_or_path="toy")
                self.pre = nn.Linear(features, features)
                parent = self
                parts = path.split(".")
                for part in parts[:-1]:
                    parent.add_module(part, nn.Module())
                    parent = parent.get_submodule(part)
                parent.add_module(parts[-1], branch)

            def forward(self, x, mask, time):
                x = self.pre(x)
                block = self.get_submodule(path)
                if hasattr(block, "upsample"):
                    return block(x, (3, 4), time)
                if hasattr(block, "attention"):
                    return block(x, mask)
                return block(x)

        class Trainer:
            def __init__(self, model):
                self.model = model
                self.ar_steps, self.output_all_steps = 7, True
                self.compute_metrics = object()
                self.rollouts = 0

            def set_ar_steps(self, steps, output_all_steps=False):
                self.ar_steps, self.output_all_steps = steps, output_all_steps

            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                assert self.ar_steps is None and self.compute_metrics is None
                return model(**inputs)

        dataset = [{"x": torch.randn(4, features), "mask": torch.tensor([1., 1., 0., 1.]),
                    "time": torch.tensor(float(i))} for i in range(5)]
        teacher = Model().eval()
        student = copy.deepcopy(teacher)
        student.pre.weight.data.mul_(0.6)
        trainer = Trainer(teacher)
        return CaptureContext(teacher, student, dataset, torch.device("cpu"), {
            "trainer": trainer, "rollout_fn": _toy_rollout, "capture_batch_size": 2, "offload_cache": True,
        })

    def direct_reference(self, context, model, layer_name):
        values = []
        def hook(_module, args):
            values.append(args[0].detach().reshape(-1, model.get_submodule(layer_name).in_features))
        handle = model.get_submodule(layer_name).register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                for inputs in torch.utils.data.DataLoader(context.calibration_data, batch_size=1):
                    model(**inputs)
        finally:
            handle.remove()
        return torch.cat(values)

    def assert_pairs(self, context, layer_name, pairs):
        torch.testing.assert_close(torch.cat([entry[1] for entry in pairs]),
                                   self.direct_reference(context, context.teacher, layer_name))
        torch.testing.assert_close(torch.cat([entry[2] for entry in pairs]),
                                   self.direct_reference(context, context.student, layer_name))
        trainer = context.options["trainer"]
        self.assertIs(trainer.model, context.teacher)
        self.assertEqual(trainer.ar_steps, 7)
        self.assertTrue(trainer.output_all_steps)
        self.assertIsNotNone(trainer.compute_metrics)
        self.assertTrue(all(not module._forward_pre_hooks for module in context.teacher.modules()))

    def replace_first(self, context, name):
        original = context.student.get_submodule(name)
        replacement = nn.Sequential(nn.Linear(original.in_features, 2, bias=False),
                                    nn.Linear(2, original.out_features))
        parent, _, leaf = name.rpartition(".")
        setattr(context.student.get_submodule(parent), leaf, replacement)

    def test_all_mlp_kinds_recompute_from_updated_student_without_rollout(self):
        from safe_svd.compress import collect_poseidon
        class FFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.intermediate = nn.Module()
                self.intermediate.dense = nn.Linear(8, 12)
                self.intermediate.intermediate_act_fn = torch.nn.functional.gelu
                self.output = nn.Module()
                self.output.dense = nn.Linear(12, 8)
            def forward(self, x):
                return self.output.dense(self.intermediate.intermediate_act_fn(self.intermediate.dense(x)))
        class ConvNext(nn.Module):
            def __init__(self):
                super().__init__()
                self.pwconv1, self.pwconv2, self.act = nn.Linear(8, 12), nn.Linear(12, 8), nn.GELU()
            def forward(self, x):
                return self.pwconv2(self.act(self.pwconv1(x)))
        cases = [
            (FFN(), "encoder.layers.0.blocks.0", "intermediate.dense", "output.dense", 8),
            (ConvNext(), "residual_blocks.0.0", "pwconv1", "pwconv2", 8),
            (nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 3)),
             "encoder.layers.0.blocks.0.attention.self.continuous_position_bias_mlp", "0", "2", 2),
        ]
        for branch, path, first, second, features in cases:
            with self.subTest(path=path):
                ctx = self.make_context(branch, path, features)
                first, second = f"{path}.{first}", f"{path}.{second}"
                self.assert_pairs(ctx, first, list(collect_poseidon(ctx, first)))
                calls = ctx.options["trainer"].rollouts
                self.replace_first(ctx, first)
                self.assert_pairs(ctx, second, list(collect_poseidon(ctx, second)))
                self.assertEqual(ctx.options["trainer"].rollouts, calls)

    def attention_context(self):
        class SelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.query, self.key, self.value = (nn.Linear(8, 8) for _ in range(3))
            def forward(self, x, mask):
                q, k, v = self.query(x), self.key(x), self.value(x)
                score = (q @ k.transpose(-2, -1)) / 8**0.5
                score = score.masked_fill(mask[:, None, :] == 0, -1e4)
                probability = score.softmax(dim=-1)
                return probability @ v, probability
        class AttentionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.Module()
                self.attention.add_module("self", SelfAttention())
                self.attention.output = nn.Module()
                self.attention.output.dense = nn.Linear(8, 8)
            def forward(self, x, mask):
                context = self.attention.get_submodule("self")(x, mask)[0]
                return self.attention.output.dense(context)
        return self.make_context(AttentionBlock(), "encoder.layers.0.blocks.0")

    def test_attention_reuses_qkv_inputs_and_recomputes_masked_context(self):
        from safe_svd.compress import collect_poseidon
        ctx = self.attention_context()
        prefix = "encoder.layers.0.blocks.0.attention"
        for role in ("query", "key", "value"):
            name = f"{prefix}.self.{role}"
            self.assert_pairs(ctx, name, list(collect_poseidon(ctx, name)))
            if role == "query":
                calls = ctx.options["trainer"].rollouts
                self.assertEqual(calls, 6)  # Three slices; multiple captures per slice.
            else:
                self.assertEqual(ctx.options["trainer"].rollouts, calls)
            self.replace_first(ctx, name)
        name = f"{prefix}.output.dense"
        self.assert_pairs(ctx, name, list(collect_poseidon(ctx, name)))
        self.assertEqual(ctx.options["trainer"].rollouts, calls)

    @unittest.skipUnless(os.environ.get("SAFE_SVD_REFERENCE_ROOT"), "Original checkout required")
    def test_local_attention_capture_matches_original_helper(self):
        from safe_svd.compress import collect_poseidon
        ctx = self.attention_context()
        prefix = "encoder.layers.0.blocks.0.attention.self"
        actual = list(collect_poseidon(ctx, f"{prefix}.query"))
        trainer = ctx.options["trainer"]
        original_capture = self.reference_helpers["attention_self_input_catch"]
        try:
            teacher_args = original_capture(trainer, ctx.teacher, prefix, ctx.calibration_data, ctx.device)
            student_args = original_capture(trainer, ctx.student, prefix, ctx.calibration_data, ctx.device)
        finally:
            trainer.model = ctx.teacher
        torch.testing.assert_close(torch.cat([entry[1] for entry in actual]),
                                   torch.cat([entry[0].reshape(-1, 8) for entry in teacher_args]))
        torch.testing.assert_close(torch.cat([entry[2] for entry in actual]),
                                   torch.cat([entry[0].reshape(-1, 8) for entry in student_args]))

    def test_poseidon_route_does_not_import_original_helpers(self):
        import builtins
        from safe_svd.compress import collect_poseidon
        original_import = builtins.__import__
        def isolated_import(name, *args, **kwargs):
            if name.split(".")[0] in {"SAFE_SVD", "poseidon_main", "scOT"}:
                raise AssertionError(f"Unexpected external import: {name}")
            return original_import(name, *args, **kwargs)
        ctx = self.attention_context()
        with patch("builtins.__import__", side_effect=isolated_import):
            pairs = list(collect_poseidon(ctx, "encoder.layers.0.blocks.0.attention.self.query"))
        self.assertTrue(pairs)

    def test_default_and_low_gpu_offload_inside_capture_before_student_rollout(self):
        from safe_svd.compress import collect_poseidon
        from safe_svd.helper.collector.poseidon import attention_module_collector as attention
        original_capture = attention.capture_rollout
        for explicit_offload in (False, True):
            ctx = self.attention_context()
            if not explicit_offload:
                ctx.options.pop("offload_cache")
            captures = []
            def capture(*args, **kwargs):
                self.assertEqual(kwargs["capture_device"], "cpu")
                result = original_capture(*args, **kwargs)
                self.assertTrue(all(value[0].device.type == "cpu" for value in result))
                captures.append(result)
                return result
            name = "encoder.layers.0.blocks.0.attention.self.query"
            with self.subTest(explicit_offload=explicit_offload), patch.object(attention, "capture_rollout", side_effect=capture):
                pairs = list(collect_poseidon(ctx, name))
            self.assertEqual(len(captures), 6)  # teacher+student for each of 3 slices.
            self.assert_pairs(ctx, name, pairs)

    def test_attention_uncached_and_skipped_value_use_source_fallback(self):
        from safe_svd.compress import collect_poseidon
        ctx = self.attention_context()
        prefix = "encoder.layers.0.blocks.0.attention"
        key = f"{prefix}.self.key"
        self.assert_pairs(ctx, key, list(collect_poseidon(ctx, key)))
        list(collect_poseidon(ctx, f"{prefix}.self.query"))
        self.replace_first(ctx, f"{prefix}.self.query")
        calls = ctx.options["trainer"].rollouts
        output = f"{prefix}.output.dense"
        self.assert_pairs(ctx, output, list(collect_poseidon(ctx, output)))
        self.assertGreater(ctx.options["trainer"].rollouts, calls)

    def test_patch_unmerging_recomputes_reshape_crop_norm_with_time(self):
        from safe_svd.compress import collect_poseidon
        class TimeNorm(nn.Module):
            def forward(self, x, time):
                return torch.nn.functional.layer_norm(x, (4,)) + time[:, None, None]
        class PatchUnmerging(nn.Module):
            def __init__(self):
                super().__init__()
                self.upsample, self.mixup, self.norm = nn.Linear(8, 16), nn.Linear(4, 4), TimeNorm()
            def maybe_crop(self, x, height, width):
                return x[:, :height, :width]
            def forward(self, x, dims, time):
                batch = x.shape[0]
                x = self.upsample(x).reshape(batch, 2, 2, 2, 2, 4)
                x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, 4, 4, 4)
                x = self.maybe_crop(x, *dims).reshape(batch, -1, 4)
                return self.mixup(self.norm(x, time))
        ctx = self.make_context(PatchUnmerging(), "decoder.layers.0.upsample")
        first, second = "decoder.layers.0.upsample.upsample", "decoder.layers.0.upsample.mixup"
        self.assert_pairs(ctx, first, list(collect_poseidon(ctx, first)))
        calls = ctx.options["trainer"].rollouts
        self.replace_first(ctx, first)
        self.assert_pairs(ctx, second, list(collect_poseidon(ctx, second)))
        self.assertEqual(ctx.options["trainer"].rollouts, calls)
        ctx.cache.clear()
        self.assert_pairs(ctx, second, list(collect_poseidon(ctx, second)))
        self.assertGreater(ctx.options["trainer"].rollouts, calls)

    def test_compress_skip_and_resume_release_attention_cache(self):
        from safe_svd.compress import compress, collect_poseidon
        ctx = self.attention_context()
        prefix = "encoder.layers.0.blocks.0.attention"
        captured_contexts = []
        def capture(active, name):
            captured_contexts.append(active)
            yield from collect_poseidon(active, name)
        query = f"{prefix}.self.query"
        output = f"{prefix}.output.dense"
        student = compress(ctx.teacher, ctx.calibration_data, {query: 1, output: None},
                           model_type="poseidon", device="cpu", route_options=ctx.options, capture_fn=capture)
        self.assertIsInstance(student.get_submodule(query), nn.Sequential)
        self.assertFalse(captured_contexts[-1].cache)
        self.assertFalse(captured_contexts[-1].runtime)
        resumed = compress(ctx.teacher, ctx.calibration_data, {query: 1, output: 1},
                           model_type="poseidon", device="cpu", route_options=ctx.options,
                           initial_student=student, completed_layers={query})
        self.assertIsInstance(resumed.get_submodule(output), nn.Sequential)

    def test_source_rollout_failure_restores_trainer_and_hooks(self):
        from safe_svd.compress import collect_poseidon
        ctx = self.attention_context()
        def fail(*args, **kwargs):
            raise ValueError("source rollout failed")
        ctx.options["trainer"].prediction_step = fail
        with self.assertRaisesRegex(ValueError, "source rollout"):
            list(collect_poseidon(ctx, "encoder.layers.0.blocks.0.attention.self.query"))
        trainer = ctx.options["trainer"]
        self.assertIs(trainer.model, ctx.teacher)
        self.assertEqual(trainer.ar_steps, 7)
        self.assertTrue(all(not module._forward_pre_hooks for module in ctx.teacher.modules()))


if __name__ == "__main__":
    unittest.main()
