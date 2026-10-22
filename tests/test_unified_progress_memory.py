"""Progress reporting and activation-lifetime regressions on CPU."""

import gc
import importlib
import io
import unittest
import weakref
from unittest.mock import patch

import torch
from torch import nn
from tqdm.auto import tqdm

from test_unified_profiling import make_case


class ProgressMemoryTests(unittest.TestCase):
    def test_fisher_reports_real_batch_totals_for_all_routes(self):
        module = importlib.import_module("safe_svd.helper.collector.fisher_collector")
        for route in ("mpp", "vicon", "poseidon"):
            model, data, options, _ = make_case(route)
            bars = []
            def progress_bar(*args, **kwargs):
                bar = tqdm(*args, **kwargs, file=io.StringIO(), mininterval=0)
                bars.append(bar)
                return bar
            with self.subTest(route=route), patch.object(module, "tqdm", side_effect=progress_bar):
                metrics = module.collect_fisher(model, data, model_type=route, device="cpu", route_options=options)
            self.assertTrue(metrics)
            self.assertEqual(len(bars), 1)
            expected = {"mpp": 2, "vicon": 4, "poseidon": 3}[route]
            self.assertEqual((bars[0].n, bars[0].total), (expected, expected))
            self.assertIn("loss=", bars[0].postfix)

    def test_batch_count_matches_limits_and_unsized_streams(self):
        from safe_svd.helper.collector.profiling_routes import calibration_batch_count, iter_calibration_batches
        model, data, options, _ = make_case("mpp")
        options["batches_per_dset"] = 1.5
        self.assertEqual(calibration_batch_count(data, "mpp", options),
                         len(list(iter_calibration_batches(data, "mpp", "cpu", options))))
        class Stream:
            def __iter__(self):
                yield "batch"
        self.assertIsNone(calibration_batch_count({"stream": Stream()}, "vicon", {"max_calib_batches": 2}))

    def test_compression_progress_counts_kept_zero_and_compressed_layers(self):
        module = importlib.import_module("safe_svd.compress")
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, 4))
        seen, bars = [], []
        def capture(context, name):
            seen.append(name)
            yield "data", torch.ones(2, 4), torch.ones(2, 4)
        def progress_bar(*args, **kwargs):
            bar = tqdm(*args, **kwargs, file=io.StringIO(), mininterval=0)
            bars.append(bar)
            return bar
        with patch.object(module, "tqdm", side_effect=progress_bar):
            student = module.compress(model, [], {"0": 1, "1": None, "2": 0}, model_type="mpp",
                                      device="cpu", capture_fn=capture)
        self.assertEqual(seen, ["0"])
        self.assertIsInstance(student[0], nn.Sequential)
        self.assertEqual(torch.count_nonzero(student[2].weight).item(), 0)
        self.assertEqual((bars[0].n, bars[0].total), (3, 3))
        self.assertIn("layer=2", bars[0].postfix)

    def test_progress_closes_on_fisher_and_compression_failure(self):
        for stage in ("fisher", "compress"):
            module = importlib.import_module("safe_svd.helper.collector.fisher_collector" if stage == "fisher"
                                             else "safe_svd.compress")
            model, data, options, _ = make_case("mpp")
            bars = []
            def progress_bar(*args, **kwargs):
                bar = tqdm(*args, **kwargs, file=io.StringIO())
                bars.append(bar)
                return bar
            def fail(*args):
                raise RuntimeError("capture failed")
            with patch.object(module, "tqdm", side_effect=progress_bar), self.assertRaisesRegex(RuntimeError, "capture failed"):
                if stage == "fisher":
                    module.collect_fisher(model, data, model_type="mpp", device="cpu", route_options=options, loss_fn=fail)
                else:
                    module.compress(model, data, {"first": 1}, model_type="mpp", device="cpu", capture_fn=fail)
            self.assertTrue(bars[0].disable)  # tqdm.close() unregisters/disables it.
            self.assertTrue(all(not layer._backward_hooks and not layer._forward_pre_hooks for layer in model.modules()))

    def test_covariance_drops_previous_activations_before_next_capture(self):
        from safe_svd.helper.collector.collector import collect_statistics
        def pairs():
            for index in range(1, 4):
                teacher, student = torch.full((2, 3), float(index)), torch.full((2, 3), float(index + 1))
                teacher_ref, student_ref = weakref.ref(teacher), weakref.ref(student)
                yield "data", teacher, student
                del teacher, student
                gc.collect()
                self.assertIsNone(teacher_ref())
                self.assertIsNone(student_ref())
        hh, hhp, hphp, count = collect_statistics(pairs(), 3, device="cpu", dtype=torch.float64,
                                                 aggregation="sample_mean", trace_balance=False)
        self.assertEqual(count, 6)
        torch.testing.assert_close(hh, torch.full((3, 3), 14 / 3, dtype=torch.float64))
        torch.testing.assert_close(hhp, torch.full((3, 3), 20 / 3, dtype=torch.float64))
        torch.testing.assert_close(hphp, torch.full((3, 3), 29 / 3, dtype=torch.float64))

    def test_fisher_metric_does_not_allocate_unused_output_identity(self):
        module = importlib.import_module("safe_svd.compress")
        model = nn.Sequential(nn.Linear(4, 8))
        metric = (torch.eye(8), torch.eye(8))
        def capture(context, name):
            yield "data", torch.randn(2, 4), torch.randn(2, 4)
        with patch.object(module.torch, "eye", wraps=torch.eye) as eye:
            module.compress(model, [], {"0": 1}, model_type="mpp", device="cpu", capture_fn=capture,
                            left_metrics={"0": metric}, progress=False)
        self.assertFalse(any(call.args[0] == 8 for call in eye.call_args_list))

    def test_eager_covariance_release_matches_existing_factorization(self):
        from safe_svd.compress import compress
        from safe_svd.helper.whitener import right_whitener
        torch.manual_seed(719)
        model = nn.Sequential(nn.Linear(4, 6)).double()
        h, hp = torch.randn(12, 4).double(), torch.randn(12, 4).double()
        left = torch.diag(torch.arange(1, 7).double())
        metric = (left, torch.linalg.inv(left))
        def capture(context, name):
            yield "data", h, hp
        for alpha in (0., 0.7, 1.):
            stats = {"hh": h.T @ h / 12, "hph": h.T @ hp / 12, "hphp": hp.T @ hp / 12}
            _, inverse = right_whitener(alpha * stats["hphp"] + (1 - alpha) * stats["hh"],
                                       method="spectral", eps=1e-6, eps_mode="absolute", sample_count=12)
            matrix = (left @ model[0].weight) @ (((1 - alpha) * stats["hh"] + alpha * stats["hph"]) @ inverse)
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
            # MPP reference dewhitens truncated factors in float32.
            root = s[:2].sqrt()
            expected_left = (metric[1].float() @ (u[:, :2] * root).float()).double()
            expected_right = ((root[:, None] * vh[:2]).float() @ inverse.T.float()).double()
            student = compress(model, [], {"0": 2}, model_type="mpp", device="cpu", alpha=alpha,
                               left_metrics={"0": metric}, capture_fn=capture, progress=False)
            torch.testing.assert_close(student[0][1].weight @ student[0][0].weight,
                                       expected_left @ expected_right, rtol=1e-8, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
