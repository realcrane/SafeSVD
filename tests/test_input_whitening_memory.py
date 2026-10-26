"""Input whitening must release raw statistics without changing its factors."""

import importlib
import unittest
import weakref
from unittest.mock import patch

import torch

from test_unified_profiling import make_case


class InputWhiteningMemoryTest(unittest.TestCase):
    def test_raw_grams_are_released_before_decomposition_and_factors_are_equivalent(self):
        module = importlib.import_module("safe_svd.helper.collector.input_collector")
        original_covariance = module.CovarianceAccumulator.covariance
        original_whitener = module.right_whitener
        raw_refs, merged, expected = {}, {}, []

        def covariance(stats, name, **kwargs):
            raw_refs[name] = [weakref.ref(value) for value in stats.sums[name].values()]
            result = original_covariance(stats, name, **kwargs)
            merged[name] = result.clone()
            return result

        def whiten(covariance, **kwargs):
            self.assertTrue(all(ref() is None for refs in raw_refs.values() for ref in refs))
            result = original_whitener(covariance, **kwargs)
            expected.append(result)
            return result

        model, data, options, _ = make_case("mpp")
        with patch.object(module.CovarianceAccumulator, "covariance", covariance), \
             patch.object(module, "right_whitener", whiten):
            metrics, counts, covariances = module.collect_inputs(
                model, data, model_type="mpp", device="cpu", route_options=options, return_statistics=True)
        self.assertEqual(len(expected), len(metrics))
        for (name, factors), wanted in zip(metrics.items(), expected):
            self.assertGreater(counts[name], 0)
            torch.testing.assert_close(covariances[name], merged[name], rtol=0, atol=0)
            for actual, reference in zip(factors, wanted):
                torch.testing.assert_close(actual, reference.float(), rtol=0, atol=0)

    def test_failure_identifies_layer_and_preserves_solver_error(self):
        module = importlib.import_module("safe_svd.helper.collector.input_collector")
        model, data, options, _ = make_case("mpp")
        with patch.object(module, "right_whitener", side_effect=RuntimeError("CUSOLVER_STATUS_INTERNAL_ERROR")):
            with self.assertRaisesRegex(RuntimeError, r"first, covariance=\(4, 4\).*CUSOLVER_STATUS_INTERNAL_ERROR") as raised:
                module.collect_inputs(model, data, model_type="mpp", device="cpu", route_options=options)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
