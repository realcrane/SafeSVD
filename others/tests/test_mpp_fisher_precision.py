"""MPP Fisher must preserve the research collector's FP32 reduction order."""

import ast
import copy
import importlib
import os
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from test_unified_profiling import make_case, source_function


def research_source():
    explicit = os.environ.get("SAFE_SVD_MPP_REFERENCE")
    if explicit:
        return Path(explicit) / "MySVD/mysvd.py"
    root = os.environ.get("SAFE_SVD_REFERENCE_ROOT")
    return Path(root).parent / "mpp/MySVD/mysvd.py" if root else None


def multiphysics_case():
    torch.manual_seed(59)
    model, data, options, _ = make_case("mpp")
    prototype = data.sub_dsets[0].get_per_file_dsets()[0]
    subsets = []
    for index, category in enumerate(("swe", "incompNS", "compNS", "diffre2d")):
        subset = copy.deepcopy(prototype)
        subset.values = subset.values * (index + 1.3)
        subset.targets = subset.targets * (index + 0.7)
        subset.get_name = lambda category=category: category
        subset.title = category
        subsets.append(subset)
    data.sub_dsets = [SimpleNamespace(get_per_file_dsets=lambda: subsets)]
    data.subset_dict = {subset.get_name(): [0, 1] for subset in subsets}
    options = {**options, "params": SimpleNamespace(batch_size=1, num_data_workers=0),
               "batches_per_dset": 2.5, "calibration_seed": 42,
               "balance_ratio": 10.0, "decay_alpha": 1.0, "gradient_loss": "l1"}
    return model, data, options


class MPPFisherPrecisionTests(unittest.TestCase):
    def test_raw_covariances_are_released_before_whitening(self):
        module = importlib.import_module("safe_svd.helper.collector.fisher_collector")
        original_covariance = module.CovarianceAccumulator.covariance
        original_whitener = module.fisher_whitener
        raw_refs, merged = {}, {}

        def covariance(stats, name, **kwargs):
            raw_refs[name] = [weakref.ref(value) for value in stats.sums[name].values()]
            result = original_covariance(stats, name, **kwargs)
            merged[name] = result.clone()
            return result

        def whiten(covariance, **kwargs):
            self.assertTrue(all(ref() is None for refs in raw_refs.values() for ref in refs))
            return original_whitener(covariance, **kwargs)

        model, data, options = multiphysics_case()
        with patch.object(module.CovarianceAccumulator, "covariance", covariance), \
             patch.object(module, "fisher_whitener", whiten):
            metrics, counts, covariances = module.collect_fisher(
                model, data, model_type="mpp", device="cpu", route_options=options,
                return_statistics=True, progress=False)
        self.assertEqual(set(metrics), {"first", "second"})
        # Original's fractional limit admits ceil(2.5) single-example batches
        # per file, with 20 spatial vectors in each of the four categories.
        self.assertEqual(counts, {"first": 240, "second": 240})
        for name in metrics:
            torch.testing.assert_close(covariances[name], merged[name], rtol=0, atol=0)

    @unittest.skipUnless(research_source() is not None, "Original MPP checkout required")
    def test_multiphysics_factors_match_actual_research_source_exactly(self):
        module = importlib.import_module("safe_svd.helper.collector.fisher_collector")
        path = research_source()
        namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional,
                     "DataLoader": torch.utils.data.DataLoader,
                     "tqdm": lambda values, **kwargs: values,
                     "rearrange": lambda values, pattern: values.permute(1, 0, 2, 3, 4)}
        tree = ast.parse(path.read_text())
        config = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "DSET_SOBOLEV_CONFIG"
                              for target in node.targets))
        namespace["DSET_SOBOLEV_CONFIG"] = ast.literal_eval(config.value)
        source_function(path, "sobolev_loss_mpp", namespace)
        reference_collect = source_function(path, "collect_fisher_mpp", namespace)
        original_covariance = module.CovarianceAccumulator.covariance

        def previous_reduction(stats, name, **kwargs):
            kwargs["preserve_accumulation_dtype"] = False
            return original_covariance(stats, name, **kwargs)

        for trace_balance in (False, True):
            with self.subTest(trace_balance=trace_balance):
                model, data, options = multiphysics_case()
                expected = reference_collect(
                    copy.deepcopy(model), data, options["params"], "cpu",
                    batches_per_dset=options["batches_per_dset"], calibration_seed=42,
                    use_trace_balance=trace_balance, balance_ratio=10.0, decay_alpha=1.0,
                    loss_type="l1")
                actual = module.collect_fisher(
                    copy.deepcopy(model), data, model_type="mpp", device="cpu",
                    route_options=options, trace_balance=trace_balance, progress=False)
                with patch.object(module.CovarianceAccumulator, "covariance", previous_reduction):
                    previous = module.collect_fisher(
                        copy.deepcopy(model), data, model_type="mpp", device="cpu",
                        route_options=options, trace_balance=trace_balance, progress=False)
                self.assertEqual(set(actual), set(expected))
                for name in actual:
                    for result, reference in zip(actual[name], expected[name]):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(previous[name][index], expected[name][index])
                                    for name in expected for index in (0, 1)),
                                "Fixture must distinguish early FP64 promotion from the reference")


if __name__ == "__main__":
    unittest.main()
