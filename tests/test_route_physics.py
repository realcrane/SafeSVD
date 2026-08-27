from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for route-physics tests")
class RoutePhysicsTest(unittest.TestCase):
    def test_vicon_perfect_prediction_has_zero_supervised_errors(self) -> None:
        from safesvd.adapters.vicon_physics import physics_metrics

        value = torch.randn(2, 1, 6, 8, 8)
        metrics = physics_metrics("NS2D", value, value)
        self.assertAlmostEqual(metrics["sobolev"], 0.0)
        self.assertAlmostEqual(metrics["vorticity_err"], 0.0)

    def test_mpp_perfect_prediction_has_zero_supervised_errors(self) -> None:
        from safesvd.adapters.mpp_physics import physics_metrics

        value = torch.randn(2, 4, 8, 8)
        metrics = physics_metrics("compNS", value, value, value)
        self.assertAlmostEqual(metrics["sobolev"], 0.0)
        self.assertIn("mass_cons", metrics)
