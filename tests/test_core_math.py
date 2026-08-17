from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # The lightweight public-interface environment has no torch.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for core-math tests")
class CoreMathTest(unittest.TestCase):
    def test_combined_factors_match_origin_formula(self) -> None:
        from safesvd.core.factorization import combined_input_factors
        from safesvd.core.whitening import svd_whitener

        torch.manual_seed(17)
        weight = torch.randn(3, 4)
        hh = torch.randn(4, 4)
        hh = hh.T @ hh
        hp_hp = torch.randn(4, 4)
        hp_hp = hp_hp.T @ hp_hp
        h_hp = torch.randn(4, 4)
        alpha = 0.7
        factors = combined_input_factors(weight, hh, h_hp, hp_hp, 2, alpha=alpha)

        right, right_inverse_transpose = svd_whitener((alpha * hp_hp + (1 - alpha) * hh).double())
        cross = ((1 - alpha) * hh.double() + alpha * h_hp.double()) @ right_inverse_transpose
        u, singular_values, vh = torch.linalg.svd(weight.double() @ cross, full_matrices=False)
        root = singular_values[:2].sqrt()
        expected_left = u[:, :2] * root.unsqueeze(0)
        expected_right = (root.unsqueeze(1) * vh[:2]) @ right_inverse_transpose.T
        self.assertTrue(torch.allclose(factors.left, expected_left.float(), atol=1e-5))
        self.assertTrue(torch.allclose(factors.right, expected_right.float(), atol=1e-5))

    def test_rank_allocator_returns_origin_rank_convention(self) -> None:
        from safesvd.core.ranks import allocate_greedy_ranks

        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
        profiles = {
            name: (torch.eye(module.in_features), torch.eye(module.in_features))
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        ranks = allocate_greedy_ranks(model, None, profiles, 0.5, "cpu")
        self.assertEqual(set(ranks), set(profiles))
        self.assertTrue(all(rank is None or rank >= 1 for rank in ranks.values()))
