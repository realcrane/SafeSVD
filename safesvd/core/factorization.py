"""The MySVD_origin weighted SVD and low-rank module construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .whitening import svd_whitener


@dataclass(frozen=True)
class LowRankFactors:
    """Weights for ``Linear(in, rank, bias=False) -> Linear(rank, out)``."""

    right: torch.Tensor
    left: torch.Tensor


@torch.no_grad()
def combined_input_factors(
    weight: torch.Tensor,
    sigma_hh: torch.Tensor,
    sigma_hph: torch.Tensor,
    sigma_hphp: torch.Tensor,
    rank: int,
    *,
    alpha: float,
    left_whitener: tuple[torch.Tensor, torch.Tensor] | None = None,
    eps: float = 1e-6,
) -> LowRankFactors:
    """Factor one linear weight using MySVD_origin's combined-input objective."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    out_features, in_features = weight.shape
    if not 1 <= rank <= min(out_features, in_features):
        raise ValueError("rank is outside the useful range for this weight")
    work_weight = weight.detach().to(dtype=torch.float64)
    device, dtype = work_weight.device, work_weight.dtype
    sigma_hh = sigma_hh.to(device=device, dtype=dtype)
    sigma_hph = sigma_hph.to(device=device, dtype=dtype)
    sigma_hphp = sigma_hphp.to(device=device, dtype=dtype)

    right_metric = alpha * sigma_hphp + (1 - alpha) * sigma_hh
    right, right_inverse_transpose = svd_whitener(right_metric, eps=eps)
    del right_metric
    if left_whitener is None:
        left = torch.eye(out_features, device=device, dtype=dtype)
        left_inverse_transpose = left
    else:
        left = left_whitener[0].to(device=device, dtype=dtype)
        left_inverse_transpose = left_whitener[1].to(device=device, dtype=dtype)

    cross = ((1 - alpha) * sigma_hh + alpha * sigma_hph) @ right_inverse_transpose
    weighted_weight = left @ work_weight
    matrix = weighted_weight @ cross
    del weighted_weight, cross, work_weight, left, right
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    del matrix
    root = torch.sqrt(singular_values[:rank])
    left_factor = u[:, :rank] * root.unsqueeze(0)
    right_factor = root.unsqueeze(1) * vh[:rank, :]
    right_factor = right_factor @ right_inverse_transpose.T
    left_factor = left_inverse_transpose @ left_factor
    result = LowRankFactors(right=right_factor.to(dtype=weight.dtype), left=left_factor.to(dtype=weight.dtype))
    del u, singular_values, vh, right_inverse_transpose, left_inverse_transpose
    del left_factor, right_factor
    return result


@torch.no_grad()
def combined_input_factors_consuming(
    weight: torch.Tensor,
    statistics: dict[str, torch.Tensor],
    rank: int,
    *,
    alpha: float,
    left_whitener: tuple[torch.Tensor, torch.Tensor] | None = None,
    eps: float = 1e-6,
) -> LowRankFactors:
    """MySVD_fast factorization that consumes covariance storage eagerly.

    ``statistics`` is emptied as each matrix becomes unnecessary.  Unlike a
    normal function call with three tensor arguments, this lets the caller drop
    covariance ownership *before* allocating the SVD work matrices.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    try:
        sigma_hh = statistics.pop("hh")
        sigma_hph = statistics.pop("hph")
        sigma_hphp = statistics.pop("hphp")
    except KeyError as exc:
        raise KeyError("statistics must contain hh, hph, and hphp") from exc

    out_features, in_features = weight.shape
    if not 1 <= rank <= min(out_features, in_features):
        raise ValueError("rank is outside the useful range for this weight")
    work_weight = weight.detach().to(dtype=torch.float64)
    device, dtype = work_weight.device, work_weight.dtype
    sigma_hh = sigma_hh.to(device=device, dtype=dtype)
    sigma_hph = sigma_hph.to(device=device, dtype=dtype)
    sigma_hphp = sigma_hphp.to(device=device, dtype=dtype)

    right_metric = sigma_hphp.mul(alpha).add(sigma_hh, alpha=1 - alpha)
    del sigma_hphp
    right, right_inverse_transpose = svd_whitener(right_metric, eps=eps)
    del right_metric

    if left_whitener is None:
        left = torch.eye(out_features, device=device, dtype=dtype)
        left_inverse_transpose = left
    else:
        left = left_whitener[0].to(device=device, dtype=dtype)
        left_inverse_transpose = left_whitener[1].to(device=device, dtype=dtype)

    cross = sigma_hh.mul(1 - alpha).add(sigma_hph, alpha=alpha)
    del sigma_hh, sigma_hph
    cross = cross @ right_inverse_transpose
    matrix = (left @ work_weight) @ cross
    del cross, work_weight, left, right
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    del matrix

    root = torch.sqrt(singular_values[:rank])
    left_factor = u[:, :rank] * root.unsqueeze(0)
    right_factor = root.unsqueeze(1) * vh[:rank, :]
    del u, singular_values, vh, root
    right_factor = right_factor @ right_inverse_transpose.T
    left_factor = left_inverse_transpose @ left_factor
    result = LowRankFactors(right=right_factor.to(dtype=weight.dtype), left=left_factor.to(dtype=weight.dtype))
    del right_inverse_transpose, left_inverse_transpose, left_factor, right_factor
    return result


@torch.no_grad()
def clean_input_factors(
    weight: torch.Tensor,
    right_whitener: tuple[torch.Tensor, torch.Tensor],
    rank: int,
    *,
    left_whitener: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> LowRankFactors:
    """Factor a layer through MySVD_origin's ``alpha == 0`` clean route."""
    out_features, in_features = weight.shape
    if not 1 <= rank <= min(out_features, in_features):
        raise ValueError("rank is outside the useful range for this weight")
    work_weight = weight.detach().to(dtype=torch.float64)
    device, dtype = work_weight.device, work_weight.dtype
    right, right_inverse_transpose = (tensor.to(device=device, dtype=dtype) for tensor in right_whitener)
    if left_whitener is None:
        left = torch.eye(out_features, device=device, dtype=dtype)
        left_inverse_transpose = left
    else:
        left, left_inverse_transpose = (tensor.to(device=device, dtype=dtype) for tensor in left_whitener)
    u, singular_values, vh = torch.linalg.svd((left @ work_weight) @ right, full_matrices=False)
    root = torch.sqrt(singular_values[:rank])
    left_factor = left_inverse_transpose @ (u[:, :rank] * root.unsqueeze(0))
    right_factor = (root.unsqueeze(1) * vh[:rank, :]) @ right_inverse_transpose.T
    return LowRankFactors(right=right_factor.to(dtype=weight.dtype), left=left_factor.to(dtype=weight.dtype))


@torch.no_grad()
def make_low_rank_linear(original: nn.Linear, factors: LowRankFactors) -> nn.Sequential:
    """Create the exact module replacement used by MySVD_origin."""
    rank = factors.right.shape[0]
    replacement = nn.Sequential(
        nn.Linear(original.in_features, rank, bias=False),
        nn.Linear(rank, original.out_features, bias=original.bias is not None),
    ).to(device=original.weight.device, dtype=original.weight.dtype)
    replacement[0].weight.copy_(factors.right)
    replacement[1].weight.copy_(factors.left)
    if original.bias is not None:
        replacement[1].bias.copy_(original.bias)
    return replacement
