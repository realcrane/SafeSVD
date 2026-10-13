"""Shared right-side whitening for the unified compressor."""

from __future__ import annotations

import torch
from torch import Tensor


def fisher_whitener(covariance: Tensor, *, care_identity: bool = False):
    """Return L, L^{-1} with L.T @ L equal to regularized gradient covariance.

    Copied from the original Fisher Cholesky/eigen routine, including its
    float32 factor and inverse convention. covariance is already a mean.
    """
    fisher = covariance.double()
    identity = torch.eye(fisher.shape[0], device=fisher.device, dtype=fisher.dtype)
    scale = fisher.diagonal().mean().item()
    if care_identity:
        regularized = fisher * (1.0 / scale if scale > 0 else 1.0) + identity
    else:
        jitter = max(scale * 0.01, 1e-18) if scale > 0 else 1e-9
        regularized = fisher + jitter * identity
    try:
        factor = torch.linalg.cholesky(regularized)
    except RuntimeError:
        values, vectors = torch.linalg.eigh(regularized)
        factor = vectors * values.clamp_min(1e-18).sqrt().unsqueeze(0)
    left = factor.float().T
    try:
        inverse = torch.linalg.inv(left)
    except RuntimeError:
        inverse = torch.linalg.pinv(left)
    return left, inverse


def right_whitener(
    covariance: Tensor,
    *,
    method: str,
    eps: float,
    eps_mode: str,
    sample_count: int,
) -> tuple[Tensor, Tensor]:
    """Return R, R^{-T}. Input covariance always follows a mean convention.

    fisher_cholesky follows the relative-jitter Fisher convention; eps options
    apply only to spectral whitening. sum_equivalent means eps / sample_count.
    """
    covariance = (covariance + covariance.T) * 0.5
    scale = max(covariance.diagonal().mean().item(), 0.0)
    identity = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    if method == "fisher_cholesky":
        jitter = max(scale * 0.01, 1e-18) if scale > 0 else 1e-9
        regularized = covariance + jitter * identity
        factor, info = torch.linalg.cholesky_ex(regularized)
        if info.item() == 0:
            inverse_transpose = torch.linalg.solve_triangular(factor, identity, upper=False).T
            return factor, inverse_transpose
        values, vectors = torch.linalg.eigh(regularized)
        roots = values.clamp_min(1e-18).sqrt()
    else:
        resolved_eps = eps
        if eps_mode == "relative":
            resolved_eps *= max(scale, torch.finfo(covariance.dtype).eps)
        elif eps_mode == "sum_equivalent":
            resolved_eps /= sample_count
        values, vectors = torch.linalg.eigh(covariance)
        roots = (values.clamp_min(0) + resolved_eps).sqrt()
    return vectors * roots.unsqueeze(0), vectors * roots.reciprocal().unsqueeze(0)
