"""Shared right-side whitening for the unified compressor."""

from __future__ import annotations

import torch
from torch import Tensor


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
