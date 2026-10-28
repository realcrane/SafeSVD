"""Numerically stable covariance whitening shared by all adapters."""

from __future__ import annotations

import torch


def svd_whitener(sigma: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``L, L_inv_transpose`` for ``sigma + eps I``.

    This is deliberately the symmetric-eigendecomposition implementation used
    by ``poseidon/MySVD_origin/helper/whitener.py``.  Keeping it here makes
    rank selection and compression use exactly the same metric.
    """
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be a square covariance matrix")
    if eps <= 0:
        raise ValueError("eps must be positive")
    eigenvalues, eigenvectors = torch.linalg.eigh((sigma + sigma.T) * 0.5)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    root = torch.sqrt(eigenvalues + eps)
    return eigenvectors * root.unsqueeze(0), eigenvectors * root.reciprocal().unsqueeze(0)


def fisher_cholesky_whitener(
    covariance_sum: torch.Tensor,
    sample_count: int,
    care_identity: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build MySVD_origin-compatible left Fisher factors.

    The orientation is important: the returned first factor is multiplied as
    ``Ll @ W`` and the second factor reverses that transform after SVD.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    fisher = (covariance_sum / sample_count).double()
    dimension = fisher.shape[0]
    identity = torch.eye(dimension, device=fisher.device, dtype=fisher.dtype)
    diagonal_mean = torch.diagonal(fisher).mean().item()
    if care_identity:
        scale = 1.0 / diagonal_mean if diagonal_mean > 0 else 1.0
        regularized = fisher * scale + identity
    else:
        jitter = max(diagonal_mean * 0.01, 1e-18) if diagonal_mean > 0 else 1e-9
        regularized = fisher + jitter * identity
    try:
        factor = torch.linalg.cholesky(regularized)
    except RuntimeError:
        values, vectors = torch.linalg.eigh(regularized)
        factor = vectors * torch.sqrt(torch.clamp(values, min=1e-18)).unsqueeze(0)
    left = factor.float().T
    try:
        left_inverse = torch.linalg.inv(left)
    except RuntimeError:
        left_inverse = torch.linalg.pinv(left)
    return left, left_inverse
