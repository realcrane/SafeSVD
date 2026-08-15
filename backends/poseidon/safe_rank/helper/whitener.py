"""Numerically stable whitening used by SAFE-SVD."""

import torch


def resolve_whitening_eps(
    covariance: torch.Tensor,
    eps: float,
    eps_mode: str = "absolute",
    sample_count: int | None = None,
) -> float:
    """Resolve a diagonal regularizer for an empirical covariance.

    In ``relative`` mode, ``eps`` is the dimensionless coefficient rho in
    ``rho * mean(diag(covariance))``.  The covariance is expected to be an
    empirical mean, not an accumulated Gram sum.

    In ``sum_equivalent`` mode, ``eps`` is expressed in raw-Gram units.  For
    ``covariance = gram_sum / N``, returning ``eps / N`` makes
    ``covariance + (eps / N) I`` exactly proportional to
    ``gram_sum + eps I``.  This mode is the controlled MySVD-speedup
    comparison; it is not an activation-scale-relative regularizer.
    """
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if eps_mode == "absolute":
        return eps
    if eps_mode == "sum_equivalent":
        if sample_count is None or sample_count <= 0:
            raise ValueError(
                "sum_equivalent eps mode requires a positive sample_count."
            )
        return eps / sample_count
    if eps_mode != "relative":
        raise ValueError(
            "Unknown eps_mode="
            f"{eps_mode!r}; expected 'absolute', 'relative', or 'sum_equivalent'."
        )

    scale = torch.diagonal(covariance).mean().item()
    # A zero-activation layer still needs a non-zero fallback regularizer.
    return eps * max(scale, torch.finfo(covariance.dtype).eps)


def svd_whitener(covariance: torch.Tensor, eps: float = 1e-6):
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    sqrt_values = torch.sqrt(eigenvalues + eps)
    whitener = eigenvectors * sqrt_values.unsqueeze(0)
    inverse_transpose = eigenvectors * (1.0 / sqrt_values).unsqueeze(0)
    return whitener, inverse_transpose


def cholesky_whitener(covariance: torch.Tensor):
    """Return right-side whitening factors using a Cholesky decomposition.

    The pair ``(factor, inverse_transpose)`` satisfies
    ``factor @ factor.T = covariance`` and
    ``inverse_transpose = factor^{-T}``.
    """
    factor = torch.linalg.cholesky(covariance).float()
    try:
        inverse_transpose = torch.linalg.inv(factor.T)
    except RuntimeError:
        inverse_transpose = torch.linalg.pinv(factor.T)
    return factor, inverse_transpose


def whitener(covariance: torch.Tensor, eps: float = 1e-6):
    """Prefer Cholesky whitening and fall back to spectral whitening.

    Cholesky is faster for positive-definite covariance matrices.  The
    spectral fallback supports positive-semidefinite or numerically indefinite
    inputs after eigenvalue clipping in :func:`svd_whitener`.
    """
    try:
        return cholesky_whitener(covariance)
    except RuntimeError:
        print("     - Cholesky whitening fail. Using SVD whitener.")
        return svd_whitener(covariance, eps=eps)


def fisher_cholesky_whitener(
    covariance_sum: torch.Tensor,
    sample_count: int,
    care_identity: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build left-side Fisher whitening factors from accumulated gradients.

    The returned pair ``(left_factor, left_inverse)`` satisfies
    ``left_factor.T @ left_factor ~= Fisher``.  It is therefore compatible
    with compression code that weights a layer as ``left_factor @ W`` and
    restores it with ``left_inverse @ U``.

    The regularization and eigen fallback deliberately match the former
    inlined Fisher-collector implementation.  ``care_identity=True`` first
    normalizes the Fisher diagonal mean to one and then adds an identity.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    # 1/N * F
    fisher = (covariance_sum / sample_count).double()
    dimension = fisher.shape[0]
    identity = torch.eye(dimension, device=fisher.device, dtype=fisher.dtype)
    diagonal_mean = torch.diagonal(fisher).mean().item()

    if care_identity:
        scale = (1.0 / diagonal_mean) if diagonal_mean > 0 else 1.0
        fisher_regularized = fisher * scale + identity
    else:
        jitter = max(diagonal_mean * 0.01, 1e-18) if diagonal_mean > 0 else 1e-9
        fisher_regularized = fisher + jitter * identity

    try:
        factor = torch.linalg.cholesky(fisher_regularized)
    except RuntimeError:
        eigenvalues, eigenvectors = torch.linalg.eigh(fisher_regularized)
        eigenvalues = torch.clamp(eigenvalues, min=1e-18)
        factor = eigenvectors * torch.sqrt(eigenvalues).unsqueeze(0)

    # Cholesky gives C @ C.T = Fisher.  The compression objective uses the
    # left factor as Ll @ W, so Ll must satisfy Ll.T @ Ll = Fisher.
    left_factor = factor.float().T
    try:
        left_inverse = torch.linalg.inv(left_factor)
    except RuntimeError:
        left_inverse = torch.linalg.pinv(left_factor)

    # Ll, Ll_inv = fisher_cholesky_whitener(gram_sum, sample_count)
    # Lr = Ll.T
    # Lr_invT = Ll_inv
    # Lr.T @ Lr = Gram
    return left_factor, left_inverse
