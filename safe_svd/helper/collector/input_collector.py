"""Unified input profiling for MPP, VICON, and Poseidon.

collect_inputs returns (input_metrics, sample_counts) by default. Metrics are
(R, R^{-T}) CPU factors. Sample counts count flattened feature vectors, not
dataset trajectories. All hooks remain active for the complete forward pass.
"""

from __future__ import annotations

import math
import torch
from torch import nn

from .collector import CovarianceAccumulator, collection_state
from .profiling_routes import validate_route, run_input_profile
from ..whitener import right_whitener, fisher_whitener


@torch.no_grad()
def collect_inputs(
    model, calibration_data, *, model_type, device="cuda", route_options=None,
    whitening=None, eps=1e-6, eps_mode="absolute", aggregation=None,
    trace_balance=None, accumulation_dtype=None, product_dtype=torch.float32,
    return_sample_counts=True, return_statistics=False, missing="identity",
):
    """Profile all original Linear layers using a shared hook/statistics loop.

    Route options: MPP params/seed/batches_per_dset; VICON trainer and
    max_calib_batches and train_seed (or cfg.train_seed); Poseidon trainer and optional rollout_fn. Default
    whitening is spectral for MPP/VICON and fisher_cholesky for Poseidon.
    return_statistics returns (metrics, counts, mean_covariances).
    """
    options = dict(route_options or {})
    validate_route(model_type, options)
    aggregation = aggregation or ("category_mean" if model_type == "vicon" else "sample_mean")
    trace_balance = model_type != "poseidon" if trace_balance is None else trace_balance
    whitening = whitening or ("fisher_cholesky" if model_type == "poseidon" else "spectral")
    accumulation_dtype = accumulation_dtype or (torch.float32 if model_type == "poseidon" else torch.float64)
    if aggregation not in {"sample_mean", "category_mean"} or missing not in {"identity", "raise"}:
        raise ValueError("Invalid aggregation or missing-layer policy")
    if whitening not in {"spectral", "fisher_cholesky"}:
        raise ValueError("Invalid whitening method")
    if eps_mode not in {"absolute", "relative", "sum_equivalent"} or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid epsilon configuration")
    if accumulation_dtype not in {torch.float32, torch.float64} or product_dtype not in {torch.float32, torch.float64}:
        raise ValueError("Input statistics require float32 or float64")
    target = torch.device(device)
    model.to(target)
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not layers:
        raise ValueError("Model contains no Linear layers")
    stats = CovarianceAccumulator(target, accumulation_dtype)
    category = "default"
    def set_category(value):
        nonlocal category
        category = value
    handles = []
    with collection_state(model, training=False):
        try:
            for name, layer in layers:
                def hook(_module, inputs, _output, name=name, features=layer.in_features):
                    stats.add(name, category, inputs[0], features, product_dtype)
                handles.append(layer.register_forward_hook(hook))
            run_input_profile(model, calibration_data, model_type, target, options, set_category)
        finally:
            for handle in handles:
                handle.remove()
    if not stats.sums:
        raise ValueError("Calibration forward did not reach any Linear layer")
    metrics, counts, covariances = {}, {}, {}
    for name, layer in layers:
        count = stats.sample_count(name)
        counts[name] = count
        if count:
            covariance = stats.covariance(name, aggregation=aggregation, trace_balance=trace_balance)
        elif missing == "identity":
            covariance = torch.eye(layer.in_features, device=target, dtype=torch.float64)
        else:
            raise ValueError(f"No input samples for {name}")
        if whitening == "fisher_cholesky":
            left, inverse = fisher_whitener(covariance)
            factors = left.T, inverse
        else:
            factors = right_whitener(covariance, method="spectral", eps=eps,
                                     eps_mode=eps_mode, sample_count=max(count, 1))
        metrics[name] = tuple(value.cpu().float() for value in factors)
        if return_statistics:
            covariances[name] = covariance.cpu()
    if return_statistics:
        return metrics, counts, covariances
    return (metrics, counts) if return_sample_counts else metrics
