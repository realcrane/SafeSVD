"""Unified Linear-output-gradient covariance collection and left whitening."""

from __future__ import annotations

import torch
from torch import nn
from tqdm.auto import tqdm

from .collector import CovarianceAccumulator, collection_state
from .profiling_routes import (
    validate_route, iter_calibration_batches, fisher_context, fisher_options,
    FISHER_LOSS_ROUTES, VICON_SOBOLEV_ORDERS,
)
from .progress import calibration_batch_count
from ..whitener import fisher_whitener


def collect_fisher(
    model, calibration_data, *, model_type, device="cuda", route_options=None,
    aggregation=None, trace_balance=None, trace_groups=None, care_identity=False,
    accumulation_dtype=None, loss_fn=None, return_statistics=False, progress=True,
):
    """Collect G.T @ G from each Linear output gradient, then return (L, L^-1).

    Fisher uses train mode, full forward and backward; no early-stop hooks.
    The default loss routes are copied locally from the original models.
    loss_fn(model, CalibrationBatch) optionally supplies a custom differentiable
    scalar. Poseidon accepts its original DataLoader (preserving batch size)
    or a Dataset with route_options.fisher_batch_size (default 16).
    return_statistics returns (left_metrics, counts, mean_covariances).
    Hooks, inplace flags, trainer state and original training modes are
    restored on success/failure. Gradients are cleared; model stays on device.
    progress shows completed forward/backward batches, category and loss.
    """
    options = fisher_options(calibration_data, model_type, route_options or {})
    validate_route(model_type, options)
    aggregation = aggregation or ("category_mean" if model_type == "vicon" else "sample_mean")
    trace_balance = model_type != "poseidon" if trace_balance is None else trace_balance
    if aggregation not in {"sample_mean", "category_mean"}:
        raise ValueError("Invalid aggregation")
    if accumulation_dtype not in {None, torch.float32, torch.float64}:
        raise ValueError("Invalid accumulation_dtype")
    if trace_groups is None and model_type == "vicon":
        orders = options.get("sobolev_orders", VICON_SOBOLEV_ORDERS)
        trace_groups = {category: orders.get(category, 1) for category in calibration_data}
    target = torch.device(device)
    model.to(target)
    layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]
    if not layers:
        raise ValueError("Model contains no Linear layers")
    stats = CovarianceAccumulator(target, accumulation_dtype)
    category = "default"
    handles = []
    with torch.enable_grad(), collection_state(model, training=True, disable_inplace=True), fisher_context(model, model_type, options):
        try:
            for name, layer in layers:
                def hook(_module, _grad_input, grad_output, name=name, features=layer.out_features):
                    if grad_output and grad_output[0] is not None:
                        stats.add(name, category, grad_output[0], features)
                handles.append(layer.register_full_backward_hook(hook))
            batches = iter_calibration_batches(calibration_data, model_type, target, options)
            total = calibration_batch_count(calibration_data, model_type, options) if progress else None
            with tqdm(batches, total=total, desc=f"Fisher ({model_type})", unit="batch",
                      dynamic_ncols=True, disable=not progress) as bar:
                for tagged in bar:
                    category = tagged.category
                    model.zero_grad(set_to_none=True)
                    value = (loss_fn(model, tagged) if loss_fn is not None else
                             FISHER_LOSS_ROUTES[model_type](model, tagged, target, options))
                    if not torch.is_tensor(value) or not value.requires_grad:
                        raise ValueError("Fisher loss must be a differentiable Tensor")
                    if value.ndim:
                        value = value.mean()
                    if not torch.isfinite(value):
                        raise ValueError("Non-finite Fisher loss")
                    value.backward()
                    if progress:
                        bar.set_postfix(dataset=category, loss=f"{value.detach().item():.4g}")
                    del value, tagged
        finally:
            for handle in handles:
                handle.remove()
            model.zero_grad(set_to_none=True)
    if not stats.sums:
        raise ValueError("No Linear output gradients were collected")
    metrics, counts, covariances = {}, {}, {}
    for name, _layer in layers:
        counts[name] = stats.sample_count(name)
        if not counts[name]:
            continue  # Rank/compress use identity when a left metric is absent.
        covariance = stats.covariance(name, aggregation=aggregation, trace_balance=trace_balance,
                                      trace_groups=trace_groups,
                                      preserve_accumulation_dtype=model_type == "mpp")
        # The merged matrix owns its storage. Drop the raw per-category
        # covariances before decomposition instead of retaining every layer
        # until the final whitening operation finishes.
        stats.sums.pop(name, None)
        stats.counts.pop(name, None)
        if target.type == "cuda":
            # cuSOLVER may allocate outside PyTorch's caching allocator.
            torch.cuda.empty_cache()
        metrics[name] = tuple(value.cpu() for value in fisher_whitener(covariance, care_identity=care_identity))
        if return_statistics:
            covariances[name] = covariance.cpu()
        del covariance
    return (metrics, counts, covariances) if return_statistics else metrics
