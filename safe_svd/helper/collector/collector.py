"""Shared hook capture and covariance accumulation for the model routes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn


CapturedPairs = Iterable[tuple[str, Tensor, Tensor]]


@dataclass(frozen=True)
class CalibrationBatch:
    category: str
    payload: Any


class CovarianceAccumulator:
    """Shared per-layer/category matrix sums and flattened-vector counts."""

    def __init__(self, device, accumulation_dtype):
        self.device = torch.device(device)
        self.dtype = accumulation_dtype
        self.sums: dict[str, dict[str, Tensor]] = {}
        self.counts: dict[str, dict[str, int]] = {}

    def add(self, name, category, values, features, product_dtype=None):
        rows = feature_rows(values.detach(), features).to(self.device)
        if not rows.shape[0]:
            return
        if product_dtype is not None:
            rows = rows.to(product_dtype)
        gram = rows.T @ rows
        if self.dtype is not None:
            gram = gram.to(self.dtype)
        if not torch.isfinite(gram).all():
            raise ValueError(f"Non-finite statistics for {name}")
        sums = self.sums.setdefault(name, {})
        counts = self.counts.setdefault(name, {})
        if category in sums:
            sums[category].add_(gram)
        else:
            sums[category] = gram
        counts[category] = counts.get(category, 0) + rows.shape[0]

    def sample_count(self, name):
        return sum(self.counts.get(name, {}).values())

    def covariance(self, name, *, aggregation, trace_balance, trace_groups=None):
        return merge_covariances(self.sums[name], self.counts[name], aggregation=aggregation,
                                 trace_balance=trace_balance, trace_groups=trace_groups)


def merge_covariances(sums, counts, *, aggregation="sample_mean", trace_balance=False, trace_groups=None):
    """Category means / global sample mean, optionally balancing within groups."""
    if aggregation not in {"sample_mean", "category_mean"}:
        raise ValueError("aggregation must be sample_mean or category_mean")
    categories = [name for name in sums if counts.get(name, 0) > 0]
    if not categories:
        raise ValueError("No samples were collected")
    if not trace_balance and aggregation == "sample_mean":
        return sum(sums[name].double() for name in categories) / sum(counts[name] for name in categories)
    means = {name: (sums[name] / counts[name]).double() for name in categories}
    if trace_balance:
        groups = {}
        for name in categories:
            groups.setdefault((trace_groups or {}).get(name, "all"), []).append(name)
        for members in groups.values():
            traces = {name: means[name].trace().item() for name in members}
            target = max(traces.values())
            for name in members:
                if traces[name] > 1e-15:
                    means[name] = means[name] * (target / traces[name])
    return sum(means.values()) / len(categories)


@contextmanager
def collection_state(model, *, training, disable_inplace=False):
    """Restore per-module modes and inplace flags even when collection fails."""
    modes = [(module, module.training) for module in model.modules()]
    inplace = [(module, module.inplace) for module in model.modules() if hasattr(module, "inplace")]
    try:
        model.train(training)
        if disable_inplace:
            for module, _value in inplace:
                module.inplace = False
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        for module, value in inplace:
            module.inplace = value


@contextmanager
def trainer_model(trainer, model):
    old_model = trainer.model
    try:
        trainer.model = model
        yield
    finally:
        trainer.model = old_model


@dataclass
class CaptureContext:
    teacher: nn.Module
    student: nn.Module
    calibration_data: Any
    device: torch.device
    options: Mapping[str, Any]
    # Consumer layer -> cached predecessor inputs/arguments. The current
    # student recomputes downstream values after predecessor replacement.
    cache: dict[str, Any] = field(default_factory=dict)
    ranks: Mapping[str, int | None] = field(default_factory=dict)
    # Per-run raw calibration batches; no external helper registry.
    runtime: dict[str, Any] = field(default_factory=dict)


CaptureFunction = Callable[[CaptureContext, str], CapturedPairs]


def cache_pair(context: CaptureContext, teacher_value, student_value):
    device = "cpu" if context.options.get("offload_cache", context.options.get("low_gpu", True)) else context.device
    return tree_to(teacher_value, device), tree_to(student_value, device)


def paired_values(teacher_values, student_values):
    if len(teacher_values) != len(student_values):
        raise RuntimeError("Teacher/student captured different numbers of rollout batches")
    return zip(teacher_values, student_values)


class CaptureStop(Exception):
    """Interrupt a forward immediately after observing the target input."""


def capture_input(model: nn.Module, name: str, forward: Callable[[], Any]) -> Tensor | None:
    """Capture the first target invocation; never leave a pre-hook installed."""
    module = model.get_submodule(name)
    if isinstance(module, nn.Sequential):
        module = module[0]
    captured = []

    def hook(_module, args):
        captured.append(args[0].detach().float())
        raise CaptureStop

    handle = module.register_forward_pre_hook(hook)
    try:
        forward()
    except CaptureStop:
        pass
    finally:
        handle.remove()
    return captured[0] if captured else None


def tree_to(value: Any, device: torch.device | str) -> Any:
    """Detach nested block arguments without discarding masks or time inputs."""
    if torch.is_tensor(value):
        return value.detach().to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(tree_to(item, device) for item in value)
    if isinstance(value, list):
        return [tree_to(item, device) for item in value]
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    return value


def feature_rows(values: Tensor | list[Tensor], in_features: int) -> Tensor:
    """Use Poseidon's feature layout convention, including transposed 2D inputs."""
    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("No captured inputs")
        values = torch.cat(list(values), dim=0)
    if values.ndim == 2 and values.shape[1] == in_features:
        return values
    if values.ndim >= 2 and values.shape[-1] == in_features:
        return values.reshape(-1, in_features)
    if values.ndim == 2 and values.shape[0] == in_features:
        return values.T
    raise ValueError(f"Expected {in_features} features, got {tuple(values.shape)}")


def collect_statistics(
    pairs: CapturedPairs,
    in_features: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    aggregation: str,
    trace_balance: bool,
    normalize: bool = True,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Return HH, H.T @ Hp, Hp.T @ Hp, and captured feature-row count.

    Default to means; normalize=False returns raw sums for MySVD_fast.

    Pairs must describe the same ordered examples/tokens. Routes must restore
    hooks/trainer state in finally blocks and use repeatable calibration data.
    """
    buckets: dict[str, tuple[Tensor, Tensor, Tensor, int]] = {}
    for category, teacher_input, student_input in pairs:
        for values in (teacher_input, student_input):
            if values.ndim < 2 or values.shape[-1] != in_features:
                raise ValueError("Captured inputs must have shape (..., in_features)")
        h = teacher_input.detach().reshape(-1, in_features).to(device=device, dtype=dtype)
        hp = student_input.detach().reshape(-1, in_features).to(device=device, dtype=dtype)
        if h.shape != hp.shape:
            raise ValueError("Teacher/student input shapes differ")
        if not h.shape[0]:
            continue
        if category not in buckets:
            zero = torch.zeros(in_features, in_features, device=device, dtype=dtype)
            buckets[category] = (zero, zero.clone(), zero.clone(), 0)
        hh, hhp, hphp, count = buckets[category]
        hh.addmm_(h.T, h)
        hhp.addmm_(h.T, hp)
        hphp.addmm_(hp.T, hp)
        buckets[category] = (hh, hhp, hphp, count + h.shape[0])
        # Release the previous batch before requesting the next capture;
        # converted float64 activations otherwise overlap the next forward.
        del h, hp, teacher_input, student_input, values

    if not buckets:
        raise ValueError("No paired calibration inputs were captured")
    count = sum(entry[3] for entry in buckets.values())
    if not normalize:
        if aggregation != "sample_mean" or trace_balance:
            raise ValueError("Raw sums require sample_mean without trace balancing")
        merged = [sum(entry[i] for entry in buckets.values()) for i in range(3)]
    elif aggregation == "sample_mean" and not trace_balance:
        merged = [sum(entry[i] for entry in buckets.values()) / count for i in range(3)]
    else:
        means = [tuple(entry[i] / entry[3] for i in range(3)) for entry in buckets.values()]
        traces = [entry[0].trace().item() for entry in means]
        target_trace = max(traces)
        scales = [target_trace / tr if trace_balance and tr > 1e-15 else 1.0 for tr in traces]
        merged = [
            sum(entry[i] * scale for entry, scale in zip(means, scales)) / len(means)
            for i in range(3)
        ]
    if not all(torch.isfinite(value).all().item() for value in merged):
        raise ValueError("Captured statistics contain NaN or infinity")
    return merged[0], merged[1], merged[2], count
