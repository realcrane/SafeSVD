"""Shared runtime glue for non-Poseidon SAFE-SVD adapters.

The mathematical operations live in :mod:`safesvd.core`.  This module keeps
the repetitive, model-independent runtime work in one place: resolving nested
``nn.Module`` paths, collecting category-aware covariance statistics, saving a
portable compressed state dict, and restoring low-rank topology for evaluation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from safesvd.core import fisher_cholesky_whitener, svd_whitener


Forward = Callable[[nn.Module, Any], Any]
Loss = Callable[[nn.Module, "TaggedBatch"], torch.Tensor]


@dataclass(frozen=True)
class TaggedBatch:
    """One raw batch and its physics/dataset category."""

    category: str
    payload: Any


class CaptureStop(RuntimeError):
    """Internal exception used to stop a forward once a target input is seen."""


def cache_tagged_batches(
    batches: Callable[[], Iterable[TaggedBatch]],
) -> Callable[[], Iterable[TaggedBatch]]:
    """Materialise calibration batches once and replay the same raw payloads.

    Sequential MySVD must compare every evolving student layer on an identical
    calibration set.  This mirrors the paper collectors' CPU raw-batch cache
    and avoids re-sampling stochastic datasets for each target layer.
    """
    cached = tuple(batches())
    if not cached:
        raise ValueError("No calibration batches were available to cache")

    def replay() -> Iterable[TaggedBatch]:
        return iter(cached)

    return replay


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a requested device without silently changing experiment semantics."""
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu.")
    return target


def get_module_by_name(model: nn.Module, dotted_name: str) -> nn.Module:
    """Look up a module path, including ``ModuleList``/``Sequential`` indices."""
    if not dotted_name:
        return model
    try:
        return model.get_submodule(dotted_name)
    except AttributeError:  # pragma: no cover - older PyTorch fallback
        module = model
        for part in dotted_name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module


def set_module_by_name(model: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    """Replace a module path while supporting numerical container components."""
    parent_name, separator, leaf = dotted_name.rpartition(".")
    parent = get_module_by_name(model, parent_name) if separator else model
    if leaf.isdigit():
        parent[int(leaf)] = replacement  # type: ignore[index]
    else:
        setattr(parent, leaf, replacement)


def disable_inplace_operations(model: nn.Module) -> list[nn.Module]:
    """Disable in-place activations and return the modules changed."""
    changed: list[nn.Module] = []
    for module in model.modules():
        if getattr(module, "inplace", False):
            module.inplace = False
            changed.append(module)
    return changed


def _linear_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]


def _as_2d(inputs: tuple[Any, ...], in_features: int, device: torch.device) -> torch.Tensor:
    if not inputs or not torch.is_tensor(inputs[0]):
        raise TypeError("Linear forward hook did not receive a tensor as its first input")
    value = inputs[0].detach()
    if value.shape[-1] != in_features:
        raise ValueError(
            f"Captured activation has final dimension {value.shape[-1]}, expected {in_features}"
        )
    return value.reshape(-1, in_features).to(device=device, dtype=torch.float64)


def _merge_category_covariances(
    sums: Mapping[str, torch.Tensor],
    counts: Mapping[str, int],
    *,
    trace_reference: Mapping[str, torch.Tensor] | None = None,
    trace_balance: bool,
    trace_groups: Mapping[str, object] | None = None,
    category_average_without_trace: bool = True,
) -> torch.Tensor:
    """Average covariance matrices using the VICON/MPP category convention.

    With trace balancing, every category is first normalised by its own sample
    count and then scaled to the maximum teacher-input trace in its group.  The
    groups parameter captures VICON's rule of balancing only datasets with the
    same Sobolev order; MPP leaves it unset and balances all categories.  The
    paper routes diverge in no-trace mode: ViCON averages categories, while
    MPP uses a global sample-weighted covariance.
    """
    categories = [category for category, count in counts.items() if count > 0 and category in sums]
    if not categories:
        raise ValueError("No activation or gradient samples were captured")
    per_category = {category: sums[category] / counts[category] for category in categories}
    if not trace_balance:
        if category_average_without_trace:
            return sum(per_category.values()) / len(per_category)
        total_count = sum(counts[category] for category in categories)
        return sum(sums[category] for category in categories) / total_count

    references = trace_reference or per_category
    groups: dict[object, list[str]] = defaultdict(list)
    for category in categories:
        groups[(trace_groups or {}).get(category, "all")].append(category)

    merged: list[torch.Tensor] = []
    for members in groups.values():
        traces = {category: torch.trace(references[category]).item() for category in members}
        target = max(traces.values())
        for category in members:
            trace = traces[category]
            scale = target / trace if trace > 1e-15 else 1.0
            merged.append(per_category[category] * scale)
    return sum(merged) / len(merged)


@torch.no_grad()
def collect_input_whiteners(
    model: nn.Module,
    batches: Callable[[], Iterable[TaggedBatch]],
    forward: Forward,
    *,
    device: str | torch.device,
    trace_balance: bool,
    trace_groups: Mapping[str, object] | None = None,
    category_average_without_trace: bool = True,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Profile per-linear input covariances and build SAFE-SVD whiteners."""
    target = resolve_device(device)
    model = model.to(target).eval()
    layers = _linear_layers(model)
    sums: dict[str, dict[str, torch.Tensor]] = {name: {} for name, _ in layers}
    counts: dict[str, dict[str, int]] = {name: defaultdict(int) for name, _ in layers}
    current_category = "default"
    handles = []

    for name, layer in layers:
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any, *, layer_name: str = name,
                 features: int = layer.in_features) -> None:
            values = _as_2d(inputs, features, target)
            covariance = values.T @ values
            if current_category not in sums[layer_name]:
                sums[layer_name][current_category] = covariance
            else:
                sums[layer_name][current_category].add_(covariance)
            counts[layer_name][current_category] += values.shape[0]

        handles.append(layer.register_forward_hook(hook))

    try:
        for tagged in batches():
            current_category = tagged.category
            forward(model, tagged.payload)
    finally:
        for handle in handles:
            handle.remove()

    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, layer in layers:
        if sums[name]:
            covariance = _merge_category_covariances(
                sums[name], counts[name], trace_balance=trace_balance, trace_groups=trace_groups,
                category_average_without_trace=category_average_without_trace,
            )
        else:
            # Optional branches can leave a Linear untouched by the selected
            # calibration set.  The paper profilers use an identity metric for
            # those layers rather than failing the full run.
            covariance = torch.eye(layer.in_features, device=target, dtype=torch.float64)
        profiles[name] = tuple(value.cpu().float() for value in svd_whitener(covariance))  # type: ignore[assignment]
        del covariance
    model.cpu()
    _empty_cuda_cache(target)
    return profiles


def collect_fisher_whiteners(
    model: nn.Module,
    batches: Callable[[], Iterable[TaggedBatch]],
    loss: Loss,
    *,
    device: str | torch.device,
    trace_balance: bool,
    trace_groups: Mapping[str, object] | None = None,
    category_average_without_trace: bool = True,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Collect output-gradient covariances and create left Fisher whiteners."""
    target = resolve_device(device)
    model = model.to(target).train()
    changed_inplace = disable_inplace_operations(model)
    layers = _linear_layers(model)
    sums: dict[str, dict[str, torch.Tensor]] = {name: {} for name, _ in layers}
    counts: dict[str, dict[str, int]] = {name: defaultdict(int) for name, _ in layers}
    current_category = "default"
    handles = []

    for name, _layer in layers:
        def hook(_module: nn.Module, _grad_input: tuple[Any, ...], grad_output: tuple[Any, ...], *,
                 layer_name: str = name) -> None:
            if not grad_output or not torch.is_tensor(grad_output[0]):
                return
            gradients = grad_output[0].detach().reshape(-1, grad_output[0].shape[-1]).to(
                device=target, dtype=torch.float64,
            )
            covariance = gradients.T @ gradients
            if current_category not in sums[layer_name]:
                sums[layer_name][current_category] = covariance
            else:
                sums[layer_name][current_category].add_(covariance)
            counts[layer_name][current_category] += gradients.shape[0]

        handles.append(_layer.register_full_backward_hook(hook))

    try:
        for tagged in batches():
            current_category = tagged.category
            model.zero_grad(set_to_none=True)
            value = loss(model, tagged)
            if value.ndim != 0:
                value = value.mean()
            value.backward()
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for module in changed_inplace:
            module.inplace = True

    whiteners: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, _layer in layers:
        if not sums[name]:
            # Match the paper Fisher collectors, which omit layers that never
            # received a gradient.  Rank selection then uses no left metric.
            continue
        covariance = _merge_category_covariances(
            sums[name], counts[name], trace_balance=trace_balance, trace_groups=trace_groups,
            category_average_without_trace=category_average_without_trace,
        )
        # ``fisher_cholesky_whitener`` expects a sum and count.  Passing a
        # covariance with count one preserves the already category-balanced
        # statistic exactly.
        left, inverse = fisher_cholesky_whitener(covariance, 1)
        whiteners[name] = (left.cpu(), inverse.cpu())
        del covariance
    model.eval().cpu()
    _empty_cuda_cache(target)
    return whiteners


class CategoryAwareStatistics:
    """Stream a target layer's teacher/student inputs into three covariances."""

    def __init__(
        self,
        batches: Callable[[], Iterable[TaggedBatch]],
        forward: Forward,
        *,
        device: str | torch.device,
        trace_balance: bool,
        category_average_without_trace: bool = True,
    ) -> None:
        self._batches = batches
        self._forward = forward
        self._target = resolve_device(device)
        self._trace_balance = trace_balance
        self._category_average_without_trace = category_average_without_trace

    @torch.no_grad()
    def __call__(self, teacher: nn.Module, student: nn.Module, layer_name: str) -> dict[str, torch.Tensor]:
        teacher_layer = get_module_by_name(teacher, layer_name)
        student_layer = get_module_by_name(student, layer_name)
        if not isinstance(teacher_layer, nn.Linear):
            raise TypeError(f"Teacher layer {layer_name} is not nn.Linear")
        if not isinstance(student_layer, nn.Linear):
            raise TypeError(f"Student layer {layer_name} is not nn.Linear")
        features = teacher_layer.in_features
        if student_layer.in_features != features:
            raise ValueError(f"Teacher/student input size differs for {layer_name}")

        hh: dict[str, torch.Tensor] = {}
        hph: dict[str, torch.Tensor] = {}
        hphp: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = defaultdict(int)

        def captured_input(module: nn.Module, inputs: tuple[Any, ...], _output: Any = None) -> None:
            values = _as_2d(inputs, features, self._target)
            # We use a forward pre-hook in registration below.  The optional
            # output keeps this small helper usable in local tests as well.
            captured_input.value = values  # type: ignore[attr-defined]
            raise CaptureStop

        def capture(model: nn.Module, layer: nn.Linear, payload: Any) -> torch.Tensor | None:
            if hasattr(captured_input, "value"):
                delattr(captured_input, "value")
            handle = layer.register_forward_pre_hook(captured_input)
            try:
                self._forward(model, payload)
            except CaptureStop:
                pass
            finally:
                handle.remove()
            value = getattr(captured_input, "value", None)
            if value is not None:
                delattr(captured_input, "value")
            return value

        teacher.eval()
        student.eval()
        for tagged in self._batches():
            teacher_values = capture(teacher, teacher_layer, tagged.payload)
            student_values = capture(student, student_layer, tagged.payload)
            if teacher_values is None and student_values is None:
                # An optional branch was not active for this calibration
                # example.  It contributes no covariance, as in MySVD.
                continue
            if teacher_values is None or student_values is None:
                raise RuntimeError(f"Teacher/student target reachability differs for {layer_name}")
            if teacher_values.shape[0] != student_values.shape[0]:
                raise ValueError(f"Teacher/student captured counts differ for {layer_name}")

            category = tagged.category
            values = (
                teacher_values.T @ teacher_values,
                teacher_values.T @ student_values,
                student_values.T @ student_values,
            )
            for target_dict, value in zip((hh, hph, hphp), values):
                if category in target_dict:
                    target_dict[category].add_(value)
                else:
                    target_dict[category] = value
            counts[category] += teacher_values.shape[0]
            del teacher_values, student_values, values

        if not counts:
            zero = torch.zeros((features, features), device=self._target, dtype=torch.float64)
            return {"hh": zero, "hph": zero.clone(), "hphp": zero.clone()}

        # Trace factors in the paper compression collector are derived from
        # each category's *mean* teacher covariance, not its raw sample sum.
        teacher_averages = {category: hh[category] / counts[category] for category in counts}
        return {
            "hh": _merge_category_covariances(
                hh, counts, trace_balance=self._trace_balance,
                category_average_without_trace=self._category_average_without_trace,
            ),
            "hph": _merge_category_covariances(
                hph, counts, trace_reference=teacher_averages, trace_balance=self._trace_balance,
                category_average_without_trace=self._category_average_without_trace,
            ),
            "hphp": _merge_category_covariances(
                hphp, counts, trace_reference=teacher_averages, trace_balance=self._trace_balance,
                category_average_without_trace=self._category_average_without_trace,
            ),
        }


def save_portable_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    model_name: str,
    ranks: Mapping[str, int | None],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a state-dict checkpoint that can be rebuilt without a pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "safe-svd-state-dict-v1",
        "model": model_name,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "ranks": dict(ranks),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)
    return path


def load_state_dict_payload(path: Path) -> Mapping[str, torch.Tensor]:
    """Read SAFE-SVD or legacy state-dict checkpoints."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} does not contain a state dictionary payload")
    state_dict = payload.get("state_dict", payload.get("model_state_dict", payload))
    if not isinstance(state_dict, Mapping) or not all(torch.is_tensor(value) for value in state_dict.values()):
        raise TypeError(f"{path} does not contain a tensor state dictionary")
    return state_dict  # type: ignore[return-value]


def rebuild_low_rank_topology(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> nn.Module:
    """Replace base Linear layers for ``<layer>.0/.1`` SAFE-SVD state keys."""
    targets: list[tuple[str, nn.Sequential]] = []
    for key, first_weight in state_dict.items():
        if not key.endswith(".0.weight"):
            continue
        layer_name = key.removesuffix(".0.weight")
        second_weight = state_dict.get(f"{layer_name}.1.weight")
        if not torch.is_tensor(second_weight):
            continue
        try:
            original = get_module_by_name(model, layer_name)
        except (AttributeError, KeyError):
            continue
        if not isinstance(original, nn.Linear):
            continue
        rank = first_weight.shape[0]
        replacement = nn.Sequential(
            nn.Linear(first_weight.shape[1], rank, bias=False),
            nn.Linear(rank, second_weight.shape[0], bias=f"{layer_name}.1.bias" in state_dict),
        )
        targets.append((layer_name, replacement))
    for name, replacement in targets:
        set_module_by_name(model, name, replacement)
    return model


def load_compressed_state(model: nn.Module, path: Path, *, strict: bool = True) -> nn.Module:
    """Rebuild low-rank layers and load a portable SAFE-SVD checkpoint."""
    state_dict = load_state_dict_payload(path)
    rebuild_low_rank_topology(model, state_dict)
    model.load_state_dict(state_dict, strict=strict)
    return model


def model_parameter_report(model: nn.Module) -> dict[str, int]:
    """Return total and Linear-only parameter counts for every evaluator."""
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "linear_parameters": sum(
            parameter.numel()
            for layer in model.modules()
            if isinstance(layer, nn.Linear)
            for parameter in layer.parameters()
        ),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
