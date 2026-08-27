"""Backend-neutral sequential MySVD compression orchestration."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn

from .factorization import (
    clean_input_factors,
    combined_input_factors_consuming,
    make_low_rank_linear,
)


CapturedInputs = Union[torch.Tensor, Sequence[torch.Tensor]]
CaptureInputs = Callable[[nn.Module, str], CapturedInputs]
CapturePairs = Callable[[nn.Module, nn.Module, str], Iterable[tuple[torch.Tensor, torch.Tensor]]]
CaptureStatistics = Callable[[nn.Module, nn.Module, str], Mapping[str, torch.Tensor]]
GetModule = Callable[[nn.Module, str], nn.Module]
SetModule = Callable[[nn.Module, str, nn.Module], None]
Checkpoint = Callable[[nn.Module, set[str]], None]
Progress = Callable[[int, int, str], None]
ReleaseCachedInputs = Callable[[str], None]


@dataclass(frozen=True)
class CompressionCallbacks:
    """Operations supplied by a model adapter to the generic compressor."""

    capture_inputs: CaptureInputs
    get_module: GetModule
    set_module: SetModule
    capture_pairs: CapturePairs | None = None
    checkpoint: Checkpoint | None = None
    progress: Progress | None = None
    release_cached_inputs: ReleaseCachedInputs | None = None
    # Multi-physics backends need to aggregate teacher/student activations by
    # dataset category before factorisation.  Keeping this optional lets
    # Poseidon retain its memory-efficient streamed pair capture while VICON
    # and MPP reproduce their trace-balanced covariance objective.
    capture_statistics: CaptureStatistics | None = None


def _as_2d(inputs: torch.Tensor, in_features: int, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(inputs):
        raise TypeError("adapter input capture must return a tensor")
    if inputs.shape[-1] != in_features:
        raise ValueError(f"captured input has final dimension {inputs.shape[-1]}, expected {in_features}")
    return inputs.detach().reshape(-1, in_features).to(device=device, dtype=torch.float64)


def _capture_batches(inputs: CapturedInputs) -> tuple[torch.Tensor, ...]:
    if torch.is_tensor(inputs):
        return (inputs,)
    if not inputs:
        raise ValueError("adapter input capture returned no batches")
    return tuple(inputs)


@torch.no_grad()
def compress_sequentially(
    teacher: nn.Module,
    ranks: Mapping[str, int | None],
    callbacks: CompressionCallbacks,
    *,
    device: str | torch.device,
    alpha: float,
    left_whiteners: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    eps: float = 1e-6,
    checkpoint_every: int | None = None,
    initial_student: nn.Module | None = None,
    compressed_layers: set[str] | None = None,
) -> nn.Module:
    """Run MySVD_origin's teacher/student, layer-by-layer compression loop.

    The adapter capture function is intentionally invoked separately for the
    teacher and current student, matching the original algorithm's sequential
    distribution shift rather than reusing stale activations.
    """
    target = torch.device(device)
    teacher = teacher.to(target).eval()
    student = (initial_student if initial_student is not None else copy.deepcopy(teacher)).to(target).eval()
    done = set() if compressed_layers is None else set(compressed_layers)
    linear_names = [(name, layer) for name, layer in teacher.named_modules() if isinstance(layer, nn.Linear)]
    since_checkpoint = 0

    total_layers = len(linear_names)
    for layer_index, (name, teacher_linear) in enumerate(linear_names, start=1):
        if name in done:
            if callbacks.release_cached_inputs is not None:
                callbacks.release_cached_inputs(name)
            if callbacks.progress is not None:
                callbacks.progress(layer_index, total_layers, name)
            continue
        rank = ranks.get(name)
        if rank is None:
            if callbacks.release_cached_inputs is not None:
                callbacks.release_cached_inputs(name)
            done.add(name)
            if callbacks.progress is not None:
                callbacks.progress(layer_index, total_layers, name)
            continue
        rank = int(rank)
        in_features, out_features = teacher_linear.in_features, teacher_linear.out_features
        if rank < 1 or rank * (in_features + out_features) >= in_features * out_features:
            if callbacks.release_cached_inputs is not None:
                callbacks.release_cached_inputs(name)
            done.add(name)
            if callbacks.progress is not None:
                callbacks.progress(layer_index, total_layers, name)
            continue

        if callbacks.capture_statistics is not None:
            captured = callbacks.capture_statistics(teacher, student, name)
            required = {"hh", "hph", "hphp"}
            missing = required.difference(captured)
            if missing:
                raise KeyError(f"adapter statistics for {name} are missing: {sorted(missing)}")
            statistics = {
                key: _as_2d_covariance(captured[key], in_features, target, key)
                for key in required
            }
        else:
            statistics = {
                "hh": torch.zeros((in_features, in_features), device=target, dtype=torch.float64),
            }
            statistics["hphp"] = torch.zeros_like(statistics["hh"])
            statistics["hph"] = torch.zeros_like(statistics["hh"])
            if callbacks.capture_pairs is not None:
                paired_batches = callbacks.capture_pairs(teacher, student, name)
            else:
                teacher_batches = _capture_batches(callbacks.capture_inputs(teacher, name))
                student_batches = _capture_batches(callbacks.capture_inputs(student, name))
                if len(teacher_batches) != len(student_batches):
                    raise ValueError(f"teacher/student capture batch count differs for {name}")
                paired_batches = zip(teacher_batches, student_batches)
            for teacher_batch, student_batch in paired_batches:
                teacher_inputs = _as_2d(teacher_batch, in_features, target)
                student_inputs = _as_2d(student_batch, in_features, target)
                if teacher_inputs.shape[0] != student_inputs.shape[0]:
                    raise ValueError(f"teacher/student capture count differs for {name}")
                statistics["hh"].addmm_(teacher_inputs.T, teacher_inputs)
                statistics["hphp"].addmm_(student_inputs.T, student_inputs)
                statistics["hph"].addmm_(teacher_inputs.T, student_inputs)
                del teacher_inputs, student_inputs
        left = None if left_whiteners is None else left_whiteners.get(name)
        factors = combined_input_factors_consuming(
            teacher_linear.weight,
            statistics,
            rank,
            alpha=alpha,
            left_whitener=left,
            eps=eps,
        )
        current = callbacks.get_module(student, name)
        if not isinstance(current, nn.Linear):
            raise TypeError(f"{name} is no longer nn.Linear in the student model")
        callbacks.set_module(student, name, make_low_rank_linear(current, factors))
        done.add(name)
        since_checkpoint += 1
        if callbacks.checkpoint is not None and checkpoint_every and since_checkpoint >= checkpoint_every:
            callbacks.checkpoint(student, done)
            since_checkpoint = 0
        if callbacks.capture_statistics is None and callbacks.capture_pairs is None:
            del teacher_batches, student_batches
        del statistics, factors, current
        if target.type == "cuda":
            torch.cuda.empty_cache()
        if callbacks.progress is not None:
            callbacks.progress(layer_index, total_layers, name)

    if callbacks.checkpoint is not None:
        callbacks.checkpoint(student, done)
    return student.cpu()


def _as_2d_covariance(
    covariance: torch.Tensor,
    in_features: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Validate a model-specific covariance before consuming it in SVD."""
    if not torch.is_tensor(covariance):
        raise TypeError(f"adapter statistic {name} must be a tensor")
    if covariance.shape != (in_features, in_features):
        raise ValueError(
            f"adapter statistic {name} has shape {tuple(covariance.shape)}, "
            f"expected {(in_features, in_features)}"
        )
    return covariance.detach().to(device=device, dtype=torch.float64)


@torch.no_grad()
def compress_clean_input(
    model: nn.Module,
    ranks: Mapping[str, int | None],
    input_whiteners: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    callbacks: CompressionCallbacks,
    *,
    device: str | torch.device,
    left_whiteners: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> nn.Module:
    """Run the exact static clean-input route selected by ``alpha == 0``."""
    target = torch.device(device)
    model = model.to(target).eval()
    layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]
    total_layers = len(layers)
    for layer_index, (name, original) in enumerate(layers, start=1):
        rank = ranks.get(name)
        if rank is None:
            if callbacks.progress is not None:
                callbacks.progress(layer_index, total_layers, name)
            continue
        rank = int(rank)
        if rank < 1 or rank * (original.in_features + original.out_features) >= original.in_features * original.out_features:
            if callbacks.progress is not None:
                callbacks.progress(layer_index, total_layers, name)
            continue
        if name not in input_whiteners:
            raise KeyError(f"input whitening is missing layer {name}")
        left = None if left_whiteners is None else left_whiteners.get(name)
        factors = clean_input_factors(original.weight, input_whiteners[name], rank, left_whitener=left)
        current = callbacks.get_module(model, name)
        if not isinstance(current, nn.Linear):
            raise TypeError(f"{name} is no longer nn.Linear")
        callbacks.set_module(model, name, make_low_rank_linear(current, factors))
        if callbacks.progress is not None:
            callbacks.progress(layer_index, total_layers, name)
    return model.cpu()
