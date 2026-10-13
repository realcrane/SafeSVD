"""Shared compression loop with model-specific input capture in helper/.

The caller supplies an already-loaded model whose target projections are
nn.Linear (convert VICON attention before calling). Ranks and Fisher factors
are prepared upstream by safe_svd.main.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from safe_svd.helper.collector.collector import CaptureContext, CaptureFunction, collect_statistics
from safe_svd.helper.collector.mpp_collector import collect_mpp
from safe_svd.helper.collector.poseidon.poseidon_collector import collect_poseidon
from safe_svd.helper.collector.vicon_collector import collect_vicon
from safe_svd.helper.whitener import right_whitener


def replace_linear(
    student: nn.Module,
    name: str,
    original: nn.Linear,
    matrix: Tensor,
    rank: int,
    left_inverse: Tensor,
    right_inverse_transpose: Tensor,
) -> None:
    """Shared SVD, inverse transforms, bias preservation, and replacement."""
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    root = singular_values[:rank].sqrt()
    left_weight = left_inverse @ (u[:, :rank] * root.unsqueeze(0))
    right_weight = (root.unsqueeze(1) * vh[:rank]) @ right_inverse_transpose.T
    replacement = nn.Sequential(
        nn.Linear(original.in_features, rank, bias=False),
        nn.Linear(rank, original.out_features, bias=original.bias is not None),
    ).to(device=original.weight.device, dtype=original.weight.dtype)
    replacement[0].weight.copy_(right_weight)
    replacement[1].weight.copy_(left_weight)
    if original.bias is not None:
        replacement[1].bias.copy_(original.bias)
    parent_name, _, attribute = name.rpartition(".")
    parent = student.get_submodule(parent_name) if parent_name else student
    setattr(parent, attribute, replacement)


@torch.no_grad()
def compress(
    model: nn.Module,
    calibration_data: Any,
    ranks: Mapping[str, int | None],
    *,
    model_type: str,
    device: str | torch.device = "cuda",
    left_metrics: Mapping[str, tuple[Tensor, Tensor]] | None = None,
    mode: str = "combined",
    input_metrics: Mapping[str, tuple[Tensor, Tensor]] | None = None,
    alpha: float = 0.7,
    aggregation: str | None = None,
    trace_balance: bool = False,
    whitening: str = "spectral",
    eps: float = 1e-6,
    eps_mode: str = "absolute",
    compute_dtype: torch.dtype = torch.float64,
    capture_fn: CaptureFunction | None = None,
    route_options: Mapping[str, Any] | None = None,
    checkpoint_fn: Callable[[nn.Module, set[str]], None] | None = None,
    checkpoint_every: int = 10,
    initial_student: nn.Module | None = None,
    completed_layers: Iterable[str] = (),
    progress: bool = True,
) -> nn.Module:
    """Compress an already-loaded model; return the student on CPU.

    left_metrics entries are (L, L^{-1}); input_metrics are (R, R^{-T}).
    Missing/None ranks keep a layer; rank 0 zeros its weight and keeps bias.
    combined captures current teacher/student inputs; clean uses input_metrics.
    Supply initial_student/completed_layers for resume. A checkpoint callback
    receives the live student and must restore its device if it moves it.
    progress shows all target Linear layers, including retained layers.
    """
    routes = {"mpp": collect_mpp, "vicon": collect_vicon, "poseidon": collect_poseidon}
    if model_type not in routes:
        raise ValueError("model_type must be mpp, vicon, or poseidon")
    if mode not in {"combined", "clean"} or not 0 <= alpha <= 1:
        raise ValueError("Invalid mode or alpha")
    aggregation = aggregation or ("category_mean" if model_type == "vicon" else "sample_mean")
    if aggregation not in {"sample_mean", "category_mean"}:
        raise ValueError("Invalid aggregation")
    if whitening not in {"spectral", "fisher_cholesky"}:
        raise ValueError("Invalid whitening method")
    if eps_mode not in {"absolute", "relative", "sum_equivalent"} or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid epsilon configuration")
    if compute_dtype not in {torch.float32, torch.float64} or checkpoint_every < 1:
        raise ValueError("Invalid compute dtype or checkpoint interval")
    if mode == "clean" and input_metrics is None:
        raise ValueError("clean mode requires input_metrics")
    if initial_student is model:
        raise ValueError("Teacher and student must be distinct models")
    done = set(completed_layers)
    if done and initial_student is None:
        raise ValueError("Resume requires initial_student")
    layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]
    names = {name for name, _ in layers}
    if "" in names:
        raise ValueError("Wrap a standalone Linear in a named container before compression")
    if (set(ranks) | done) - names:
        raise ValueError("Ranks/progress refer to unknown Linear layers")
    for name, layer in layers:
        rank = ranks.get(name)
        if rank is not None and (isinstance(rank, bool) or int(rank) != rank or not 0 <= rank <= min(layer.weight.shape)):
            raise ValueError(f"Invalid rank for {name}: {rank}")

    target = torch.device(device)
    teacher = model.to(target).eval()
    student = (copy.deepcopy(teacher) if initial_student is None else initial_student).to(target).eval()
    context = CaptureContext(teacher, student, calibration_data, target, route_options or {}, ranks=ranks)
    collector = capture_fn if capture_fn is not None else routes[model_type]
    since_checkpoint = 0
    bar = tqdm(layers, desc=f"Compression ({model_type})", unit="layer",
               dynamic_ncols=True, disable=not progress)
    try:
        for name, original in bar:
            if progress:
                bar.set_postfix(layer=name, rank=ranks.get(name) if ranks.get(name) is not None else "keep")
            if name in done:
                context.cache.pop(name, None)
                continue
            rank = ranks.get(name)
            if rank is None or rank * (original.in_features + original.out_features) >= original.weight.numel():
                context.cache.pop(name, None)
            elif rank == 0:
                current = student.get_submodule(name)
                if not isinstance(current, nn.Linear):
                    raise TypeError(f"Expected an unprocessed Linear at {name}")
                current.weight.zero_()
                if original.bias is not None:
                    current.bias.copy_(original.bias)
                context.cache.pop(name, None)
            else:
                rank = int(rank)
                if mode == "clean":
                    right, right_inverse_transpose = input_metrics[name]
                    right = right.to(device=target, dtype=compute_dtype)
                    right_inverse_transpose = right_inverse_transpose.to(device=target, dtype=compute_dtype)
                else:
                    hh, hhp, hphp, count = collect_statistics(
                        collector(context, name), original.in_features,
                        device=target, dtype=compute_dtype,
                        aggregation=aggregation, trace_balance=trace_balance,
                    )
                    right_metric = hphp.mul_(alpha).add_(hh, alpha=1 - alpha)
                    del hphp
                    right, right_inverse_transpose = right_whitener(
                        right_metric,
                        method=whitening, eps=eps, eps_mode=eps_mode, sample_count=count,
                    )
                    del right_metric, right
                    cross = hh.mul_(1 - alpha).add_(hhp, alpha=alpha)
                    del hh, hhp
                    cross = cross @ right_inverse_transpose
                # Keep decomposition workspaces out of the capture forward.
                # Do not allocate a large identity when a Fisher factor exists.
                weight = original.weight.detach().to(dtype=compute_dtype)
                metric = (left_metrics or {}).get(name)
                if metric is None:
                    left = torch.eye(original.out_features, device=target, dtype=compute_dtype)
                    left_inverse = left
                else:
                    left, left_inverse = (value.to(device=target, dtype=compute_dtype) for value in metric)
                matrix = (left @ weight) @ (right if mode == "clean" else cross)
                if mode != "clean":
                    del cross
                else:
                    del right
                del weight, left
                replace_linear(student, name, original, matrix, rank, left_inverse, right_inverse_transpose)
                context.cache.pop(name, None)
                del left_inverse, right_inverse_transpose, matrix
            done.add(name)
            since_checkpoint += 1
            if checkpoint_fn is not None and since_checkpoint >= checkpoint_every:
                checkpoint_fn(student, set(done))
                since_checkpoint = 0
            if target.type == "cuda":
                torch.cuda.empty_cache()
        if checkpoint_fn is not None and since_checkpoint:
            checkpoint_fn(student, set(done))
        return student.cpu()
    finally:
        bar.close()
        context.cache.clear()
        context.runtime.clear()
