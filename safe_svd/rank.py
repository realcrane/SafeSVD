"""Budget-constrained rank allocation for linear layers."""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Mapping
from typing import Literal

import torch
import torch.nn as nn


RankSelector = Literal["legacy", "safe"]
RANK_SELECTORS: tuple[RankSelector, ...] = ("legacy", "safe")


def _energy_scale(name, energy_scales):
    if energy_scales is None:
        return 1
    if name not in energy_scales:
        raise KeyError(f"Missing input sample count for {name}")
    count = energy_scales[name]
    if isinstance(count, bool) or int(count) != count or count <= 0:
        raise ValueError(f"Input sample count must be a positive integer for {name}")
    return int(count)


def _linear_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]


@torch.no_grad()
def allocate_greedy_ranks(
    model: nn.Module,
    left_whiteners: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None,
    input_whiteners: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    rank_ratio: float,
    device: str | torch.device,
    *,
    energy_scales: Mapping[str, int] | None = None,
) -> dict[str, int | None]:
    """Allocate ranks exactly as MySVD_origin's global greedy selector.

    ``None`` means retaining the original linear layer; positive values are
    the rank of the two-factor replacement.  A layer with no selected singular
    value receives rank one, matching the legacy implementation.
    """
    if not 0 < rank_ratio <= 1:
        raise ValueError("rank_ratio must be in (0, 1]")
    target = torch.device(device)
    model = model.to(target).eval()
    layers = _linear_layers(model)
    missing = [name for name, _ in layers if name not in input_whiteners]
    if missing:
        raise KeyError(f"input whitening is missing {len(missing)} linear layers; first: {missing[0]}")

    pool: list[tuple[float, str, int, int]] = []
    metadata: dict[str, tuple[int, int, int, int]] = {}
    total_original = 0
    for name, linear in layers:
        weight = linear.weight.detach().to(target, dtype=torch.float64)
        out_features, in_features = weight.shape
        original = out_features * in_features
        cost = out_features + in_features
        metadata[name] = (out_features, in_features, min(out_features, in_features), original)
        total_original += original

        right = input_whiteners[name][0].to(target, dtype=weight.dtype)
        if left_whiteners is not None and name in left_whiteners:
            left = left_whiteners[name][0].to(target, dtype=weight.dtype)
        else:
            left = torch.eye(out_features, device=target, dtype=weight.dtype)
        singular_values = torch.linalg.svdvals((left @ weight) @ right)
        scale = _energy_scale(name, energy_scales)
        # The research selector squares Python floats. Tensor.square() can
        # round a near-tied score one ULP differently and change the winning
        # layer at the budget boundary. Preserve its arithmetic and stable
        # tie ordering; scale is one for MPP/VICON.
        pool.extend(((value.item() ** 2) * scale / cost, name, index, cost)
                    for index, value in enumerate(singular_values))

    pool.sort(key=lambda item: item[0], reverse=True)
    budget = int(total_original * rank_ratio)
    counts = {name: 0 for name, _ in layers}
    spent = 0
    for _, name, singular_index, cost in pool:
        if singular_index == counts[name] and spent + cost <= budget:
            counts[name] += 1
            spent += cost

    def layer_cost(name: str, rank: int) -> int:
        out_features, in_features, _, original = metadata[name]
        return original if rank == 0 else min(rank * (out_features + in_features), original)

    actual = sum(layer_cost(name, rank) for name, rank in counts.items())
    if actual < budget:
        for _, name, singular_index, _ in pool:
            if singular_index != counts[name] or counts[name] >= metadata[name][2]:
                continue
            marginal = layer_cost(name, counts[name] + 1) - layer_cost(name, counts[name])
            if actual + marginal <= budget:
                counts[name] += 1
                actual += marginal

    ranks: dict[str, int | None] = {}
    for name, _ in layers:
        out_features, in_features, _, original = metadata[name]
        rank = counts[name]
        ranks[name] = 1 if rank == 0 else (None if rank * (out_features + in_features) >= original else rank)
    return ranks


@torch.no_grad()
def allocate_safe_greedy_ranks(
    model: nn.Module,
    left_whiteners: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None,
    input_whiteners: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    rank_ratio: float,
    device: str | torch.device,
    *,
    enable_pruning: bool = False,
    energy_scales: Mapping[str, int] | None = None,
) -> dict[str, int | None]:
    """Allocate ranks with budget-consistent SAFE-SVD dynamic greedy search.

    Every non-pruned, compressible layer starts at rank one and that mandatory
    cost is charged before optional singular directions are selected.  Integer
    ranks are restricted to representations that strictly save parameters.
    Retaining an original layer is represented by an explicit transition whose
    value is its remaining singular-value energy divided by its marginal cost.

    ``None`` retains the original linear layer.  Zero is returned only when
    ``enable_pruning`` is true; the public SAFE-SVD pipeline currently keeps
    pruning disabled.
    """
    if not 0 < rank_ratio <= 1:
        raise ValueError("rank_ratio must be in (0, 1]")
    target = torch.device(device)
    model = model.to(target).eval()
    layers = _linear_layers(model)
    missing = [name for name, _ in layers if name not in input_whiteners]
    if missing:
        raise KeyError(f"input whitening is missing {len(missing)} linear layers; first: {missing[0]}")

    metadata: dict[str, tuple[int, int, int]] = {}
    energies: dict[str, list[float]] = {}
    total_original = 0
    for name, linear in layers:
        weight = linear.weight.detach().to(target, dtype=torch.float64)
        out_features, in_features = weight.shape
        original = out_features * in_features
        cost = out_features + in_features
        total_original += original

        right = input_whiteners[name][0].to(target, dtype=weight.dtype)
        if left_whiteners is not None and name in left_whiteners:
            left = left_whiteners[name][0].to(target, dtype=weight.dtype)
        else:
            left = torch.eye(out_features, device=target, dtype=weight.dtype)
        singular_values = torch.linalg.svdvals((left @ weight) @ right)
        energies[name] = (singular_values.square() * _energy_scale(name, energy_scales)).detach().cpu().tolist()
        max_compressed_rank = min(
            len(energies[name]),
            (original - 1) // cost,
        )
        metadata[name] = (original, cost, max_compressed_rank)

    budget = int(total_original * rank_ratio)
    ranks: dict[str, int | None] = {}
    spent = 0
    for name, _ in layers:
        original, cost, max_compressed_rank = metadata[name]
        if enable_pruning:
            ranks[name] = 0
        elif max_compressed_rank == 0:
            ranks[name] = None
            spent += original
        else:
            ranks[name] = 1
            spent += cost

    if spent > budget:
        raise ValueError(
            f"rank budget {budget} is insufficient for mandatory base layers "
            f"(requires {spent})"
        )

    # (-value, tie_breaker, candidate_type, layer_name, expected_rank, cost)
    candidates: list[tuple[float, int, str, str, int, int]] = []
    tie_breaker = itertools.count()

    def push_available(name: str) -> None:
        current = ranks[name]
        if current is None:
            return
        original, cost, max_compressed_rank = metadata[name]
        layer_energy = energies[name]
        if current < max_compressed_rank:
            heapq.heappush(
                candidates,
                (
                    -(layer_energy[current] / cost),
                    next(tie_breaker),
                    "rank",
                    name,
                    current,
                    cost,
                ),
            )
        full_cost = original - current * cost
        if current < len(layer_energy) and full_cost > 0:
            heapq.heappush(
                candidates,
                (
                    -(sum(layer_energy[current:]) / full_cost),
                    next(tie_breaker),
                    "full",
                    name,
                    current,
                    full_cost,
                ),
            )

    for name, _ in layers:
        push_available(name)

    while candidates:
        _, _, candidate_type, name, expected_rank, marginal_cost = heapq.heappop(candidates)
        if ranks[name] != expected_rank:
            continue
        if spent + marginal_cost > budget:
            continue
        spent += marginal_cost
        if candidate_type == "rank":
            ranks[name] = expected_rank + 1
            push_available(name)
        else:
            ranks[name] = None
    return ranks


def allocate_ranks(
    model: nn.Module,
    left_whiteners: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None,
    input_whiteners: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    rank_ratio: float,
    device: str | torch.device,
    *,
    selector: RankSelector = "safe",
    energy_scales: Mapping[str, int] | None = None,
) -> dict[str, int | None]:
    """Dispatch to a named, reproducible rank-allocation algorithm."""
    if selector == "legacy":
        return allocate_greedy_ranks(model, left_whiteners, input_whiteners, rank_ratio, device, energy_scales=energy_scales)
    if selector == "safe":
        return allocate_safe_greedy_ranks(model, left_whiteners, input_whiteners, rank_ratio, device, energy_scales=energy_scales)
    raise ValueError(f"unknown rank selector {selector!r}; expected one of: {', '.join(RANK_SELECTORS)}")
