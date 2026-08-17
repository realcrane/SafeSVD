"""Budget-constrained rank allocation for linear layers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn


def _linear_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]


@torch.no_grad()
def allocate_greedy_ranks(
    model: nn.Module,
    left_whiteners: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None,
    input_whiteners: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    rank_ratio: float,
    device: str | torch.device,
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
        pool.extend((float(value.square() / cost), name, index, cost) for index, value in enumerate(singular_values))

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
