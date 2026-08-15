"""Global parameter-budget rank allocation for SAFE-SVD."""

import heapq
import itertools

import torch
import torch.nn as nn
from tqdm import tqdm

@torch.no_grad()
def get_global_greedy_rank_dict_v2(
    trainer,
    Ldict,
    profile_mat,
    rank_ratio,
    enable_pruning=False,
    dev="cuda",
):
    """Allocate ranks from the unmodified whitening objective."""
    return _get_global_greedy_rank_dict(
        trainer,
        Ldict,
        profile_mat,
        rank_ratio,
        enable_pruning=enable_pruning,
        dev=dev,
    )


@torch.no_grad()
def get_global_greedy_rank_dict_covariance_invariant(
    trainer,
    Ldict,
    profile_mat,
    rank_ratio,
    sample_counts,
    enable_pruning=False,
    dev="cuda",
):
    """Allocate ranks without letting per-layer covariance means skew scores.

    With an empirical covariance ``H.T @ H / N_l``, the corresponding right
    whitening factor is scaled by ``1 / sqrt(N_l)`` relative to the Gram-sum
    factor.  Consequently each squared singular value used by the global
    greedy allocator is scaled by ``1 / N_l``.  That is appropriate for a
    per-layer mean loss, but not for the existing global parameter-budget
    objective: layers with more flattened activation vectors would otherwise
    receive artificially smaller scores and lose rank.

    This selector multiplies every singular-value energy for layer ``l`` by
    its collected ``N_l`` before comparing layers.  It changes *only* global
    rank allocation; the covariance mean and the compression reconstruction
    remain unchanged.
    """
    return _get_global_greedy_rank_dict(
        trainer,
        Ldict,
        profile_mat,
        rank_ratio,
        energy_scales=sample_counts,
        enable_pruning=enable_pruning,
        dev=dev,
    )


@torch.no_grad()
def _get_global_greedy_rank_dict(
    trainer,
    Ldict,
    profile_mat,
    rank_ratio,
    energy_scales=None,
    enable_pruning=False,
    dev="cuda",
):
    device = torch.device(dev)
    model = trainer.model.to(device).eval()
    linears = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    total_original_params = 0
    layer_meta = {}
    singular_energies = {}

    print("Step 1: Calculating singular values and costs for all layers...")
    for name, linear in tqdm(linears):
        weight = linear.weight.data.double().to(device)
        out_c, in_c = weight.shape
        original_params = out_c * in_c
        cost_per_rank = out_c + in_c
        total_original_params += original_params

        right_whitener, _ = profile_mat[name]
        right_whitener = right_whitener.to(weight.device, weight.dtype)

        if Ldict is not None and name in Ldict:
            left_whitener = Ldict[name][0].to(weight.device, weight.dtype)
        else:
            left_whitener = torch.eye(out_c, device=weight.device, dtype=weight.dtype)

        m_star = (left_whitener @ weight) @ right_whitener
        singular_values = torch.linalg.svdvals(m_star)
        energy = singular_values.square().detach().cpu().tolist()
        if energy_scales is not None:
            if name not in energy_scales:
                raise KeyError(
                    f"Missing sample count for normalized covariance layer {name!r}."
                )
            scale = int(energy_scales[name])
            if scale <= 0:
                raise ValueError(
                    f"Sample count for layer {name!r} must be positive, got {scale}."
                )
            energy = [value * scale for value in energy]

        # 仅保留严格比原矩阵省参数的低秩表示。
        max_compressed_rank = min(
            len(energy),
            (original_params - 1) // cost_per_rank,
        )

        layer_meta[name] = {
            "original": original_params,
            "cost_per_rank": cost_per_rank,
            "max_compressed_rank": max_compressed_rank,
        }
        singular_energies[name] = energy

    target_budget = int(total_original_params * rank_ratio)

    # rank_counts[name]:
    #   None -> 保留原始 full-rank 矩阵
    #   0    -> 剪枝（仅 enable_pruning=True 时可出现）
    #   k>0  -> 使用 rank-k 低秩分解
    rank_counts = {}
    greedy_spent = 0

    for name, _ in linears:
        meta = layer_meta[name]

        if enable_pruning:
            rank_counts[name] = 0
        elif meta["max_compressed_rank"] == 0:
            # 连 rank 1 都不省参数，只能保持 full-rank。
            rank_counts[name] = None
            greedy_spent += meta["original"]
        else:
            # 非 pruning 模式：每层至少 rank 1。
            rank_counts[name] = 1
            greedy_spent += meta["cost_per_rank"]

    if greedy_spent > target_budget:
        raise ValueError(
            f"Budget {target_budget} is insufficient for mandatory base layers "
            f"(requires {greedy_spent})."
        )

    # (-value, tie_breaker, candidate_type, layer_name, expected_rank, extra_cost)
    candidate_heap = []
    counter = itertools.count()

    def push_available_candidates(name):
        """为该层当前 rank 加入下一个低秩候选和 full-rank transition 候选。"""
        current_rank = rank_counts[name]
        if current_rank is None:
            return

        meta = layer_meta[name]
        energy = singular_energies[name]
        cost_per_rank = meta["cost_per_rank"]

        # 购买下一个连续的奇异值：rank k -> rank k+1。
        if current_rank < meta["max_compressed_rank"]:
            next_energy = energy[current_rank]
            heapq.heappush(
                candidate_heap,
                (
                    -(next_energy / cost_per_rank),
                    next(counter),
                    "rank",
                    name,
                    current_rank,
                    cost_per_rank,
                ),
            )

        # rank k -> full-rank：只计算尚未保留的 tail energy。
        additional_cost = meta["original"] - current_rank * cost_per_rank
        if current_rank < len(energy) and additional_cost > 0:
            tail_energy = sum(energy[current_rank:])
            heapq.heappush(
                candidate_heap,
                (
                    -(tail_energy / additional_cost),
                    next(counter),
                    "full",
                    name,
                    current_rank,
                    additional_cost,
                ),
            )

    for name, _ in linears:
        push_available_candidates(name)

    print(f"Step 2: Dynamic greedy allocation... (Budget: {target_budget} params)")

    while candidate_heap:
        neg_value, _, candidate_type, name, expected_rank, extra_cost = heapq.heappop(
            candidate_heap
        )

        # 该候选在同层状态变化后失效，直接跳过。
        if rank_counts[name] != expected_rank:
            continue

        # 参数量只会增加；当前候选放不下，以后也不会放得下。
        if greedy_spent + extra_cost > target_budget:
            continue

        greedy_spent += extra_cost

        if candidate_type == "rank":
            rank_counts[name] += 1
            push_available_candidates(name)
        else:
            rank_counts[name] = None

    print(
        f"Allocation done! Actual params: {greedy_spent} / "
        f"{total_original_params} "
        f"({greedy_spent / total_original_params * 100:.2f}%)"
    )
    return rank_counts
