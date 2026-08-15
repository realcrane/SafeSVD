#coding:utf8
import os
import sys
import argparse
import json
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

import torch
import torch.nn as nn
from tqdm import tqdm

@torch.no_grad()
def get_global_greedy_rank_dict(trainer, Ldict, Rdict, rank_ratio, dev, eps_whiten=1e-6):
    device = torch.device(dev)
    model = trainer.model.to(device).eval()
    linears = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    
    global_sv_pool = []
    total_original_params = 0
    layer_meta = {}
    
    print("Step 1: Calculating singular values and costs for all layers...")
    for idx, (name, lin) in enumerate(linears):
        W = lin.weight.data.double().to(device)
        out_c, in_c = W.shape
        original_params = out_c * in_c
        total_original_params += original_params
        cost_per_rank = out_c + in_c
        
        # 超过此 rank 压缩不再节省参数
        max_rank = min(out_c, in_c)
        layer_meta[name] = (out_c, in_c, max_rank, original_params)
        
        Lr_clean, _ = Rdict[name]
        Lr_clean = Lr_clean.to(W.device, W.dtype)
        
        Ll = Ldict[name][0].to(W.device, W.dtype)

            
        M_clean = (Ll @ W) @ Lr_clean
        S = torch.linalg.svdvals(M_clean)
        
        for i in range(len(S)):
            global_sv_pool.append((
                (S[i].item() ** 2) / cost_per_rank,
                name,
                i,
                cost_per_rank,
            ))

    global_sv_pool.sort(key=lambda x: x[0], reverse=True)
    target_budget = int(total_original_params * rank_ratio)
    
    # =============================================
    # Step 2: 第一轮贪心（out+in 记账）
    # =============================================
    rank_counts = {name: 0 for name, _ in linears}
    greedy_spent = 0
    
    print(f"Step 2: Greedy allocation... (Budget: {target_budget} params)")
    for score, name, sv_idx, cost in global_sv_pool:
        if sv_idx != rank_counts[name]:
            continue
        if greedy_spent + cost <= target_budget:
            rank_counts[name] += 1
            greedy_spent += cost

    # =============================================
    # Step 3: 计算真实参数量 + 补偿轮
    # =============================================
    def get_layer_cost(name, k):
        out_c, in_c, _, original = layer_meta[name]
        if k == 0:
            return original          # 不压缩，保留原始
        compressed = k * (out_c + in_c)
        return min(compressed, original)
    
    actual_params = sum(get_layer_cost(n, rank_counts[n]) for n in rank_counts)
    
    # 补偿轮：真实参数 < 预算，用省出来的余量继续买
    if actual_params < target_budget:
        remaining = [item for item in global_sv_pool 
                     if item[2] >= rank_counts[item[1]]           # 还没买的
                     and rank_counts[item[1]] < layer_meta[item[1]][2]]  # 还没到 max_useful
        remaining.sort(key=lambda x: x[0], reverse=True)
        
        for score, name, sv_idx, cost in remaining:
            if sv_idx != rank_counts[name]:
                continue
            old_total = get_layer_cost(name, rank_counts[name])
            new_total = get_layer_cost(name, rank_counts[name] + 1)
            marginal = new_total - old_total
            
            if actual_params + marginal <= target_budget:
                rank_counts[name] += 1
                actual_params += marginal

    # =============================================
    # Step 4: 构建 rank_dict
    # =============================================
    rank_dict = {}
    for name, lin in linears:
        out_c, in_c, max_useful, original = layer_meta[name]
        k = rank_counts[name]
        if k == 0:
            rank_dict[name] = 1
        elif k * (out_c + in_c) >= original:
            rank_dict[name] = None
        else:
            rank_dict[name] = k
            
    print(f"Allocation done! Actual params: {actual_params} / {total_original_params} "
          f"({actual_params/total_original_params*100:.2f}%)")
    
    return rank_dict



def energy_base_rank_selector(U, S, VT, max_rank, energy_keep):
    """Select rank k such that cumulative energy >= energy_keep."""
    sigma_sq = S.pow(2)
    cum = torch.cumsum(sigma_sq, dim=0)
    total = sigma_sq.sum()
    k = torch.searchsorted(cum, total * energy_keep).item() + 1
    k = max(1, min(k, max_rank))
    U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
    sqrtS = torch.sqrt(S_k)
    U_k = (U_k * sqrtS.unsqueeze(0)).float()
    V_k = (sqrtS.unsqueeze(1) * VT_k).float()
    return U_k, V_k, k