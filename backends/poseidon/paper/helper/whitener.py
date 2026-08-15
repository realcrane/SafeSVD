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

# Poseidon / ScOT 依赖（仅用于取数据与构建 trainer；不做任何 whitening/profiling）
from poseidon_main.scOT.inference import get_trainer, rollout
from poseidon_main.scOT.problems.base import get_dataset

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# Profiling Origin Input
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io, os, logging
from transformers.utils import logging as hf_logging


def cholesky_psd_with_jitter(G: torch.Tensor, eps_min: float = 1e-6) -> torch.Tensor:
    """对称(半)正定矩阵 G 做 Cholesky，必要时加抖动；返回下三角 L，使 G ≈ L L^T。"""
    G = (G + G.T) * 0.5
    try:
        return torch.linalg.cholesky(G)
    except Exception:
        try:
            eigs = torch.linalg.eigvalsh(G)
            jitter = (-eigs.min().item() + eps_min) if eigs.min().item() < 0 else eps_min
            G = G + jitter * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
            return torch.linalg.cholesky(G)
        except Exception:
            eps_min = 1e-4
            eigs = torch.linalg.eigvalsh(G)
            jitter = (-eigs.min().item() + eps_min) if eigs.min().item() < 0 else eps_min
            G = G + jitter * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
            return torch.linalg.cholesky(G)

def empirical_cov(H):
    # H: (d, N)
    N = H.shape[1]
    return (H @ H.T) / max(N, 1)

def chol_psd(A, eps=1e-4):
    """Numerically stable Cholesky for (near) PSD matrices: returns lower-tri L with A ≈ L L^T."""
    A = (A + A.T) * 0.5
    n = A.shape[0]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    return torch.linalg.cholesky(A + eps * I)  # lower-tri

# def svd_whitener(Sigma, eps=1e-6):
#     """
#     SVD 白化：Sigma = U diag(s) U^T
#     返回 L, LinvT，使得  L L^T ≈ Sigma + eps I，且 LinvT = L^{-T}
#     适用于秩亏/近奇异情形，比 Cholesky 更稳。
#     """
#     # 用特征分解（对称正定/半正定更稳），等价于 SVD 的对称特化
#     s, U = torch.linalg.eigh((Sigma + Sigma.T) * 0.5)
#     s = torch.clamp(s, min=0)
#     L      = U @ torch.diag(torch.sqrt(s + eps))                 # L L^T ≈ Sigma + eps I
#     LinvT  = U @ torch.diag(1.0 / torch.sqrt(s + eps))           # L^{-T}
#     return L, LinvT
def svd_whitener(Sigma, eps=1e-6):
    s, U = torch.linalg.eigh((Sigma + Sigma.T) * 0.5)
    s = torch.clamp(s, min=0)
    sqrt_s = torch.sqrt(s + eps)
    L     = U * sqrt_s.unsqueeze(0)         
    LinvT = U * (1.0 / sqrt_s).unsqueeze(0)
    return L, LinvT

# --------- 通用白化→截断→去白化（兼容 Cholesky / SVD 白化） ---------
def whiten_svd_truncate_dewhite(W, L, LinvT, energy_keep):
    # 设备和 dtype 对齐
    L = L.to(W.device, dtype=W.dtype)

    out_c, in_c = W.shape
    if L.shape != (in_c, in_c):
        raise ValueError(f"L 的形状 {tuple(L.shape)} 应为 (in_features,in_features)=({in_c},{in_c})")

    # 1) 白化权重
    Ww = W @ L

    # 2) SVD 并按能量选秩
    U, S, VT = torch.linalg.svd(Ww, full_matrices=False)
    max_rank = min(out_c, in_c)
    sigma_sq = S.pow(2)
    cum = torch.cumsum(sigma_sq, dim=0)
    k = torch.searchsorted(cum, sigma_sq.sum() * energy_keep).item() + 1
    k = max(1, min(k, max_rank))
    U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
    sqrtS = torch.sqrt(S_k)
    U_k = (U_k * sqrtS.unsqueeze(0)).float()
    V_k = (sqrtS.unsqueeze(1) * VT_k).float()

    # 3) 去白化（解 (L^T) X^T = V_k^T）
    V_k = (V_k @ LinvT.T.to(V_k.dtype)).to(W.dtype)
    return U_k, V_k, k