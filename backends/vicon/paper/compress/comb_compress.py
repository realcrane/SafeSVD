# =====================================================================
# Core MySVD compression (fixed rank) — adapted for VICON
# =====================================================================
# 【改动说明】
#   1. mysvd_comb 新增 use_trace_align 超参，传递给 collect_covariances_vicon
#   2. cached_calib_batches 存储 (dataset_type, batch) 二元组
# =====================================================================
import os
import copy
import json
import math
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from VICON.src import utils
from VICON.src.dataset import all_datasets
import VICON.src.models as models
from VICON.src.trainer import Trainer
from MySVD.convert_attn import convert_attn_to_linear
from MySVD.helper.fisher_collector import *
from MySVD.helper.input_collector import *
from MySVD.helper.rank_selector import *
from MySVD.helper.whitener import *

@torch.no_grad()
def mysvd_comb(
    trainer: Trainer,
    calib_loaders: dict,
    dev: str = "cuda",
    ranks_json_path: Optional[str] = None,
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    eps_whiten: float = 1e-6,
    max_calib_batches: int = None,
    alpha: float = 0.5,
    use_trace_align: bool = True,     # ← 新增超参
):
    device = torch.device(dev)
    model_teacher: nn.Module = trainer.model.to(device).eval()
    model_student = copy.deepcopy(model_teacher)

    loaded_ranks = {}
    if ranks_json_path is not None and os.path.exists(ranks_json_path):
        with open(ranks_json_path, "r") as f:
            loaded_ranks = json.load(f) or {}
        print(f"[fixed-rank] loaded {len(loaded_ranks)} ranks from {ranks_json_path}")

    linear_names = [(name, m) for name, m in model_teacher.named_modules()
                    if isinstance(m, nn.Linear)]

    align_tag = "ON" if use_trace_align else "OFF"
    print(f"Start MySVD compression on {len(linear_names)} linear layers (trace_align={align_tag})...")

    # 缓存时保留 dataset_type 信息
    print("Pre-fetching raw calibration batches (CPU) ...")
    cached_calib_batches = []
    for dataset_type, loader in calib_loaders.items():
        for batch_cnt, batch in enumerate(loader):
            if max_calib_batches is not None and batch_cnt >= max_calib_batches:
                break
            cached_calib_batches.append((dataset_type, batch))
    print(f"Successfully cached {len(cached_calib_batches)} raw batches.")

    for j, (name, lin) in enumerate(tqdm(linear_names)):
        if name in loaded_ranks:
            k = loaded_ranks[name]
            if k is None:
                print(f"[skip] {name}: forced skip by ranks.json")
                continue
            k = int(k)
        else: continue

        in_c, out_c = lin.in_features, lin.out_features
        num_W  = out_c * in_c
        num_UV = out_c * k + k * in_c
        if num_UV >= num_W:
            print(f"[{j:02d}] {name}: Skipped (Full Rank better) k={k}")
            continue
        if not isinstance(lin, nn.Linear): continue

        print("Collect Covariances")
        sigma_HH, sigma_HHp, sigma_HpHp = collect_covariances_vicon(
            trainer=trainer,
            model_teacher=model_teacher,
            model_student=model_student,
            layer_name=name,
            cached_batches=cached_calib_batches,
            device=device,
            use_trace_align=use_trace_align,   # ← 传递超参
        )

        W = lin.weight.data.double().to(device)
        max_rank = min(out_c, in_c)
        print(f"Compressing (alpha={alpha}) ...")

        sigma_comb = alpha * sigma_HpHp + (1 - alpha) * sigma_HH

        Lr, Lr_invT = svd_whitener(sigma_comb, eps=eps_whiten)
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        if Ldict is not None and (name in Ldict):
            Ll, Ll_invT = Ldict[name]
            Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)
        else:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = torch.eye(out_c, device=W.device, dtype=W.dtype)

        C_comb_right = ((1 - alpha) * sigma_HH + alpha * sigma_HHp).to(W.dtype) @ Lr_invT
        Mstar_Rcomb = (Ll @ W) @ C_comb_right

        U, S, VT = torch.linalg.svd(Mstar_Rcomb, full_matrices=False)

        U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
        sqrtS = torch.sqrt(S_k)
        U_k = (U_k * sqrtS.unsqueeze(0)).float()
        V_k = (sqrtS.unsqueeze(1) * VT_k).float()

        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(W.dtype)

        seq = nn.Sequential(
            nn.Linear(in_c, k, bias=False),
            nn.Linear(k, out_c, bias=(lin.bias is not None)),
        ).to(device)

        if lin.bias is not None:
            seq[1].bias.data.copy_(lin.bias.data)
        seq[0].weight.data.copy_(V_k)
        seq[1].weight.data.copy_(U_k)

        if '.' in name:
            parent, attr = name.rsplit('.', 1)
            parent_mod = model_student.get_submodule(parent)
        else:
            parent_mod = model_student
            attr = name
        setattr(parent_mod, attr, seq)

        print(f"[{j:02d}] {name}: Compressed to rank {k}/{max_rank}")

    model_student = model_student.cpu()
    print("MySVD compression complete.")
    return model_student