# =====================================================================
# Core MySVD compression (fixed rank) — adapted for VICON
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
def mysvd_compress_fixedrank_vicon(
    trainer: Trainer,
    calib_loaders: dict,
    dev: str = "cuda",
    ranks_json_path: Optional[str] = None,
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    eps_whiten: float = 1e-6,
    max_calib_batches: int = None,
):
    """
    MySVD compression with fixed per-layer ranks (from JSON) or a global fixed rank.
    
    This is the VICON port of svd_compress_sliced_pairwise_fixedrank.
    Key difference: instead of Poseidon's rollout-based input_catch, we use
    VICON's trainer._model_forward with hooks to collect activations.
    """
    device = torch.device(dev)
    model_teacher: nn.Module = trainer.model.to(device).eval()
    model_student = copy.deepcopy(model_teacher)

    # --- Load per-layer ranks ---
    loaded_ranks = {}
    if ranks_json_path is not None and os.path.exists(ranks_json_path):
        with open(ranks_json_path, "r") as f:
            loaded_ranks = json.load(f) or {}
        print(f"[fixed-rank] loaded {len(loaded_ranks)} ranks from {ranks_json_path}")

    linear_names = [(name, m) for name, m in model_teacher.named_modules()
                    if isinstance(m, nn.Linear)]
    print(f"Start MySVD compression on {len(linear_names)} linear layers...")

    # ================= 新增：提前缓存所有的 Calibration 数据 =================
    print("Pre-fetching raw calibration batches (CPU) to avoid redundant disk I/O...")
    cached_calib_batches = []
    for dataset_type, loader in calib_loaders.items():
        for batch_cnt, batch in enumerate(loader):
            if max_calib_batches is not None and batch_cnt >= max_calib_batches:
                break
            cached_calib_batches.append(batch) 
    print(f"Successfully cached {len(cached_calib_batches)} raw batches.")


    for j, (name, lin) in enumerate(tqdm(linear_names)):
        # --- Determine rank ---
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

        # --- Collect teacher H and student H' via VICON forward hooks ---
        print("Collect Covariances")
        sigma_HHp, sigma_HpHp = collect_covariances_vicon(
            trainer=trainer,
            model_teacher=model_teacher,
            model_student=model_student,
            layer_name=name,
            cached_batches=cached_calib_batches,
            device=device
        )

        W = lin.weight.data.double().to(device)
        max_rank = min(out_c, in_c)
        print("Compressing ...")
        # --- Right whitener from student input covariance ---
        Lr, Lr_invT = svd_whitener(sigma_HpHp, eps=eps_whiten)
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        # --- Left whitener from Fisher (Ldict) ---
        if Ldict is not None and (name in Ldict):
            Ll, Ll_invT = Ldict[name]
            Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)
        else:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = torch.eye(out_c, device=W.device, dtype=W.dtype)

        # --- SVD decomposition ---
        if j == 0:
            # First layer: direct connection (no cross-covariance)
            Ww = (Ll @ W) @ Lr
            U, S, VT = torch.linalg.svd(Ww, full_matrices=False)
        else:
            Mstar = (Ll @ W) @ (sigma_HHp.to(W.dtype) @ Lr_invT)
            U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)

        # --- Truncate ---
        U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
        sqrtS = torch.sqrt(S_k)
        U_k = (U_k * sqrtS.unsqueeze(0)).float()
        V_k = (sqrtS.unsqueeze(1) * VT_k).float()

        # --- De-whiten ---
        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(W.dtype)

        # --- Replace with low-rank Sequential ---
        seq = nn.Sequential(
            nn.Linear(in_c, k, bias=False),
            nn.Linear(k, out_c, bias=(lin.bias is not None)),
        ).to(device)

        if lin.bias is not None:
            seq[1].bias.data.copy_(lin.bias.data)
        seq[0].weight.data.copy_(V_k)
        seq[1].weight.data.copy_(U_k)

        # Write back to student model (VICON style)
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


@torch.no_grad()
def mysvd_clean_input(
    trainer: Trainer,
    dev: str = "cuda",
    ranks_json_path: Optional[str] = None,
    Rdict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
):
    """
    MySVD compression with fixed per-layer ranks (from JSON) or a global fixed rank.
    
    This is the VICON port of svd_compress_sliced_pairwise_fixedrank.
    Key difference: instead of Poseidon's rollout-based input_catch, we use
    VICON's trainer._model_forward with hooks to collect activations.
    """
    device = torch.device(dev)
    model = trainer.model.to(device).eval()

    # --- Load per-layer ranks ---
    loaded_ranks = {}
    if ranks_json_path is not None and os.path.exists(ranks_json_path):
        with open(ranks_json_path, "r") as f:
            loaded_ranks = json.load(f) or {}
        print(f"[fixed-rank] loaded {len(loaded_ranks)} ranks from {ranks_json_path}")

    linear_names = [(name, m) for name, m in model.named_modules()
                    if isinstance(m, nn.Linear)]
    print(f"Start MySVD compression on {len(linear_names)} linear layers...")

    for j, (name, lin) in enumerate(tqdm(linear_names)):
        # --- Determine rank ---
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

        W = lin.weight.data.double().to(device)
        max_rank = min(out_c, in_c)


        Ll, Ll_invT = Ldict[name]
        Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)

        Lr, Lr_invT = Rdict[name]
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        M = Ll @ W @ Lr
        U, S, VT = torch.linalg.svd(M, full_matrices=False)
        # --- Truncate ---
        U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
        sqrtS = torch.sqrt(S_k)
        U_k = (U_k * sqrtS.unsqueeze(0)).float()
        V_k = (sqrtS.unsqueeze(1) * VT_k).float()

        # --- De-whiten ---
        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(W.dtype)

        # --- Replace with low-rank Sequential ---
        seq = nn.Sequential(
            nn.Linear(in_c, k, bias=False),
            nn.Linear(k, out_c, bias=(lin.bias is not None)),
        ).to(device)

        if lin.bias is not None:
            seq[1].bias.data.copy_(lin.bias.data)
        seq[0].weight.data.copy_(V_k)
        seq[1].weight.data.copy_(U_k)

        # Write back to student model (VICON style)
        if '.' in name:
            parent, attr = name.rsplit('.', 1)
            parent_mod = model.get_submodule(parent)
        else:
            parent_mod = model
            attr = name
        setattr(parent_mod, attr, seq)

        print(f"[{j:02d}] {name}: Compressed to rank {k}/{max_rank}")

    model = model.cpu()
    print("MySVD compression complete.")
    return model


# =====================================================================
# Core MySVD compression (fixed energy ratio) — adapted for VICON
# =====================================================================

@torch.no_grad()
def mysvd_compress_fixedenergy_vicon(
    trainer: Trainer,
    calib_loaders: dict,
    dev: str = "cuda",
    energy_json_path: Optional[str] = None,
    fixed_rank: Optional[int] = None,
    energy_keep: float = 0.8,
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    eps_whiten: float = 1e-6,
    max_calib_batches: int = None,
):
    """
    MySVD compression with per-layer energy ratios (from JSON).
    
    This is the VICON port of svd_compress_sliced_pairwise_fixedenergy.
    """
    device = torch.device(dev)
    model_teacher: nn.Module = trainer.model.to(device).eval()
    model_student = copy.deepcopy(model_teacher)

    # --- Load per-layer energy ratios ---
    loaded_energies = {}
    if energy_json_path is not None and os.path.exists(energy_json_path):
        with open(energy_json_path, "r") as f:
            loaded_energies = json.load(f) or {}
        print(f"[dynamic-energy] loaded {len(loaded_energies)} layer energy ratios")

    linear_names = [(name, m) for name, m in model_teacher.named_modules()
                    if isinstance(m, nn.Linear)]
    print(f"Start MySVD compression on {len(linear_names)} linear layers...")

    for j, (name, lin) in enumerate(tqdm(linear_names)):
        # --- Determine target energy ratio ---
        target_ratio = None
        if name in loaded_energies:
            if loaded_energies[name] is None:
                print(f"[skip] {name}: forced skip by energy.json")
                continue
            try:
                target_ratio = float(loaded_energies[name])
            except Exception:
                target_ratio = None
        if target_ratio is not None and target_ratio >= 1.0:
            print(f"[{j:02d}] {name}: Skipped (energy ratio = 1.0, keep full rank)")
            continue

        in_c, out_c = lin.in_features, lin.out_features
        if not isinstance(lin, nn.Linear):
            continue

        # --- Collect teacher H and student H' ---
        H = collect_layer_input_vicon(
            trainer, model_teacher, name, calib_loaders, device,
            max_calib_batches=max_calib_batches
        )
        H = as_2d_x(H, in_c).to(device, non_blocking=True).double()
        H = to_2d_features(H, in_c)

        Hp = collect_layer_input_vicon(
            trainer, model_student, name, calib_loaders, device,
            max_calib_batches=max_calib_batches
        )
        Hp = as_2d_x(Hp, in_c).to(device, non_blocking=True).double()
        Hp = to_2d_features(Hp, in_c)

        sigma_HpHp = Hp.T @ Hp
        sigma_HHp  = H.T @ Hp

        del H, Hp
        torch.cuda.empty_cache()

        W = lin.weight.data.double().to(device)
        max_rank = min(out_c, in_c)

        # --- Right whitener ---
        Lr, Lr_invT = svd_whitener(sigma_HpHp, eps=eps_whiten)
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        # --- Left whitener (Fisher) ---
        if Ldict is not None and (name in Ldict):
            Ll, Ll_invT = Ldict[name]
            Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)
        else:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = torch.eye(out_c, device=W.device, dtype=W.dtype)

        # --- SVD ---
        if j == 0:
            Ww = (Ll @ W) @ Lr
            U, S, VT = torch.linalg.svd(Ww, full_matrices=False)
        else:
            Mstar = (Ll @ W) @ (sigma_HHp.to(W.dtype) @ Lr_invT)
            U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)

        # --- Dynamic rank selection based on energy ratio ---
        if target_ratio is not None:
            S_sq = S ** 2
            total_energy = S_sq.sum().item()

            if total_energy == 0 or target_ratio == 0:
                k = 1
            else:
                cumsum_energy = torch.cumsum(S_sq, dim=0)
                k_idx = (cumsum_energy / total_energy >= target_ratio).nonzero(as_tuple=True)[0]
                if len(k_idx) > 0:
                    k = k_idx[0].item() + 1
                else:
                    k = len(S)

            k = max(1, min(k, max_rank))
            U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
            sqrtS = torch.sqrt(S_k)
            U_k = (U_k * sqrtS.unsqueeze(0)).float()
            V_k = (sqrtS.unsqueeze(1) * VT_k).float()

        elif fixed_rank is not None:
            k = max(1, min(int(fixed_rank), max_rank))
            U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
            sqrtS = torch.sqrt(S_k)
            U_k = (U_k * sqrtS.unsqueeze(0)).float()
            V_k = (sqrtS.unsqueeze(1) * VT_k).float()
        else:
            U_k, V_k, k = energy_base_rank_selector(U, S, VT, max_rank, energy_keep)

        # --- De-whiten ---
        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(W.dtype)

        num_W  = out_c * in_c
        num_UV = out_c * k + k * in_c

        if num_UV >= num_W or target_ratio == 1.0:
            print(f"[{j:02d}] {name}: Skipped (Full Rank better or Ratio=1.0) k={k}")
            continue

        # --- Replace module ---
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

        ratio_str = f"Energy Ratio: {target_ratio:.4f}" if target_ratio is not None else "fallback"
        print(f"[{j:02d}] {name}: Compressed to rank {k}/{max_rank} ({ratio_str})")

    model_student = model_student.cpu()
    print("MySVD compression complete.")
    return model_student