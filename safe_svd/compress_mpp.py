import os
import sys
import json
import argparse
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
from einops import rearrange
from tqdm import tqdm

from multiple_physics_pretraining.data_utils.datasets import get_data_loader, DSET_NAME_TO_OBJECT
from multiple_physics_pretraining.models.avit import build_avit
from multiple_physics_pretraining.utils.YParams import YParams

# ===================================================================
# Dataset type → Sobolev config
# ===================================================================
DSET_SOBOLEV_CONFIG = {
    'incompNS':  {'mode': 'spatial_only',   'has_divergence': True,
                  'vel_indices': [0, 1], 'order': 2},             # Laplacian present
    'compNS':    {'mode': 'spatial_only', 'has_divergence': False,
                  'vel_indices': None, 'order': 1},               # Mass/momentum flux
    'swe':       {'mode': 'spatial_only',   'has_divergence': False,
                  'vel_indices': None, 'order': 1},               # Hyperbolic conservation
    'diffre2d':  {'mode': 'spatial_only',   'has_divergence': False,
                  'vel_indices': None, 'order': 2},               # Diffusion driven
}


@torch.no_grad()
def mysvd_compress_mpp(model, train_dataset, params, dev, rank_dict,
                       Ldict=None, alpha=0.5, eps_whiten=1e-6,
                       batches_per_dset=16, save_path=None, checkpoint_percent=0.1,
                       use_trace_balance=True, calibration_seed=42):
    device = torch.device(dev)
    model_teacher = model.to(device).eval()

    ckpt_path = os.path.join(save_path, "compressing_model.pt") if save_path else None
    final_path = os.path.join(save_path, "compressed_model.pt") if save_path else None

    if ckpt_path and os.path.exists(ckpt_path) and (final_path and not os.path.exists(final_path)):
        print(f"[Compress] Resuming from checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_student = ckpt['model'].to(device).eval()
        compressed_layers = set(ckpt['compressed_layers'])
    else:
        model_student = copy.deepcopy(model_teacher)
        compressed_layers = set()

    linears = [(name, m) for name, m in model_teacher.named_modules() if isinstance(m, nn.Linear)]
    total_layers = len(linears)
    checkpoint_interval = max(int(total_layers * checkpoint_percent), 1)
    layers_since_save = 0

    print(f"[Compress] Starting MySVD on {total_layers} linear layers (Trace Balance: {use_trace_balance})...")

    for j, (name, lin) in enumerate(tqdm(linears)):
        if name in compressed_layers:
            continue

        if name not in rank_dict or rank_dict[name] is None:
            compressed_layers.add(name)
            layers_since_save += 1
            continue

        k = int(rank_dict[name])
        in_c, out_c = lin.in_features, lin.out_features
        num_W = out_c * in_c
        num_UV = out_c * k + k * in_c
        if num_UV >= num_W:
            print(f"  [{j}] {name}: skip (full rank cheaper), k={k}")
            compressed_layers.add(name)
            layers_since_save += 1
            continue

        cat_sigma_HH = {}
        cat_sigma_HpHp = {}
        cat_sigma_HHp = {}
        cat_counts = {}

        target_t = get_module_by_name(model_teacher, name)
        target_s = get_module_by_name(model_student, name)
        H_buf_t, H_buf_s = [], []

        def make_interrupting_pre(buf):
            def pre(m, inp):
                x = inp[0].detach().float().reshape(-1, in_c)
                buf.append(x)
                raise StopForwardException("Data collected, stopping forward pass.")
            return pre

        for subset_group in train_dataset.sub_dsets:
            for subset in subset_group.get_per_file_dsets():
                temp_loader = DataLoader(
                    subset, batch_size=params.batch_size,
                    num_workers=params.num_data_workers,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(calibration_seed),
                    drop_last=True,
                )
                dset_batch_count = 0
                for data in temp_loader:
                    if dset_batch_count >= batches_per_dset:
                        break
                    inp_raw, bcs, tar = map(lambda x: x.to(device), data)
                    labels = torch.tensor(
                        train_dataset.subset_dict.get(
                            subset.get_name(),
                            [-1] * len(train_dataset.subset_dict.get(subset.get_name(), [0]))
                        ),
                        device=device,
                    ).unsqueeze(0).expand(tar.shape[0], -1)
                    inp = rearrange(inp_raw, 'b t c h w -> t b c h w')

                    h_t = target_t.register_forward_pre_hook(make_interrupting_pre(H_buf_t))
                    try:
                        _ = model_teacher(inp, labels, bcs)
                    except StopForwardException:
                        pass
                    finally:
                        h_t.remove()

                    student_mod = get_module_by_name(model_student, name)
                    if isinstance(student_mod, nn.Sequential):
                        hook_target_s = student_mod[0]
                    else:
                        hook_target_s = student_mod
                        
                    h_s = hook_target_s.register_forward_pre_hook(make_interrupting_pre(H_buf_s))
                    try:
                        _ = model_student(inp, labels, bcs)
                    except StopForwardException:
                        pass
                    finally:
                        h_s.remove()

                    cat = subset.get_name()
                    if cat not in cat_sigma_HH:
                        cat_sigma_HH[cat] = torch.zeros(in_c, in_c, device=device, dtype=torch.float64)
                        cat_sigma_HpHp[cat] = torch.zeros(in_c, in_c, device=device, dtype=torch.float64)
                        cat_sigma_HHp[cat] = torch.zeros(in_c, in_c, device=device, dtype=torch.float64)
                        cat_counts[cat] = 0
                    
                    H = H_buf_t[-1].double()
                    Hp = H_buf_s[-1].double()
                    cat_sigma_HH[cat].addmm_(H.T, H)
                    cat_sigma_HpHp[cat].addmm_(Hp.T, Hp)
                    cat_sigma_HHp[cat].addmm_(H.T, Hp)
                    cat_counts[cat] += H.shape[0]

                    H_buf_t.clear()
                    H_buf_s.clear()
                    dset_batch_count += 1

                del temp_loader
                torch.cuda.empty_cache()

        # --- Build combined whitener and compress ---
        cats = list(cat_sigma_HH.keys())
        if not cats: continue
            
        if use_trace_balance:
            # ====== Trace Balance Logic ======
            avg_HH = {c: cat_sigma_HH[c] / max(cat_counts[c], 1) for c in cats}
            avg_HpHp = {c: cat_sigma_HpHp[c] / max(cat_counts[c], 1) for c in cats}
            avg_HHp = {c: cat_sigma_HHp[c] / max(cat_counts[c], 1) for c in cats}

            traces_HH = {c: torch.trace(avg_HH[c]).item() for c in cats}
            target_trace = max(traces_HH.values())

            scale_factors = {}
            for c in cats:
                tr = traces_HH[c]
                if tr > 1e-15:
                    scale_factors[c] = target_trace / tr
                else:
                    scale_factors[c] = 1.0

            balanced_HH = [avg_HH[c] * scale_factors[c] for c in cats]
            balanced_HpHp = [avg_HpHp[c] * scale_factors[c] for c in cats]
            balanced_HHp = [avg_HHp[c] * scale_factors[c] for c in cats]

            sigma_HH = sum(balanced_HH) / len(balanced_HH)
            sigma_HpHp = sum(balanced_HpHp) / len(balanced_HpHp)
            sigma_HHp = sum(balanced_HHp) / len(balanced_HHp)
        else:
            # ====== Global Accumulation Logic ======
            total_count = max(sum(cat_counts.values()), 1)
            sigma_HH = sum(cat_sigma_HH.values()) / total_count
            sigma_HpHp = sum(cat_sigma_HpHp.values()) / total_count
            sigma_HHp = sum(cat_sigma_HHp.values()) / total_count
        
        W = lin.weight.data.double().to(device)
        max_rank = min(out_c, in_c)

        sigma_comb = alpha * sigma_HpHp + (1 - alpha) * sigma_HH
        Lr, Lr_invT = svd_whitener(sigma_comb, eps=eps_whiten)
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        if Ldict is not None and name in Ldict:
            Ll, Ll_invT = Ldict[name]
            Ll = Ll.to(W.device, W.dtype)
            Ll_invT = Ll_invT.to(W.device, W.dtype)
        else:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = torch.eye(out_c, device=W.device, dtype=W.dtype)

        C_comb_right = ((1 - alpha) * sigma_HH + alpha * sigma_HHp).to(W.dtype) @ Lr_invT
        Mstar = (Ll @ W) @ C_comb_right

        U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)
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
        seq[0].weight.data.copy_(V_k.float())
        seq[1].weight.data.copy_(U_k.float())

        set_module_by_name(model_student, name, seq)
        compressed_layers.add(name)
        layers_since_save += 1
        print(f"  [{j}] {name}: rank {k}/{max_rank}")

        if save_path and layers_since_save >= checkpoint_interval:
            os.makedirs(save_path, exist_ok=True)
            torch.save({'model': model_student.cpu(), 'compressed_layers': list(compressed_layers)}, ckpt_path)
            model_student = model_student.to(device)
            layers_since_save = 0
            pct = len(compressed_layers) / total_layers * 100
            print(f"  [Checkpoint] {pct:.1f}% ({len(compressed_layers)}/{total_layers})")

    model_student = model_student.cpu()
    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("[Compress] Done.")
    return model_student
