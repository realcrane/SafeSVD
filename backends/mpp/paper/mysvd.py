"""
MySVD Compression for MPP-AViT
Ported from MySVD (Poseidon) to MPP's data/model interface.
Supports optional Trace Energy Balancing for multi-physics datasets.

Pipeline:
  1. profile_input_gram()   — collect X^T X per Linear, SVD-whiten -> (Lr, Lr_invT)
  2. collect_fisher_mpp()   — Fisher with optional Sobolev regularization -> Ldict with (Ll, Ll_invT)
  3. get_global_greedy_rank_dict_mpp() — greedy rank allocation under param budget
  4. mysvd_compress_mpp()   — combined-input SVD compression with teacher/student

Usage:
  python mysvd_mpp.py --yaml_config ... --ckpt_path ... --rank_ratio 0.2 --use_trace_balance y
"""
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


# ===================================================================
# Sobolev loss for MPP (operates on raw-scale tensors)
# ===================================================================

def sobolev_loss_mpp(dset_name, pred, tar, prev_frame, loss_func,
                     order_control=1, balance_ratio=0.5, decay_alpha=1.0,
                     base_loss_val=1.0):
    cfg = DSET_SOBOLEV_CONFIG.get(dset_name, {'mode': 'spatial_only',
                                               'has_divergence': False,
                                               'vel_indices': None})
    total_sob_loss = torch.tensor(0.0, device=pred.device)

    # ---- 1. Spatial Sobolev (all datasets) ----
    current_preds  = [pred]
    current_targets = [tar]

    for k in range(1, order_control + 1):
        next_preds, next_targets = [], []
        current_order_loss = torch.tensor(0.0, device=pred.device)

        for p, t in zip(current_preds, current_targets):
            dx_p = p[..., 1:] - p[..., :-1]
            dx_t = t[..., 1:] - t[..., :-1]
            dy_p = p[..., 1:, :] - p[..., :-1, :]
            dy_t = t[..., 1:, :] - t[..., :-1, :]

            current_order_loss = current_order_loss + loss_func(dx_p, dx_t) + loss_func(dy_p, dy_t)

            if k < order_control:
                next_preds.extend([dx_p, dy_p])
                next_targets.extend([dx_t, dy_t])

        order_val = current_order_loss.item() if isinstance(current_order_loss, torch.Tensor) else current_order_loss
        if order_val > 1e-20:
            norm_weight = (base_loss_val / order_val) * balance_ratio
        else:
            norm_weight = 0.0
        norm_weight = min(norm_weight, 1000.0)
        decay = decay_alpha ** (k - 1)
        total_sob_loss = total_sob_loss + norm_weight * decay * current_order_loss

        current_preds = next_preds
        current_targets = next_targets

    # ---- 2. Divergence term for incompNS ----
    if cfg['has_divergence'] and cfg['vel_indices'] is not None:
        u_idx, v_idx = cfg['vel_indices']
        du_dx_p = pred[:, u_idx, :, 1:] - pred[:, u_idx, :, :-1]
        dv_dy_p = pred[:, v_idx, 1:, :] - pred[:, v_idx, :-1, :]
        du_dx_t = tar[:, u_idx, :, 1:]  - tar[:, u_idx, :, :-1]
        dv_dy_t = tar[:, v_idx, 1:, :]  - tar[:, v_idx, :-1, :]
        min_h = min(du_dx_p.shape[-2], dv_dy_p.shape[-2])
        min_w = min(du_dx_p.shape[-1], dv_dy_p.shape[-1])
        div_pred = du_dx_p[..., :min_h, :min_w] + dv_dy_p[..., :min_h, :min_w]
        div_gt   = du_dx_t[..., :min_h, :min_w] + dv_dy_t[..., :min_h, :min_w]
        div_loss = loss_func(div_pred, div_gt)
        div_val = div_loss.item()
        if div_val > 1e-20:
            div_weight = min((base_loss_val / div_val) * balance_ratio, 1000.0)
        else:
            div_weight = 0.0
        total_sob_loss = total_sob_loss + div_weight * div_loss

    # ---- 3. Temporal Sobolev (compNS only) ----
    if cfg['mode'] == 'spatiotemporal' and prev_frame is not None:
        pred_time_diff = pred - prev_frame
        tar_time_diff  = tar  - prev_frame
        time_loss = loss_func(pred_time_diff, tar_time_diff)
        time_val = time_loss.item()
        if time_val > 1e-20:
            time_weight = min((base_loss_val / time_val) * balance_ratio, 1000.0)
        else:
            time_weight = 0.0
        total_sob_loss = total_sob_loss + time_weight * time_loss

    return total_sob_loss


# ===================================================================
# Utility
# ===================================================================
class StopForwardException(Exception):
    pass

def load_mpp_model(params, checkpoint_path, device):
    model = build_avit(params).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def set_module_by_name(model, dotted_name, new_module):
    parts = dotted_name.rsplit('.', 1)
    if len(parts) == 2:
        parent_name, attr = parts
        parent = model
        for sub in parent_name.split('.'):
            parent = getattr(parent, sub)
    else:
        parent = model
        attr = parts[0]
    setattr(parent, attr, new_module)

def get_module_by_name(model, dotted_name):
    m = model
    for p in dotted_name.split('.'):
        m = getattr(m, p)
    return m

def svd_whitener(Sigma, eps=1e-6):
    s, U = torch.linalg.eigh((Sigma + Sigma.T) * 0.5)
    s = torch.clamp(s, min=0)
    sqrt_s = torch.sqrt(s + eps)
    L     = U * sqrt_s.unsqueeze(0)
    LinvT = U * (1.0 / sqrt_s).unsqueeze(0)
    return L, LinvT

def iter_mpp_batches(dataset, params, device, batches_per_dset=16, seed=42):
    train_dataset = dataset
    for subset_group in train_dataset.sub_dsets:
        for subset in subset_group.get_per_file_dsets():
            temp_loader = DataLoader(
                subset, batch_size=params.batch_size,
                num_workers=params.num_data_workers,
                shuffle=True, generator=torch.Generator().manual_seed(seed),
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
                yield inp, labels, bcs, tar, subset
                dset_batch_count += 1
            del temp_loader


# ===================================================================
# Step 1: Input Profiling
# ===================================================================

@torch.no_grad()
def profile_input_gram(
    model,
    train_dataset,
    params,
    dev,
    batches_per_dset=16,
    use_trace_balance=True,
    calibration_seed=42,
):
    device = torch.device(dev)
    model = model.to(device).eval()
    linears = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    handles = []

    def hook(module, inp, out):
        x = inp[0].detach().float()
        C = x.shape[-1]
        X = x.reshape(-1, C)
        cat = getattr(model, '_current_category', 'default') 
        if cat not in module._gram:
            module._gram[cat] = torch.zeros(C, C, device=x.device, dtype=torch.float64)
            module._gram_count[cat] = 0
        module._gram[cat] += (X.T @ X).double()
        module._gram_count[cat] += X.shape[0]

    for _, module in linears:
        module._gram = {}       
        module._gram_count = {} 
        handles.append(module.register_forward_hook(hook))

    print(f"[Profiling] Collecting Gram matrices (batches_per_dset={batches_per_dset})...")
    for inp, labels, bcs, tar, subset in tqdm(
        iter_mpp_batches(
            train_dataset,
            params,
            device,
            batches_per_dset=batches_per_dset,
            seed=calibration_seed,
        )
    ):
        model._current_category = subset.get_name()
        _ = model(inp, labels, bcs)

    for h in handles:
        h.remove()

    profiling_mat = {}
    print(f"[Profiling] Computing SVD whitening matrices (Trace Balance: {use_trace_balance})...")
    for name, module in tqdm(linears):
        if len(module._gram_count) > 0:
            if use_trace_balance:
                # ====== Trace Balance Logic ======
                cat_Gs = [(module._gram[c] / module._gram_count[c]) for c in module._gram if module._gram_count[c] > 0]
                if len(cat_Gs) > 0:
                    traces = [torch.trace(C) for C in cat_Gs]
                    target_trace = max(traces)
                    balanced_Gs = []
                    for C, tr in zip(cat_Gs, traces):
                        if tr > 1e-15:
                            balanced_Gs.append(C * (target_trace / tr))
                        else:
                            balanced_Gs.append(C)
                    G = (sum(balanced_Gs) / len(balanced_Gs)).double()
                else:
                    G = torch.eye(module.in_features, device=device, dtype=torch.float64)
            else:
                # ====== Global Accumulation Logic ======
                total_gram = sum(module._gram.values())
                total_count = sum(module._gram_count.values())
                G = (total_gram / max(total_count, 1)).double()
        else:
            G = torch.eye(module.in_features, device=device, dtype=torch.float64)

        Lr, Lr_invT = svd_whitener(G, eps=1e-6)
        profiling_mat[name] = (Lr.cpu().float(), Lr_invT.cpu().float())

        del module._gram, module._gram_count
    model.cpu()
    torch.cuda.empty_cache()
    return profiling_mat


# ===================================================================
# Step 2: Fisher Collection
# ===================================================================

def collect_fisher_mpp(model, train_dataset, params, dev, batches_per_dset=16,
                       loss_type='l1', balance_ratio=0.5, decay_alpha=1.0,
                       use_trace_balance=True, calibration_seed=42):
    if loss_type == 'l1':
        loss_func = F.l1_loss
    else:
        loss_func = F.mse_loss

    device = torch.device(dev)
    model = model.to(device).train()
    model.zero_grad()

    def disable_inplace(m):
        if hasattr(m, 'inplace'):
            m.inplace = False
    model.apply(disable_inplace)

    linears = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    cov_dict = {}
    count_dict = {}
    handles = []

    def get_fisher_hook(name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0].detach()
            if g.dim() > 2:
                g = g.reshape(-1, g.shape[-1])
            cov = g.t() @ g
            
            cat = getattr(model, '_current_category', 'default')
            if name not in cov_dict:
                cov_dict[name] = {}
                count_dict[name] = {}
                
            if cat not in cov_dict[name]:
                cov_dict[name][cat] = cov
                count_dict[name][cat] = g.shape[0]
            else:
                cov_dict[name][cat] += cov
                count_dict[name][cat] += g.shape[0]
        return hook

    for name, module in linears:
        handles.append(module.register_full_backward_hook(get_fisher_hook(name)))

    print(f"[Fisher] Collecting gradient covariance (batches_per_dset={batches_per_dset}, loss={loss_type})...")

    for subset_group in train_dataset.sub_dsets:
        for subset in subset_group.get_per_file_dsets():
            dset_name = subset.get_name()
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
                model._current_category = dset_name
                model.zero_grad()
                output = model(inp, labels, bcs)

                spatial_dims = tuple(range(output.ndim))[2:] 
                residuals = output - tar
                tar_norm = 1e-7 + tar.pow(2).mean(spatial_dims, keepdim=True)
                loss = (residuals.pow(2).mean(spatial_dims, keepdim=True) / tar_norm).mean()
                base_loss = loss.detach() 
                
                cfg = DSET_SOBOLEV_CONFIG.get(dset_name, {})
                current_order = cfg.get('order', None)
                if current_order > 0:
                    prev_frame = inp[-1]
                    sob_loss = sobolev_loss_mpp(
                        dset_name, output, tar, prev_frame, loss_func,
                        order_control=current_order, # <-- Pass the specific order
                        balance_ratio=balance_ratio,
                        decay_alpha=decay_alpha, base_loss_val=base_loss.item(),
                    )
                    loss = loss + sob_loss

                loss.backward()
                dset_batch_count += 1

                if dset_batch_count == 1:
                    # Update print string to log the dynamic order
                    sob_str = f" | sob(O{current_order}): {sob_loss.item():.6f}" if current_order > 0 else ""
                    print(f"  [{subset.title}] First batch loss: {base_loss.item():.6f}{sob_str}")
                del output, loss
            del temp_loader

    for h in handles:
        h.remove()

    Ldict = {}
    print(f"[Fisher] Computing whitening matrices from Fisher (Trace Balance: {use_trace_balance})...")
    for name, cov in tqdm(cov_dict.items()):
        try:
            if use_trace_balance:
                # ====== Trace Balance Logic ======
                cat_sigmas = [(cov[c] / count_dict[name][c]) for c in cov if count_dict[name][c] > 0]
                if len(cat_sigmas) == 0:
                    raise ValueError("No valid gradients collected.")
                
                traces = [torch.trace(C) for C in cat_sigmas]
                target_trace = max(traces)
                balanced_sigmas = []
                for C, tr in zip(cat_sigmas, traces):
                    if tr > 1e-15:
                        balanced_sigmas.append(C * (target_trace / tr))
                    else:
                        balanced_sigmas.append(C)
                Sigma = (sum(balanced_sigmas) / len(balanced_sigmas)).double()
            else:
                # ====== Global Accumulation Logic ======
                total_cov = sum(cov.values())
                total_count = sum(count_dict[name].values())
                if total_count == 0:
                    raise ValueError("No valid gradients collected.")
                Sigma = (total_cov / total_count).double()
            
            dim = Sigma.shape[0]
            diag_mean = torch.diagonal(Sigma).mean().item()
            jitter = max(diag_mean * 0.01, 1e-18) if diag_mean > 0 else 1e-9
            Sigma_final = Sigma + torch.eye(dim, device=device, dtype=torch.float64) * jitter

            try:
                L_mat = torch.linalg.cholesky(Sigma_final)
            except RuntimeError:
                eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_final)
                eigenvalues = torch.clamp(eigenvalues, min=1e-18)
                L_mat = eigenvectors * torch.sqrt(eigenvalues).unsqueeze(0)

            L_mat = L_mat.float()
            Lt = L_mat.t()
            try:
                Lt_inv = torch.linalg.inv(Lt)
            except RuntimeError:
                Lt_inv = torch.linalg.pinv(Lt)

            Ldict[name] = (Lt.cpu(), Lt_inv.cpu())
            del Sigma, Sigma_final, L_mat, Lt, Lt_inv
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [Warning] Fisher processing failed for {name}: {e}")
            dim = cov_dict[name][list(cov_dict[name].keys())[0]].shape[0]
            I_cpu = torch.eye(dim)
            Ldict[name] = (I_cpu, I_cpu)

    model.eval()
    return Ldict


# ===================================================================
# Step 3: Greedy Rank Allocation
# ===================================================================

@torch.no_grad()
def get_global_greedy_rank_dict_mpp(model, Ldict, profiling_mat, rank_ratio, dev):
    device = torch.device(dev)
    model = model.to(device).eval()
    linears = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]

    global_sv_pool = []
    total_original_params = 0
    layer_meta = {}

    print("[Rank] Computing weighted singular values for all layers...")
    for name, lin in tqdm(linears):
        W = lin.weight.data.double().to(device)
        out_c, in_c = W.shape
        original_params = out_c * in_c
        total_original_params += original_params
        cost_per_rank = out_c + in_c
        max_rank = min(out_c, in_c)
        layer_meta[name] = (out_c, in_c, max_rank, original_params)

        Lr, _ = profiling_mat[name]
        Lr = Lr.to(W.device, W.dtype)

        if Ldict is not None and name in Ldict:
            Ll = Ldict[name][0].to(W.device, W.dtype)
        else:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)

        M = (Ll @ W) @ Lr
        S = torch.linalg.svdvals(M)

        for i in range(len(S)):
            score = (S[i].item() ** 2) / cost_per_rank
            global_sv_pool.append((score, name, i, cost_per_rank))

    global_sv_pool.sort(key=lambda x: x[0], reverse=True)
    target_budget = int(total_original_params * rank_ratio)

    rank_counts = {name: 0 for name, _ in linears}
    greedy_spent = 0
    print(f"[Rank] Greedy allocation (budget: {target_budget} / {total_original_params} params)...")
    for score, name, sv_idx, cost in global_sv_pool:
        if sv_idx != rank_counts[name]:
            continue
        if greedy_spent + cost <= target_budget:
            rank_counts[name] += 1
            greedy_spent += cost

    def get_layer_cost(name, k):
        out_c, in_c, _, original = layer_meta[name]
        if k == 0:
            return original
        return min(k * (out_c + in_c), original)

    actual_params = sum(get_layer_cost(n, rank_counts[n]) for n in rank_counts)

    if actual_params < target_budget:
        remaining = [item for item in global_sv_pool
                     if item[2] >= rank_counts[item[1]]
                     and rank_counts[item[1]] < layer_meta[item[1]][2]]
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

    print(f"[Rank] Done. Actual params: {actual_params} / {total_original_params} "
          f"({actual_params / total_original_params * 100:.2f}%)")
    model.cpu()
    torch.cuda.empty_cache()
    return rank_dict


# ===================================================================
# Step 4: MySVD Combined-Input Compression
# ===================================================================

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


# ===================================================================
# Main
# ===================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MySVD Compression for MPP-AViT")
    parser.add_argument("--yaml_config", default="multiple_physics_pretraining/config/mpp_avit_L_config.yaml")
    parser.add_argument("--config", default='basic_config', type=str)
    parser.add_argument("--ckpt_path", default="models/MPP_AViT_L")
    parser.add_argument(
        "--data_root",
        default=None,
        help="Prefix relative dataset paths in the MPP YAML configuration with this directory.",
    )
    parser.add_argument('--rank_ratio', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for student input (0=teacher only, 1=student only)')
    parser.add_argument('--DEV', type=str, default='cuda')
    parser.add_argument('--calibration_num', type=int, default=16, help='Number of calibration batches per sub-dataset')
    parser.add_argument(
        '--calibration_seed',
        type=int,
        default=42,
        help='Seed used to select shuffled calibration batches in every MySVD stage.',
    )
    parser.add_argument(
        '--pipeline_stage',
        choices=('all', 'rank_selection', 'compression'),
        default='all',
        help='Run the full legacy pipeline, rank selection only, or compression.',
    )
    
    # Trace Balance
    parser.add_argument('--use_trace_balance', type=str, default='y', choices=['y', 'n'], 
                        help='Enable trace energy balancing for multi-physics datasets to prevent dominant fields.')
    
    # Fisher Args
    parser.add_argument('--use_fisher', type=str, default='y', choices=['y', 'n'])
    parser.add_argument('--fisher_loss', type=str, default='l1', choices=['l1', 'mse'])
    
    # Sobolev Args
    parser.add_argument('--balance_ratio', type=float, default=10.0, help='Weight of Sobolev terms vs base loss')
    parser.add_argument('--decay_alpha', type=float, default=1.0, help='Exponential decay per Sobolev order')

    # Paths
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--profile_cache', type=str, default=None)
    parser.add_argument('--fisher_cache', type=str, default=None)
    parser.add_argument('--rank_cache', type=str, default=None)

    args = parser.parse_args()
    device = torch.device(args.DEV if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse trace balance flag
    trace_bal = (args.use_trace_balance == 'y')

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.data_root:
        data_root = os.path.abspath(args.data_root)

        def _rebase_data_paths(paths):
            rebased = []
            for path_spec in paths:
                path_spec = list(path_spec)
                if not os.path.isabs(path_spec[0]):
                    path_spec[0] = os.path.join(data_root, path_spec[0])
                rebased.append(path_spec)
            return rebased

        params.train_data_paths = _rebase_data_paths(params.train_data_paths)
        params.valid_data_paths = _rebase_data_paths(params.valid_data_paths)
    params.use_ddp = False
    params.log_to_wandb = False
    batch_size = params.batch_size
    batch_num = args.calibration_num / batch_size
    
    print("Loading datasets...")
    _, train_dataset, _ = get_data_loader(
        params, params.train_data_paths, distributed=False, split='train',
        rank=0, train_offset=params.embedding_offset,
    )

    model = load_mpp_model(params, args.ckpt_path, device)

    # --- Auto Cache paths with Trace flag ---
    sob_tag = ""
    trace_tag = "_trace" if trace_bal else "_notrace"
    calibration_tag = f"cali{args.calibration_num}_seed{args.calibration_seed}"
    base_cache = f"MySVD/cache/{calibration_tag}"
    
    if args.profile_cache is None:
        args.profile_cache = f"{base_cache}/profiling_mat{trace_tag}.pt"
    if args.fisher_cache is None:
        args.fisher_cache = f"{base_cache}/fisher_Ldict_{args.fisher_loss}{sob_tag}{trace_tag}.pt"
    if args.rank_cache is None:
        args.rank_cache = f"{base_cache}/{args.rank_ratio}{sob_tag}{trace_tag}.json"
    if args.save_path is None:
        fisher_tag = f"{args.fisher_loss}{sob_tag}" if args.use_fisher == 'y' else "no_fisher"
        args.save_path = (
            f"compression/{args.rank_ratio}/MySVD/"
            f"{calibration_tag}_{fisher_tag}_alpha{args.alpha}{trace_tag}"
        )

    # --- Step 1: Input Profiling ---
    if os.path.exists(args.profile_cache):
        print(f"Loading cached profiling_mat from {args.profile_cache}")
        profiling_mat = torch.load(args.profile_cache)
    elif args.pipeline_stage == 'compression':
        raise FileNotFoundError(
            "Compression stage requires the profiling cache produced by rank "
            f"selection: {args.profile_cache}"
        )
    else:
        profiling_mat = profile_input_gram(model, train_dataset, params, dev=args.DEV,
                                           batches_per_dset=batch_num, use_trace_balance=trace_bal,
                                           calibration_seed=args.calibration_seed)
        os.makedirs(os.path.dirname(args.profile_cache), exist_ok=True)
        torch.save(profiling_mat, args.profile_cache)
        print(f"Saved profiling_mat to {args.profile_cache}")
        model = model.to(device)

    # --- Step 2: Fisher ---
    Ldict = None
    if args.use_fisher == 'y':
        if os.path.exists(args.fisher_cache):
            print(f"Loading cached Ldict from {args.fisher_cache}")
            Ldict = torch.load(args.fisher_cache)
        elif args.pipeline_stage == 'compression':
            raise FileNotFoundError(
                "Compression stage requires the Fisher cache produced by rank "
                f"selection: {args.fisher_cache}"
            )
        else:
            Ldict = collect_fisher_mpp(model, train_dataset, params, dev=args.DEV,
                                       batches_per_dset=batch_num, loss_type=args.fisher_loss,
                                       balance_ratio=args.balance_ratio,
                                       decay_alpha=args.decay_alpha, use_trace_balance=trace_bal,
                                       calibration_seed=args.calibration_seed)
            os.makedirs(os.path.dirname(args.fisher_cache), exist_ok=True)
            torch.save(Ldict, args.fisher_cache)
            print(f"Saved Ldict to {args.fisher_cache}")
            model = model.to(device)

    # --- Step 3: Rank Allocation ---
    if os.path.exists(args.rank_cache):
        print(f"Loading cached rank_dict from {args.rank_cache}")
        with open(args.rank_cache, 'r') as f:
            rank_dict = json.load(f)
    elif args.pipeline_stage == 'compression':
        raise FileNotFoundError(
            "Compression stage requires the rank cache produced by rank "
            f"selection: {args.rank_cache}"
        )
    else:
        rank_dict = get_global_greedy_rank_dict_mpp(model, Ldict, profiling_mat,
                                                     args.rank_ratio, dev=args.DEV)
        os.makedirs(os.path.dirname(args.rank_cache), exist_ok=True)
        with open(args.rank_cache, 'w') as f:
            json.dump(rank_dict, f, indent=2)
        print(f"Saved rank_dict to {args.rank_cache}")
        model = model.to(device)

    if args.pipeline_stage == 'rank_selection':
        print(
            "Rank selection completed successfully; "
            f"compression intentionally skipped ({args.rank_cache})."
        )
        raise SystemExit(0)

    # --- Step 4: Compress ---
    compressed_model = mysvd_compress_mpp(
        model, train_dataset, params, dev=args.DEV,
        rank_dict=rank_dict, Ldict=Ldict, alpha=args.alpha,
        batches_per_dset=batch_num, save_path=args.save_path,
        use_trace_balance=trace_bal,
        calibration_seed=args.calibration_seed,
    )

    # --- Save ---
    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, "compressed_model.pt")
    torch.save(compressed_model, save_file)
    if rank_dict is not None:
        with open(os.path.join(args.save_path, "rank_dict.json"), "w") as f:
            json.dump({k: v for k, v in rank_dict.items() if v is not None}, f, indent=2)
    print(f"Success! Saved to {save_file}")
