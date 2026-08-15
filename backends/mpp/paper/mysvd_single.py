"""
MySVD Compression for MPP-AViT
Ported from MySVD (Poseidon) to MPP's data/model interface.

Pipeline:
  1. profile_input_gram()   — collect X^T X per Linear, SVD-whiten -> (Lr, Lr_invT)
  2. collect_fisher_mpp()   — Fisher with optional Sobolev regularization -> Ldict with (Ll, Ll_invT)
  3. get_global_greedy_rank_dict_mpp() — greedy rank allocation under param budget
  4. mysvd_compress_mpp()   — combined-input SVD compression with teacher/student

Usage:
  python mysvd_mpp.py --yaml_config ... --ckpt_path ... --rank_ratio 0.2
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
# Maps dataset get_name() to which Sobolev terms to use.
#   'spatial_only': compute spatial derivative loss (all datasets)
#   'spatiotemporal': also add temporal derivative loss (compNS only)
#   For incompNS we add an explicit divergence term on top of spatial Sobolev.

DSET_SOBOLEV_CONFIG = {
    'incompNS':  {'mode': 'spatial_only',   'has_divergence': True,
                  'vel_indices': [0, 1]},                         # Vx, Vy, particles
    'compNS':    {'mode': 'spatiotemporal', 'has_divergence': False,
                  'vel_indices': None},                           # Vx, Vy, density, pressure
    'swe':       {'mode': 'spatial_only',   'has_divergence': False,
                  'vel_indices': None},                           # h
    'diffre2d':  {'mode': 'spatial_only',   'has_divergence': False,
                  'vel_indices': None},                           # activator, inhibitor
}


# ===================================================================
# Sobolev loss for MPP (operates on raw-scale tensors)
# ===================================================================

def sobolev_loss_mpp(dset_name, pred, tar, prev_frame, loss_func,
                     order_control=1, balance_ratio=0.5, decay_alpha=1.0,
                     base_loss_val=1.0):
    """
    Compute spatial (and optionally temporal) Sobolev regularisation loss
    for MPP datasets.

    Args:
        dset_name: str from subset.get_name(), e.g. 'incompNS', 'compNS', ...
        pred:       (B, C, H, W) model output (already denormalised by AViT)
        tar:        (B, C, H, W) ground truth target
        prev_frame: (B, C, H, W) last input time step (inp[-1]), same scale
        loss_func:  F.l1_loss or F.mse_loss
        order_control: how many orders of spatial derivatives (1, 2, ...)
        balance_ratio: relative weight of Sobolev vs base loss
        decay_alpha: exponential decay per derivative order
        base_loss_val: scalar, used to auto-balance weights

    Returns:
        total Sobolev loss (torch scalar, differentiable)
    """
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
        # du/dx + dv/dy should be zero; penalise pred divergence vs target divergence
        du_dx_p = pred[:, u_idx, :, 1:] - pred[:, u_idx, :, :-1]
        dv_dy_p = pred[:, v_idx, 1:, :] - pred[:, v_idx, :-1, :]
        du_dx_t = tar[:, u_idx, :, 1:]  - tar[:, u_idx, :, :-1]
        dv_dy_t = tar[:, v_idx, 1:, :]  - tar[:, v_idx, :-1, :]
        # Align spatial dims
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
    """SVD-based whitening: returns (L, L_invT) such that L L^T ≈ Sigma + eps I."""
    s, U = torch.linalg.eigh((Sigma + Sigma.T) * 0.5)
    s = torch.clamp(s, min=0)
    sqrt_s = torch.sqrt(s + eps)
    L     = U * sqrt_s.unsqueeze(0)
    LinvT = U * (1.0 / sqrt_s).unsqueeze(0)
    return L, LinvT


def iter_mpp_batches(dataset, params, device, batches_per_dset=16, seed=42, target_pde=None):
    """
    Iterate over all sub-datasets' batches in MPP style,
    sampling up to `batches_per_dset` batches from EACH sub-dataset
    to ensure balanced coverage across different physics.

    Yields (inp, labels, bcs, tar, subset) per batch.
    inp: (T, B, C, H, W), labels: (B, n_fields), bcs: (B, 2), tar: (B, C, H, W)
    """
    train_dataset = dataset
    for subset_group in train_dataset.sub_dsets:
        for subset in subset_group.get_per_file_dsets():
            if target_pde is not None and subset.get_name() != target_pde:
                continue
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
# Step 1: Input Profiling (Gram matrix X^T X → SVD whitener)
# ===================================================================

@torch.no_grad()
def profile_input_gram(model, train_dataset, params, dev, batches_per_dset=16, target_pde=None):
    """
    Collect X^T X Gram matrices for each nn.Linear via forward hooks,
    then compute SVD whitening matrices (Lr, Lr_invT).
    """
    device = torch.device(dev)
    model = model.to(device).eval()
    linears = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    handles = []

    def hook(module, inp, out):
        x = inp[0].detach().float()
        C = x.shape[-1]
        X = x.reshape(-1, C)
        module._gram += (X.T @ X).double()
        module._gram_count += X.shape[0]

    for _, module in linears:
        module._gram = torch.zeros(module.in_features, module.in_features, device=device, dtype=torch.float64)
        module._gram_count = 0
        handles.append(module.register_forward_hook(hook))

    print(f"[Profiling] Collecting Gram matrices (batches_per_dset={batches_per_dset})...")
    for inp, labels, bcs, tar, subset in tqdm(iter_mpp_batches(train_dataset, params, device, batches_per_dset=batches_per_dset, target_pde=target_pde)):
        _ = model(inp, labels, bcs)

    for h in handles:
        h.remove()

    profiling_mat = {}
    print("[Profiling] Computing SVD whitening matrices...")
    for name, module in tqdm(linears):
        if module._gram_count > 0:
            G = (module._gram / module._gram_count).double()
        else:
            G = torch.eye(module.in_features, device=device, dtype=torch.float64)
        Lr, Lr_invT = svd_whitener(G, eps=1e-6)
        profiling_mat[name] = (Lr.cpu().float(), Lr_invT.cpu().float())
        del module._gram, module._gram_count

    model.cpu()
    torch.cuda.empty_cache()
    return profiling_mat


# ===================================================================
# Step 2: Fisher Collection (with optional Sobolev regularisation)
# ===================================================================

def collect_fisher_mpp(model, train_dataset, params, dev, batches_per_dset=16,
                       loss_type='l1',
                       order_control=0, balance_ratio=0.5, decay_alpha=1.0, 
                       target_pde=None):
    """
    Collect Fisher information (output gradient covariance) for each nn.Linear.

    When order_control > 0, adds spatial Sobolev loss (and temporal for compNS)
    as regularisation on top of the base reconstruction loss. This makes the
    Fisher information aware of derivative-level accuracy, not just pointwise MSE/L1.

    Args:
        order_control: 0 = no Sobolev (original behaviour), 1+ = Sobolev order
        balance_ratio: relative weight of Sobolev terms vs base loss
        decay_alpha:   exponential decay per derivative order
    """
    if loss_type == 'l1':
        loss_func = F.l1_loss
    else:
        loss_func = F.mse_loss

    device = torch.device(dev)
    model = model.to(device).train()
    model.zero_grad()

    # Disable inplace ops to prevent backward hook errors
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
            if name not in cov_dict:
                cov_dict[name] = cov
                count_dict[name] = g.shape[0]
            else:
                cov_dict[name] += cov
                count_dict[name] += g.shape[0]
        return hook

    for name, module in linears:
        handles.append(module.register_full_backward_hook(get_fisher_hook(name)))

    sob_tag = f", Sobolev order={order_control}, balance={balance_ratio}" if order_control > 0 else ""
    print(f"[Fisher] Collecting gradient covariance (batches_per_dset={batches_per_dset}, loss={loss_type}{sob_tag})...")

    for subset_group in train_dataset.sub_dsets:
        for subset in subset_group.get_per_file_dsets():
            dset_name = subset.get_name()  # e.g. 'incompNS', 'compNS', ...
            if target_pde is not None and dset_name != target_pde:
                continue
            temp_loader = DataLoader(
                subset, batch_size=params.batch_size,
                num_workers=params.num_data_workers,
                shuffle=True, generator=torch.Generator().manual_seed(42),
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

                model.zero_grad()
                output = model(inp, labels, bcs)

                # NRMSE loss — same as train_basic.py
                spatial_dims = tuple(range(output.ndim))[2:]  # skip B, C
                residuals = output - tar
                tar_norm = 1e-7 + tar.pow(2).mean(spatial_dims, keepdim=True)
                loss = (residuals.pow(2).mean(spatial_dims, keepdim=True) / tar_norm).mean()
                base_loss = loss.detach()  # for balancing, not backpropagating through this value
                # Add Sobolev regularisation if order_control > 0
                if order_control > 0:
                    # prev_frame = last input timestep (already in physical scale)
                    prev_frame = inp[-1]  # (B, C, H, W)
                    sob_loss = sobolev_loss_mpp(
                        dset_name, output, tar, prev_frame, loss_func,
                        order_control=order_control,
                        balance_ratio=balance_ratio,
                        decay_alpha=decay_alpha,
                        base_loss_val=base_loss.item(),
                    )
                    loss = loss + sob_loss

                loss.backward()
                dset_batch_count += 1

                if dset_batch_count == 1:
                    sob_str = f" | sob: {sob_loss.item():.6f}" if order_control > 0 else ""
                    print(f"  [{subset.title}] First batch loss: {base_loss.item():.6f}{sob_str}")
                del output, loss
            del temp_loader

    for h in handles:
        h.remove()

    # Post-process: Cholesky / Eigh → (Ll, Ll_invT)
    Ldict = {}
    print("[Fisher] Computing whitening matrices from Fisher covariance...")
    for name, cov in tqdm(cov_dict.items()):
        try:
            N = count_dict[name]
            Sigma = (cov / N).double()
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
            dim = cov.shape[0]
            I_cpu = torch.eye(dim)
            Ldict[name] = (I_cpu, I_cpu)

    model.eval()
    return Ldict


# ===================================================================
# Step 3: Greedy Rank Allocation (model-agnostic, reused from MySVD)
# ===================================================================

@torch.no_grad()
def get_global_greedy_rank_dict_mpp(model, Ldict, profiling_mat, rank_ratio, dev):
    """
    Global greedy rank allocation under parameter budget.
    Scores each singular value by (sigma^2) / (out + in) and greedily allocates.
    """
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

    # Greedy pass 1
    rank_counts = {name: 0 for name, _ in linears}
    greedy_spent = 0
    print(f"[Rank] Greedy allocation (budget: {target_budget} / {total_original_params} params)...")
    for score, name, sv_idx, cost in global_sv_pool:
        if sv_idx != rank_counts[name]:
            continue
        if greedy_spent + cost <= target_budget:
            rank_counts[name] += 1
            greedy_spent += cost

    # Real cost computation + compensation pass
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

    # Build rank_dict
    rank_dict = {}
    for name, lin in linears:
        out_c, in_c, max_useful, original = layer_meta[name]
        k = rank_counts[name]
        if k == 0:
            rank_dict[name] = 1
        elif k * (out_c + in_c) >= original:
            rank_dict[name] = None  # don't compress
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
def _collect_layer_input(model, module_name, dataset, params, device, batches_per_dset=16):
    """
    Run forward passes and collect the input activations for a specific Linear layer.
    Returns tensor of shape [N, in_features].
    """
    target = get_module_by_name(model, module_name)
    xs = []

    def pre_hook(m, inp):
        x = inp[0].detach().float()
        x = x.reshape(-1, x.shape[-1])
        xs.append(x.cpu())

    h = target.register_forward_pre_hook(pre_hook)
    model.eval()
    try:
        for inp, labels, bcs, tar, subset in iter_mpp_batches(dataset, params, device, batches_per_dset=batches_per_dset):
            _ = model(inp, labels, bcs)
    finally:
        h.remove()

    return torch.cat(xs, dim=0)


@torch.no_grad()
def mysvd_compress_mpp(model, train_dataset, params, dev, rank_dict,
                       Ldict=None, alpha=0.5, eps_whiten=1e-6,
                       batches_per_dset=16,
                       save_path=None, checkpoint_percent=0.1, target_pde=None):
    """
    MySVD combined-input compression for MPP-AViT.

    For each Linear layer:
      1. Collect H (teacher input) and H' (student input) via forward hooks
      2. Build combined covariance: Σ_comb = α Σ_{H'H'} + (1-α) Σ_{HH}
      3. SVD-whiten with Σ_comb, Fisher-whiten with Ldict
      4. Truncated SVD on M* = Ll W C_right, keep top-k
      5. De-whiten and replace layer with Sequential(V_k, U_k)
    """
    device = torch.device(dev)
    model_teacher = model.to(device).eval()

    # Checkpoint handling
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

    print(f"[Compress] Starting MySVD on {total_layers} linear layers...")

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

        # --- Collect H (teacher) and H' (student) Gram matrices batch-by-batch ---
        sigma_HH  = torch.zeros(in_c, in_c, device=device, dtype=torch.float64)
        sigma_HpHp = torch.zeros(in_c, in_c, device=device, dtype=torch.float64)
        sigma_HHp = torch.zeros(in_c, in_c, device=device, dtype=torch.float64)

        # Hook-based collection (teacher)
        target_t = get_module_by_name(model_teacher, name)
        target_s = get_module_by_name(model_student, name)
        H_buf_t, H_buf_s = [], []

        def make_interrupting_pre(buf):
            def pre(m, inp):
                x = inp[0].detach().float().reshape(-1, in_c)
                buf.append(x)
                raise StopForwardException("Data collected, stopping forward pass.")
            return pre

        batch_count = 0
        for subset_group in train_dataset.sub_dsets:
            for subset in subset_group.get_per_file_dsets():
                if target_pde is not None and subset.get_name() != target_pde:
                    continue
                temp_loader = DataLoader(
                    subset, batch_size=params.batch_size,
                    num_workers=params.num_data_workers,
                    shuffle=True, generator=torch.Generator().manual_seed(42),
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

                    # Teacher forward with hook
                    h_t = target_t.register_forward_pre_hook(make_interrupting_pre(H_buf_t))
                    try:
                        _ = model_teacher(inp, labels, bcs)
                    except StopForwardException:
                        pass
                    finally:
                        h_t.remove()

                    # Student forward with hook
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

                    # Accumulate Gram matrices
                    H = H_buf_t[-1].double()
                    Hp = H_buf_s[-1].double()
                    sigma_HH.addmm_(H.T, H)
                    sigma_HpHp.addmm_(Hp.T, Hp)
                    sigma_HHp.addmm_(H.T, Hp)

                    H_buf_t.clear()
                    H_buf_s.clear()
                    dset_batch_count += 1
                del temp_loader
                torch.cuda.empty_cache()

        # --- Build combined whitener and compress ---
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

        # M* = Ll @ W @ C_right, where C_right = ((1-α)Σ_HH + α Σ_HHp) @ Lr_invT
        C_comb_right = ((1 - alpha) * sigma_HH + alpha * sigma_HHp).to(W.dtype) @ Lr_invT
        Mstar = (Ll @ W) @ C_comb_right

        U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)
        U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
        sqrtS = torch.sqrt(S_k)
        U_k = (U_k * sqrtS.unsqueeze(0)).float()
        V_k = (sqrtS.unsqueeze(1) * VT_k).float()

        # De-whiten
        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(W.dtype)

        # Replace layer in student
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

        # Periodic checkpoint
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
    parser.add_argument('--rank_ratio', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for student input (0=teacher only, 1=student only)')
    parser.add_argument('--DEV', type=str, default='cuda')
    parser.add_argument('--calibration_num', type=int, default=16,
                        help='Number of calibration batches per sub-dataset (ensures balanced coverage)')
    parser.add_argument('--target_pde', type=str, default=None, 
                        help="Specific PDE to calibrate on (e.g., incompNS, compNS, swe, diffre2d). If None, use all.")
    parser.add_argument('--use_fisher', type=str, default='y', choices=['y', 'n'])
    parser.add_argument('--fisher_loss', type=str, default='l1', choices=['l1', 'mse'])
    # Sobolev args
    parser.add_argument('--order_control', type=int, default=2,
                        help='Sobolev derivative order for Fisher (0=off, 1=first order, 2=second order, ...)')
    parser.add_argument('--balance_ratio', type=float, default=10.0,
                        help='Relative weight of Sobolev terms vs base loss in Fisher')
    parser.add_argument('--decay_alpha', type=float, default=1.0,
                        help='Exponential decay per Sobolev derivative order')

    parser.add_argument('--save_path', type=str, default=None)
    # Cache paths
    parser.add_argument('--profile_cache', type=str, default=None)
    parser.add_argument('--fisher_cache', type=str, default=None)
    parser.add_argument('--rank_cache', type=str, default=None)

    args = parser.parse_args()
    device = torch.device(args.DEV if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load params and data ---
    params = YParams(os.path.abspath(args.yaml_config), args.config)
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

    # --- Auto paths ---
    sob_tag = f"_sob{args.order_control}" if args.order_control > 0 else ""
    pde_tag = f"_{args.target_pde}" if args.target_pde else "_all_pdes"
    base_cache = f"MySVD/cache/cali{args.calibration_num}{pde_tag}"
    if args.profile_cache is None:
        args.profile_cache = f"{base_cache}/profiling_mat.pt"
    if args.fisher_cache is None:
        args.fisher_cache = f"{base_cache}/fisher_Ldict_{args.fisher_loss}{sob_tag}.pt"
    if args.rank_cache is None:
        args.rank_cache = f"{base_cache}/{args.rank_ratio}{sob_tag}.json"
    if args.save_path is None:
        fisher_tag = f"{args.fisher_loss}{sob_tag}" if args.use_fisher == 'y' else "no_fisher"
        args.save_path = f"compression/{args.rank_ratio}/MySVD/cali{args.calibration_num}_{fisher_tag}_alpha{args.alpha}{pde_tag}"

    # --- Step 1: Input Profiling ---
    if os.path.exists(args.profile_cache):
        print(f"Loading cached profiling_mat from {args.profile_cache}")
        profiling_mat = torch.load(args.profile_cache)
    else:
        profiling_mat = profile_input_gram(model, train_dataset, params, dev=args.DEV,
                                                batches_per_dset=batch_num, 
                                                target_pde=args.target_pde) # <-- 传参
        os.makedirs(os.path.dirname(args.profile_cache), exist_ok=True)
        torch.save(profiling_mat, args.profile_cache)
        print(f"Saved profiling_mat to {args.profile_cache}")
        model = model.to(device)  # might have been moved to cpu

    # --- Step 2: Fisher ---
    Ldict = None
    if args.use_fisher == 'y':
        if os.path.exists(args.fisher_cache):
            print(f"Loading cached Ldict from {args.fisher_cache}")
            Ldict = torch.load(args.fisher_cache)
        else:
            Ldict = collect_fisher_mpp(model, train_dataset, params, dev=args.DEV,
                                                batches_per_dset=batch_num,
                                                loss_type=args.fisher_loss,
                                                order_control=args.order_control,
                                                balance_ratio=args.balance_ratio,
                                                decay_alpha=args.decay_alpha, 
                                                target_pde=args.target_pde) # <-- 传参
            os.makedirs(os.path.dirname(args.fisher_cache), exist_ok=True)
            torch.save(Ldict, args.fisher_cache)
            print(f"Saved Ldict to {args.fisher_cache}")
            model = model.to(device)

    # --- Step 3: Rank Allocation ---
    if os.path.exists(args.rank_cache):
        print(f"Loading cached rank_dict from {args.rank_cache}")
        with open(args.rank_cache, 'r') as f:
            rank_dict = json.load(f)
    else:
        rank_dict = get_global_greedy_rank_dict_mpp(model, Ldict, profiling_mat,
                                                     args.rank_ratio, dev=args.DEV)
        os.makedirs(os.path.dirname(args.rank_cache), exist_ok=True)
        with open(args.rank_cache, 'w') as f:
            json.dump(rank_dict, f, indent=2)
        print(f"Saved rank_dict to {args.rank_cache}")
        model = model.to(device)

    # --- Step 4: Compress ---
    compressed_model = mysvd_compress_mpp(
            model, train_dataset, params, dev=args.DEV,
            rank_dict=rank_dict, Ldict=Ldict,
            alpha=args.alpha, batches_per_dset=batch_num,
            save_path=args.save_path, 
            target_pde=args.target_pde # <-- 传参
        )

    # --- Save ---
    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, "compressed_model.pt")
    torch.save(compressed_model, save_file)
    if rank_dict is not None:
        with open(os.path.join(args.save_path, "rank_dict.json"), "w") as f:
            json.dump({k: v for k, v in rank_dict.items() if v is not None}, f, indent=2)
    print(f"Success! Saved to {save_file}")