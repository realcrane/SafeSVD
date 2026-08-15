import os
import sys
import argparse
import json
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from poseidon_main.scOT.inference import get_trainer, rollout
from poseidon_main.scOT.problems.base import get_dataset

from MySVD.helper.input_collector import *
from MySVD.helper.rank_selector import *
from MySVD.helper.whitener import *

def get_physics_aligned_inputs(batch, eps=1e-4):
    """
    从 batch 中提取输入，并构造沿着物理演化方向扰动的输入
    """
    # 1. 拿到当前态 (X_t) 和 未来态 (X_{t+dt})
    x_base = batch['pixel_values'].float() # [B, C, H, W]
    x_future = batch['labels'].float()     # [B, C, H, W]
    
    # 注意维度匹配
    if x_future.ndim > x_base.ndim:
        x_future = x_future.squeeze(1)

    # 2. 计算物理演化向量 (Velocity Vector)
    # v = X_{t+dt} - X_t
    velocity = x_future - x_base
    
    # 3. 归一化方向 (Direction)
    # 我们只想要方向，大小由 eps 控制
    # 展平做 norm，防止某些像素变化极大掩盖了其他
    flat_v = velocity.reshape(velocity.shape[0], -1)
    norm = torch.norm(flat_v, dim=1, keepdim=True) + 1e-8
    
    # 恢复形状: [B, 1] -> [B, 1, 1, 1] 以便广播
    norm = norm.view(velocity.shape[0], 1, 1, 1) # 根据你的维度调整
    
    direction = velocity / norm
    
    # 4. 构造扰动输入
    x_perturbed = x_base + eps * direction
    
    return x_base, x_perturbed

@torch.no_grad()
def profile_inputs_physics(
    trainer,
    calib_loader,
    dev,
    flatten: bool = True,
    to_cpu: bool = True,
):
    """
    基于 Trainer + rollout，收集每个 nn.Linear 层的输入激活 H 和输出激活 Z。
    - H: 模块输入（pre-hook 收集）
    - Z: 模块输出（forward-hook 收集）
    - 默认把每个 batch 的输入/输出展平成 [N, dim] 并搬到 CPU

    返回:
      inputs_by_layer:  Dict[str, Tensor 或 List[Tensor]]  (H)
      outputs_by_layer: Dict[str, Tensor 或 List[Tensor]]  (Z)
    """
    device = torch.device(dev)
    model = trainer.model.to(device).eval()

    handles = []
    names, modules = [], []

    # ---- hooks ----
    def make_hooks(layer_name: str, module):

        # pre-hook: 收集输入 H
        def _pre(mod, inp):
            x = inp[0]
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.detach()
            if flatten:
                x = x.reshape(-1, x.shape[-1])  # [N, C_in]
            if to_cpu:
                x = x.float().cpu()
            mod._inputs_bucket.append(x)

        # forward-hook: 收集输出 Z = W H
        def _fwd(mod, inp, out):
            z = out[0] if isinstance(out, (list, tuple)) else out
            z = z.detach()
            if flatten:
                z = z.reshape(-1, z.shape[-1])  # [N, C_out]
            if to_cpu:
                z = z.float().cpu()
            mod._outputs_bucket.append(z)

        return _pre, _fwd

    # 注册所有 Linear
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._inputs_bucket = []
            module._outputs_bucket = []
            pre_hook, fwd_hook = make_hooks(name, module)
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(fwd_hook))
            names.append(name)
            modules.append(module)

    print("Start gathering inputs/outputs via rollout…")
    _ = rollout(
        trainer,
        calib_loader,
        model_path=getattr(trainer.model.config, "name_or_path", ""),
    )

    # 卸载 hook
    for h in handles:
        h.remove()

    # 组装返回
    inputs_by_layer = {}
    outputs_by_layer = {}

    for name, module in zip(names, modules):
        xs = module._inputs_bucket
        zs = module._outputs_bucket

        if len(xs) == 0:
            # 这个层在 rollout 里没被调用
            del module._inputs_bucket
            del module._outputs_bucket
            continue

        # H
        try:
            inputs_by_layer[name] = torch.cat(xs, dim=0)   # [N_total, C_in]
        except Exception:
            inputs_by_layer[name] = xs                     # 维度不齐就保留为 list

        # Z
        try:
            outputs_by_layer[name] = torch.cat(zs, dim=0)  # [N_total, C_out]
        except Exception:
            outputs_by_layer[name] = zs

        del module._inputs_bucket
        del module._outputs_bucket

    model = model.cpu()
    torch.cuda.empty_cache()

    return inputs_by_layer, outputs_by_layer

@torch.no_grad()
def profile_inputs_and_dHdx_dx_seperate(
    trainer,
    calib_loader,
    dev: str,
    eps: float = 1e-4,
    flatten: bool = True,
    to_cpu: bool = True,
    only_linear: bool = True,   # 只收集 nn.Linear；置 False 可对所有模块
    aggregate: str = "concat",  # "concat" 或 "list"
):
    """
    两次 rollout（原始 vs 加入 collator 的 eps*dx 扰动），
    收集每层输入 H 以及 dH/dx ≈ (H_eps - H)/eps。

    返回:
      H_by_layer: Dict[str, Tensor 或 List[Tensor]]
      dHdx_by_layer: Dict[str, Tensor 或 List[Tensor]]
    """
    device = torch.device(dev)
    model = trainer.model.to(device).eval()

    def is_target_module(m):
        return isinstance(m, nn.Linear) if only_linear else True

    # -------- 共用的收集逻辑（一次 rollout）--------
    from contextlib import nullcontext

    def collect_once(patch_dx: bool) -> Dict[str, List[torch.Tensor]]:
        handles = []
        buckets: Dict[str, List[torch.Tensor]] = {}

        def make_pre_hook(layer_name: str):
            def _pre(module, inp):
                x = inp[0]
                if isinstance(x, (list, tuple)):
                    x = x[0]
                if flatten:
                    x = x.reshape(-1, x.shape[-1])      # [N, C]
                # 关键：立刻搬到 CPU，避免长期占用 GPU 显存
                x = x.detach().to('cpu', copy=True)
                buckets.setdefault(layer_name, []).append(x)
            return _pre

        for name, module in model.named_modules():
            if is_target_module(module):
                handles.append(module.register_forward_pre_hook(make_pre_hook(name)))

        try:
            ctx = patch_collator_dx(trainer, eps=eps) if patch_dx else nullcontext()
            with ctx, disable_ar_steps(trainer), swallow_stop_and_mute_metrics(trainer), quiet_everything():
                _ = rollout(trainer, calib_loader, model_path=getattr(trainer.model.config, 'name_or_path', ''))
        finally:
            for h in handles:
                h.remove()

        # 这一步能显著减少碎片/保留缓存导致的失败
        torch.cuda.empty_cache()
        return buckets

    # -------- 两次收集 --------
    from contextlib import nullcontext
    H_lists = collect_once(patch_dx=False)     # 原始
    He_lists = collect_once(patch_dx=True)     # 加扰动（通过 collator 注入 eps*dx）

    # -------- 对齐并做差分 --------
    H_by_layer: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {}
    dHdx_by_layer: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {}

    common_layers = sorted(set(H_lists.keys()) & set(He_lists.keys()))
    # 可选：报告未命中的层
    missed_in_eps = sorted(set(H_lists.keys()) - set(He_lists.keys()))
    missed_in_base = sorted(set(He_lists.keys()) - set(H_lists.keys()))
    if missed_in_eps:
        print(f"[warn] 以下层在原始有但扰动里没收集到: {missed_in_eps}")
    if missed_in_base:
        print(f"[warn] 以下层在扰动有但原始里没收集到: {missed_in_base}")

    def maybe_to_cpu(t):
        return t.float().cpu() if to_cpu else t

    for name in common_layers:
        xs  = H_lists[name]
        xes = He_lists[name]
        # 逐 batch 做差分，避免形状不齐直接 cat 失败
        d_list, h_list = [], []
        for x, xe in zip(xs, xes):
            # 两次 rollout 顺序一致时形状应一致；否则这里会抛错，便于定位
            if x.shape != xe.shape:
                raise RuntimeError(f"Shape mismatch at layer {name}: {x.shape} vs {xe.shape}")
            d = (xe - x) / eps
            d_list.append(maybe_to_cpu(d))
            h_list.append(maybe_to_cpu(x))

        if aggregate == "concat":
            try:
                H_by_layer[name]    = torch.cat(h_list,  dim=0)
                dHdx_by_layer[name] = torch.cat(d_list, dim=0)
            except Exception:
                # 有些层可能批间最后维不一致，退回 list
                H_by_layer[name]    = h_list
                dHdx_by_layer[name] = d_list
        else:
            H_by_layer[name]    = h_list
            dHdx_by_layer[name] = d_list

    # 简单 sanity check：所有层的 max |H_eps - H| 打一行
    try:
        max_abs = max((torch.max(torch.abs(d)).item() for d in dHdx_by_layer.values()
                       if isinstance(d, torch.Tensor)), default=float('nan'))
        print(f"global max |(H_eps-H)/eps| (concat-only layers): {max_abs:.6g}")
    except Exception:
        pass

    # 清理
    model = model.cpu()
    torch.cuda.empty_cache()
    return H_by_layer, dHdx_by_layer


# # =============================================================
# # 普通 SVD 压缩（无 whitening）：每层都“整 dataset 重新推理，早停到该层”
# # =============================================================
# @torch.no_grad()
# def svd_compress_sliced_pairwise_fixedrank(
#     trainer,
#     energy_keep: float,
#     dev: str,
#     ranks_json_path: Optional[str] = None,   # 新增/沿用：每层固定秩
#     fixed_rank: Optional[int] = None,        # 新增：全局固定秩（当 json 没指定时用）
#     capture_outputs: bool = False,
#     calib_dataset=None,
#     act1=None,
#     act2=None
# ):
#     device = torch.device(dev)
#     model: nn.Module = trainer.model.to(device).eval()
#     trainer.set_ar_steps(None, output_all_steps=False)

#     if calib_dataset is None:
#         raise ValueError("svd_compress_sliced_pairwise 需要 calib_dataset")

#     # 读取每层秩（可选）
#     loaded_ranks = {}
#     if ranks_json_path is not None:
#         try:
#             with open(ranks_json_path, "r") as f:
#                 loaded_ranks = json.load(f) or {}
#             print(f"[fixed-rank] loaded {len(loaded_ranks)} ranks from {ranks_json_path}")
#         except Exception as e:
#             print(f"[fixed-rank][warn] failed to load ranks json: {e}")

#     linear_names = linear_forward_order(trainer, calib_dataset)
#     print("number of linear layers:", len(linear_names))
#     if not linear_names:
#         raise RuntimeError("未检测到任何 nn.Linear 层")

#     print("Start per-layer SVD compression (fixed-rank, dataset-wide rollout + early-stop)…")

#     for j, name in enumerate(linear_names):
#         lin = get_module_by_name(model, name)
#         if not isinstance(lin, nn.Linear):
#             continue
#         # 收集该层真实输入
#         # x_in_cpu = forward_until_via_rollout(trainer, name, calib_dataset, device)
#         x_in_cpu, J_cpu = forward_until_via_rollout_jxs(trainer, name, calib_dataset, device, eps=1e-4)

#         if isinstance(x_in_cpu, list):
#             try: x_in_cpu = torch.cat(x_in_cpu, dim=0)
#             except: x_in_cpu = x_in_cpu[0]

#         in_c, out_c = lin.in_features, lin.out_features
#         Hp = torch.cat([
#             as_2d_x(x_in_cpu, in_c),
#             as_2d_x(J_cpu,   in_c)
#         ], dim=0).to(device, non_blocking=True).to(torch.float64)  # [2N, in_c]
#         Hp = to_2d_features(Hp.to(device, non_blocking=True), in_c).double()

#         H_zero = as_2d_x(act1[name], in_c)          # [N, in_c] or 更多样本后拼成 [*, in_c]
#         H_grad = as_2d_x(act2[name], in_c)          # [N, in_c]
#         H = torch.cat([H_zero, H_grad], dim=0).to(device, non_blocking=True).to(torch.float64)     # [2N, in_c]
#         H  = to_2d_features(H, in_c).double().to(device)

#         W = lin.weight.data.double().to(device)  # [out_c, in_c]
#         max_rank = min(out_c, in_c)

#         # ====== 选 k：优先用 ranks.json，其次 fixed_rank，最后才回退 energy_keep ======
#         k_forced = None
#         if name in loaded_ranks:
#             if loaded_ranks[name] is None:
#                 print(f"[skip] {name}: forced skip by ranks.json")
#                 continue
#             try:
#                 k_forced = int(loaded_ranks[name])
#             except Exception:
#                 k_forced = None
#         if k_forced is None and fixed_rank is not None:
#             k_forced = int(fixed_rank)

#         # ====== 第一层：右白化 W@L，SVD 截断，去白化 ======
#         if j == 0:
#             sigma_H = (H.T @ H).to(torch.float64)
#             L, LinvT = svd_whitener(sigma_H, eps=1e-6)   # LinvT = L^{-T}
#             L, LinvT = L.to(W.device, W.dtype), LinvT.to(W.device, W.dtype)

#             Ww = W @ L  # 白化域
#             U, S, VT = torch.linalg.svd(Ww, full_matrices=False)

#             if k_forced is not None:
#                 k = max(1, min(int(k_forced), max_rank))
#                 U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
#                 sqrtS = torch.sqrt(S_k)
#                 U_k = (U_k * sqrtS.unsqueeze(0)).float()
#                 V_k = (sqrtS.unsqueeze(1) * VT_k).float()
#             else:
#                 # 回退能量阈值（备用）
#                 sigma_sq = S.pow(2)
#                 cum = torch.cumsum(sigma_sq, dim=0)
#                 k = torch.searchsorted(cum, sigma_sq.sum() * energy_keep).item() + 1
#                 k = max(1, min(k, max_rank))
#                 U_k, V_k, _ = energy_base_rank_selector(U, S, VT, max_rank, energy_keep)

#             # 去白化（SVD/eigh 白化：用 matmul）
#             V_k = (V_k @ LinvT.T.to(V_k.dtype)).to(W.dtype)

#         # ====== 其余层：交叉白化 M* = W @ (H^T Hp @ L^{-T})，SVD 截断，去白化 ======
#         else:
#             sigma_Hp   = (Hp.T @ Hp).to(torch.float64)
#             Lp, Lp_invT = svd_whitener(sigma_Hp, eps=1e-6)
#             Lp, Lp_invT = Lp.to(W.device, W.dtype), Lp_invT.to(W.device, W.dtype)

#             sigma_H_Hp = (H.T @ Hp).to(W.dtype)         # [in_c, in_c]
#             Mstar = W @ (sigma_H_Hp @ Lp_invT)          # 右白化

#             U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)

#             if k_forced is not None:
#                 k = max(1, min(int(k_forced), max_rank))
#                 U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
#                 sqrtS = torch.sqrt(S_k)
#                 U_k = (U_k * sqrtS.unsqueeze(0)).float()
#                 V_k = (sqrtS.unsqueeze(1) * VT_k).float()
#             else:
#                 k = None
#                 U_k, V_k, _ = energy_base_rank_selector(U, S, VT, max_rank, energy_keep)

#             # 去白化（SVD/eigh 白化：用 matmul）
#             V_k = (V_k @ Lp_invT.T.to(V_k.dtype)).to(W.dtype)

#         # 参数量检查：若不省参就跳过
#         num_W  = out_c * in_c
#         num_UV = out_c * k + k * in_c
#         if num_UV >= num_W:
#             print(f"[{j:02d}] {name}: keep full-rank (no compression gain) (k={k})")
#             continue

#         # 替换为两层线性
#         seq = nn.Sequential(
#             nn.Linear(in_c, k, bias=False),
#             nn.Linear(k, out_c, bias=(lin.bias is not None)),
#         ).to(device)
#         if lin.bias is not None:
#             seq[1].bias.data.copy_(lin.bias.data)
#         seq[0].weight.data.copy_(V_k)
#         seq[1].weight.data.copy_(U_k)

#         set_module_by_name(model, name, seq)
#         print(f"[{j:02d}] {name}: rank {k}/{max_rank}")

#     model = model.cpu()
#     print("Per-layer SVD compression done (fixed-rank).")
#     return model

@torch.no_grad()
def svd_compress_sliced_pairwise_fixedrank(
    trainer,
    energy_keep: float,
    dev: str,
    ranks_json_path: Optional[str] = None,
    fixed_rank: Optional[int] = None,
    capture_outputs: bool = False,
    calib_dataset=None,
    act1=None,
    act2=None,
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,  # ← 左侧白化字典
    eps_whiten: float = 1e-6,
):
    device = torch.device(dev)
    model: nn.Module = trainer.model.to(device).eval()
    trainer.set_ar_steps(None, output_all_steps=False)

    if calib_dataset is None:
        raise ValueError("svd_compress_sliced_pairwise 需要 calib_dataset")

    # 读取每层秩（可选）
    loaded_ranks = {}
    if ranks_json_path is not None:
        try:
            with open(ranks_json_path, "r") as f:
                loaded_ranks = json.load(f) or {}
            print(f"[fixed-rank] loaded {len(loaded_ranks)} ranks from {ranks_json_path}")
        except Exception as e:
            print(f"[fixed-rank][warn] failed to load ranks json: {e}")

    linear_names = linear_forward_order(trainer, calib_dataset)
    print("number of linear layers:", len(linear_names))
    if not linear_names:
        raise RuntimeError("未检测到任何 nn.Linear 层")

    print("Start per-layer SVD compression (two-sided whitening, fixed-rank)…")

    for j, name in enumerate(linear_names):
        lin = get_module_by_name(model, name)
        if not isinstance(lin, nn.Linear):
            continue

        # 真实输入（右侧统计 Hp）
        x_in_cpu, J_cpu = forward_until_via_rollout_jxs(trainer, name, calib_dataset, device, eps=1e-4)

        if isinstance(x_in_cpu, list):
            try: x_in_cpu = torch.cat(x_in_cpu, dim=0)
            except: x_in_cpu = x_in_cpu[0]

        in_c, out_c = lin.in_features, lin.out_features
        Hp = torch.cat([
            as_2d_x(x_in_cpu, in_c),
            as_2d_x(J_cpu,   in_c)
        ], dim=0).to(device, non_blocking=True).to(torch.float64)  # [2N, in_c]
        Hp = to_2d_features(Hp.to(device, non_blocking=True), in_c).double()

        H_zero = as_2d_x(act1[name], in_c)          # [N, in_c] or 更多样本后拼成 [*, in_c]
        H_grad = as_2d_x(act2[name], in_c)          # [N, in_c]
        H = torch.cat([H_zero, H_grad], dim=0).to(device, non_blocking=True).to(torch.float64)     # [2N, in_c]
        H  = to_2d_features(H, in_c).double().to(device)

        W = lin.weight.data.double().to(device)  # [out_c, in_c]
        max_rank = min(out_c, in_c)

        # 右侧白化（输入域）
        if j == 0:
            # 第一层：用目标域 H
            sigma_H = (H.T @ H).to(torch.float64)
            Lr, Lr_invT = svd_whitener(sigma_H, eps=eps_whiten)
        else:
            # 其余层：用上游 Hp（交叉白化）
            sigma_Hp = (Hp.T @ Hp).to(torch.float64)
            Lr, Lr_invT = svd_whitener(sigma_Hp, eps=eps_whiten)
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        # 左侧白化（输出域）：来自你收集好的 Ldict[name]；若缺失则用单位阵
        if Ldict is not None and (name in Ldict):
            Ll, Ll_invT = Ldict[name]
            Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)
        else:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = torch.eye(out_c, device=W.device, dtype=W.dtype)

        # ===== SVD in whitened space =====
        if j == 0:
            # 右白化只用 Lr；但我们现在加入左白化 Ll：Ww = Ll @ W @ Lr
            Ww = (Ll @ W) @ Lr
            U, S, VT = torch.linalg.svd(Ww, full_matrices=False)
        else:
            # 交叉白化：M* = Ll @ W @ (H^T Hp @ Lr_invT)
            sigma_H_Hp = (H.T @ Hp).to(W.dtype)      # [in_c, in_c]
            Mstar = (Ll @ W) @ (sigma_H_Hp @ Lr_invT)
            U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)

        # ===== 选秩 k =====
        k_forced = None
        if name in loaded_ranks:
            if loaded_ranks[name] is None:
                print(f"[skip] {name}: forced skip by ranks.json")
                continue
            try: k_forced = int(loaded_ranks[name])
            except: k_forced = None
        if k_forced is None and fixed_rank is not None:
            k_forced = int(fixed_rank)

        if k_forced is not None:
            k = max(1, min(int(k_forced), max_rank))
            U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
            sqrtS = torch.sqrt(S_k)
            # 把 Σ 的平方根分到两边（方便还原为两层线性）
            U_k = (U_k * sqrtS.unsqueeze(0)).float()
            V_k = (sqrtS.unsqueeze(1) * VT_k).float()
        else:
            # 能量阈值（与你原版一致）
            sigma_sq = S.pow(2)
            cum = torch.cumsum(sigma_sq, dim=0)
            k = torch.searchsorted(cum, sigma_sq.sum() * energy_keep).item() + 1
            k = max(1, min(k, max_rank))
            U_k, V_k, _ = energy_base_rank_selector(U, S, VT, max_rank, energy_keep)

        # ===== de-whiten 回原坐标 =====
        # 右侧去白化：V_k ← V_k @ Lr_invT^T
        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        # 左侧去白化：U_k ← Ll_invT^T @ U_k
        U_k = (Ll_invT.T.to(U_k.dtype) @ U_k).to(W.dtype)

        # ===== 参数量检查：若不省参就跳过 =====
        num_W  = out_c * in_c
        num_UV = out_c * k + k * in_c
        if num_UV >= num_W:
            print(f"[{j:02d}] {name}: keep full-rank (no compression gain) (k={k})")
            continue

        # ===== 用两层线性替换 =====
        seq = nn.Sequential(
            nn.Linear(in_c, k, bias=False),
            nn.Linear(k, out_c, bias=(lin.bias is not None)),
        ).to(device)
        if lin.bias is not None:
            seq[1].bias.data.copy_(lin.bias.data)
        seq[0].weight.data.copy_(V_k)
        seq[1].weight.data.copy_(U_k)

        set_module_by_name(model, name, seq)
        print(f"[{j:02d}] {name}: rank {k}/{max_rank}")

    model = model.cpu()
    print("Per-layer SVD compression done (two-sided whitening, fixed-rank).")
    return model


# =============================================================
# CLI（精简）
# =============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='模型路径或名称（传给 get_trainer）')
    parser.add_argument('--dataset', type=str, default='NONE', help='数据集名称；若为 NONE 则使用预训练物理数据样例')
    parser.add_argument('--save_path', type=str, default=None, help='保存压缩模型/统计（若不需要就别传）')
    parser.add_argument('--energy_keep', type=float, default=0.999, help='SVD 能量阈值，保留累计能量≥该阈值的最小秩')
    parser.add_argument('--DEV', type=str, default='cuda', help='device')

    args = parser.parse_args()

    # 数据配置（与原仓库保持兼容的最小实现）
    if args.dataset != "NONE":
        data_type = "down_streams"
        dataset = args.dataset

    print("==== Loading dataset ====")
    cali_data = get_dataset(
        dataset=dataset,
        which="val",
        num_trajectories=1,
        data_path=f"poseidon_main/camlab-ethz/{data_type}",
        fix_input_to_time_step=0,
        time_step_size=10,
        max_num_time_steps=1,
    )

    print("==== Building trainer ====")
    trainer = get_trainer(
        model_path=args.model,
        batch_size=256,
        dataset=cali_data,
    )
    from torch.utils.data import DataLoader

    calib_loader = DataLoader(
        cali_data,
        batch_size=1,  # 或者随便一个合适的
        shuffle=False,
    )
    # layer_name = "encoder.layers.0.blocks.0.attention.output.dense"

    for batch in cali_data:
        print(batch.keys())
        phy_info = batch["pixel_values"]
        print(phy_info.shape)
        u, v, tracer = phy_info[0, :, :], phy_info[1, :, :], phy_info[2, :, :]
        dux, duy = central_diff_2d(u, dx=1.0, dy=1.0, bc="one-sided")
        dvx, dvy = central_diff_2d(v, dx=1.0, dy=1.0, bc="one-sided")
        dtx, dty = central_diff_2d(tracer, dx=1.0, dy=1.0, bc="one-sided")
        dX_dx = torch.stack([dux, dvx, dtx], dim=0)   # [3, H, W]
        dX_dy = torch.stack([duy, dvy, dty], dim=0)   # [3, H, W]
        break

    # idx = args.save_path.rfind("/")
    # parent_dir = args.save_path[:idx]
    # Ldict_dir = f"{parent_dir}/Ldict.pt"
    # try:
    #     Ldict = torch.load(Ldict_dir)
    # except:
    #     Ldict = collect_left_whiteners_dict(
    #         trainer, calib_dataset=cali_data, dev="cuda",
    #         only_linear=True, m_probes=8, max_batches=4, eps=1e-6
    #     )
    #     os.makedirs(parent_dir, exist_ok=True)
    #     torch.save(Ldict, Ldict_dir)

    # # 采集所有 Linear 的输入激活，展平成 [N, in_features]，全部搬到 CPU
    # print("==== Profiling ====")
    # H, dHdx = profile_inputs_and_dHdx_dx_seperate(
    #     trainer, cali_data, dev="cuda", eps=1e-6, flatten=True, to_cpu=True,
    #     only_linear=True, aggregate="concat"
    #     )
    # print(len(H), len(dHdx))

    # print("==== Running per-layer SVD compression (dataset-wide rollout + early-stop) ====")
    # model = svd_compress_sliced_pairwise_fixedrank(
    #     trainer,
    #     energy_keep=args.energy_keep,
    #     dev=args.DEV,
    #     ranks_json_path="results/ranks.json",
    #     fixed_rank = None,
    #     capture_outputs=False,
    #     calib_dataset=cali_data,
    #     act1 = H,
    #     act2 = dHdx
    # )
    # print("Done.")

    # if args.save_path is not None:
    #     os.makedirs(
    #         args.save_path,
    #         exist_ok=True
    #     )
    #     torch.save(model, f"{args.save_path}/compressed_model.pt")   # fp32
    #     print(f"Save to {args.save_path} Successfully")



# @torch.no_grad()
# def svd_compress_sliced_pairwise(
#     trainer,
#     energy_keep: float,
#     dev: str,
#     ranks_json_path: Optional[str] = None,
#     capture_outputs: bool = False,
#     calib_dataset=None,
#     acts=None
# ) -> Tuple[nn.Module, Dict[str, dict], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#     """
#     新版策略（按你的要求）：
#     - 不再做 pairwise 推进；对每一层：
#         1) 用 forward_until_safely(trainer, layer, dataset, device) “整 dataset + 早停到该层”
#            抓到该层真实输入；
#         2) 对该层做普通 SVD 压缩并替换（两层线性）；
#         3) （可选）再来一次 forward_until_safely 抓替换后的输出。
#     - rollout 始终“整 dataset 只调用一次”，但通过补丁让每个 batch 都早停。
#     """
#     solve_tri = lambda A, B: torch.linalg.solve_triangular(A, B, upper=True)
#     device = torch.device(dev)
#     model: nn.Module = trainer.model.to(device).eval()
#     # 拿到 trainer 之后立刻关
#     trainer.set_ar_steps(None, output_all_steps=False)
#     print("AR steps:", trainer.ar_steps, "output_all_steps:", trainer.output_all_steps)  # 应该是 None 和 False

#     if calib_dataset is None:
#         raise ValueError("svd_compress_sliced_pairwise 需要 calib_dataset 作为校准输入来源")

#     # 线性层的前向顺序（一次 batch 探测）
#     linear_names = linear_forward_order(trainer, calib_dataset)
#     print("number of linear layers:", len(linear_names))
#     if not linear_names:
#         raise RuntimeError("未检测到任何 nn.Linear 层")

#     svd_stats: Dict[str, dict] = {}
#     inputs_before: Dict[str, torch.Tensor] = {}
#     outputs_after: Dict[str, torch.Tensor] = {}

#     print("Start per-layer SVD compression with dataset-wide early-stop rollout…")

#     for j, name in enumerate(linear_names):
#         lin = get_module_by_name(model, name)
#         if not isinstance(lin, nn.Linear):
#             continue

#         # (A) 真实输入：整 dataset rollout + 命中该层早停
#         x_in_cpu = forward_until_via_rollout(trainer, name, calib_dataset, device)
#         if isinstance(x_in_cpu, list):
#             try:    x_in_cpu = torch.cat(x_in_cpu, dim=0)
#             except: x_in_cpu = x_in_cpu[0]

#         in_c, out_c = lin.in_features, lin.out_features
#         inputs_before[name] = x_in_cpu
#         Hp = x_in_cpu.to(device, non_blocking=True)
#         Hp = to_2d_features(Hp, in_c).double().to(device)      # [N, in_c]

#         X = acts[name]
#         H = to_2d_features(X, in_c).double().to(device)      # [N, in_c]
#         # ---------- (B) 选择秩并替换 ----------
#         W = lin.weight.data.double().to(device)   # [out, in]
#         out_c, in_c = W.shape
#         max_rank = min(out_c, in_c)
#         num_W = out_c * in_c
#         if j == 0:
#             sigma_H = (H.T @ H)                                    # [in_c, in_c]，不除 N 也行
#             L, LinvT = svd_whitener(sigma_H, eps=1e-6)
#             U_k, V_k, k = whiten_svd_truncate_dewhite(W, L, LinvT, energy_keep)
#         else:  
#             sigma_Hp = Hp.T @ Hp
#             Lp, Lp_invT = svd_whitener(sigma_Hp, eps=1e-6)
#             sigma_H_Hp = H.T @ Hp
#             Mstar = W @ (sigma_H_Hp @ Lp_invT)
#             U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)
#             # Decide rank
#             U_k, V_k, k = energy_base_rank_selector(U, S, VT, max_rank, energy_keep)
#             V_k = (V_k @ Lp_invT.T.to(V_k.dtype)).to(W.dtype)

#         # 3) 若参数不降则跳过
#         num_UV = out_c * k + k * in_c
#         if num_UV >= num_W:
#             print(f"[{j:02d}] {name}: keep full-rank (no compression gain)")
#             continue
#         seq = nn.Sequential(
#             nn.Linear(in_c, k, bias=False),
#             nn.Linear(k, out_c, bias=(lin.bias is not None)),
#         )
#         if lin.bias is not None:
#             seq[1].bias.data.copy_(lin.bias.data)
#         seq[0].weight.data.copy_(V_k)
#         seq[1].weight.data.copy_(U_k)
#         seq = seq.to(device)
#         _set_module_by_name(model, name, seq)

#         print(f"[{j:02d}] {name}: rank {k}/{max_rank}")

#         # ---------- (C) （可选）再跑一次整 dataset 到该层，记录替换后的输出 ----------
#         if capture_outputs:
#             _, y_after_cpu = forward_until_safely(trainer, name, calib_dataset, device)
#             if isinstance(y_after_cpu, list):
#                 try:
#                     y_after_cpu = torch.cat(y_after_cpu, dim=0)
#                 except Exception:
#                     y_after_cpu = y_after_cpu[0]
#             outputs_after[name] = y_after_cpu

#     model = model.cpu()
#     print("Per-layer SVD compression done (dataset-wide rollout with per-batch early-stop).")
#     return model