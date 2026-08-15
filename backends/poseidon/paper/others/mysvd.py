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
import csv
import os
import pandas as pd
import math
@torch.no_grad()
def profile_inputs_physics(trainer, calib_loader, dev, flatten: bool = True, to_cpu: bool = True):
    """
    基于 Trainer + rollout，收集每个 nn.Linear 层的输入激活 X。
    - 不再计算 X^T X 和 Cholesky
    - 默认把每个 batch 的输入展平成 [N, in_features] 并搬到 CPU
    返回: Dict[str, Tensor 或 List[Tensor]]，键是 layer 的名字
    """
    device = torch.device(dev)
    model = trainer.model.to(device).eval()

    handles = []
    names, modules = [], []

    # 用 pre-hook 抓输入
    def make_pre_hook(layer_name: str):
        def _pre(module, inp):
            x = inp[0].detach()
            if isinstance(x, (list, tuple)):
                x = x[0]
            if flatten:
                x = x.reshape(-1, x.shape[-1])  # [N, C]
            if to_cpu:
                x = x.float().cpu()
            module._inputs_bucket.append(x)
        return _pre

    # 注册所有 Linear
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._inputs_bucket = []
            handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
            names.append(name)
            modules.append(module)

    print("Start gathering inputs via rollout…")
    _ = rollout(trainer, calib_loader, model_path=getattr(trainer.model.config, 'name_or_path', ''))

    # 卸载 hook + 组装返回
    for h in handles:
        h.remove()

    inputs_by_layer = {}
    for name, module in zip(names, modules):
        xs = module._inputs_bucket
        if len(xs) == 0:
            continue
        try:
            inputs_by_layer[name] = torch.cat(xs, dim=0)  # [N, C]
        except Exception:
            # 维度不齐时保留为 list
            inputs_by_layer[name] = xs
        del module._inputs_bucket

    model = model.cpu()
    torch.cuda.empty_cache()
    return inputs_by_layer

import torch
import torch.nn as nn
from typing import Dict, List, Union

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


# =============================================================
# 普通 SVD 压缩（无 whitening）：每层都“整 dataset 重新推理，早停到该层”
# =============================================================
@torch.no_grad()
def svd_compress_sliced_pairwise_fixedrank(
    trainer,
    energy_keep: float,
    dev: str,
    ranks_json_path: Optional[str] = None,   # 新增/沿用：每层固定秩
    fixed_rank: Optional[int] = None,        # 新增：全局固定秩（当 json 没指定时用）
    capture_outputs: bool = False,
    calib_dataset=None,
    acts=None,
    stats={}
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

    print("Start per-layer SVD compression (fixed-rank, dataset-wide rollout + early-stop)…")

    for j, name in enumerate(linear_names):
        stats[name] = {
            "fixed_rank": None,
            "frob_loss": None
        }
        lin = get_module_by_name(model, name)
        if not isinstance(lin, nn.Linear):
            continue
        # 收集该层真实输入
        x_in_cpu = forward_until_via_rollout(trainer, name, calib_dataset, device)

        if isinstance(x_in_cpu, list):
            try: x_in_cpu = torch.cat(x_in_cpu, dim=0)
            except: x_in_cpu = x_in_cpu[0]

        in_c, out_c = lin.in_features, lin.out_features

        Hp = to_2d_features(x_in_cpu.to(device, non_blocking=True), in_c).double()
        H  = to_2d_features(acts[name], in_c).double().to(device)

        W = lin.weight.data.double().to(device)  # [out_c, in_c]
        max_rank = min(out_c, in_c)

        # ====== 选 k：优先用 ranks.json，其次 fixed_rank，最后才回退 energy_keep ======
        k_forced = None
        if name in loaded_ranks:
            if loaded_ranks[name] is None:
                print(f"[skip] {name}: forced skip by ranks.json")
                continue
            try:
                k_forced = int(loaded_ranks[name])
            except Exception:
                k_forced = None
        if k_forced is None and fixed_rank is not None:
            k_forced = int(fixed_rank)

        # ====== 第一层：右白化 W@L，SVD 截断，去白化 ======
        if j == 0:
            sigma_H = (H.T @ H).to(torch.float64)
            L, LinvT = svd_whitener(sigma_H, eps=1e-6)   # LinvT = L^{-T}
            L, LinvT = L.to(W.device, W.dtype), LinvT.to(W.device, W.dtype)

            Ww = W @ L  # 白化域
            U, S, VT = torch.linalg.svd(Ww, full_matrices=False)

            if k_forced is not None:
                k = max(1, min(int(k_forced), max_rank))
                U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
                sqrtS = torch.sqrt(S_k)
                U_k = (U_k * sqrtS.unsqueeze(0)).float()
                V_k = (sqrtS.unsqueeze(1) * VT_k).float()
            else:
                # 回退能量阈值（备用）
                sigma_sq = S.pow(2)
                cum = torch.cumsum(sigma_sq, dim=0)
                k = torch.searchsorted(cum, sigma_sq.sum() * energy_keep).item() + 1
                k = max(1, min(k, max_rank))
                U_k, V_k, _ = energy_base_rank_selector(U, S, VT, max_rank, energy_keep)

            # 去白化（SVD/eigh 白化：用 matmul）
            V_k = (V_k @ LinvT.T.to(V_k.dtype)).to(W.dtype)

        # ====== 其余层：交叉白化 M* = W @ (H^T Hp @ L^{-T})，SVD 截断，去白化 ======
        else:
            sigma_Hp   = (Hp.T @ Hp).to(torch.float64)
            Lp, Lp_invT = svd_whitener(sigma_Hp, eps=1e-6)
            Lp, Lp_invT = Lp.to(W.device, W.dtype), Lp_invT.to(W.device, W.dtype)

            sigma_H_Hp = (H.T @ Hp).to(W.dtype)         # [in_c, in_c]
            Mstar = W @ (sigma_H_Hp @ Lp_invT)          # 右白化

            U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)

            if k_forced is not None:
                k = max(1, min(int(k_forced), max_rank))
                U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
                sqrtS = torch.sqrt(S_k)
                U_k = (U_k * sqrtS.unsqueeze(0)).float()
                V_k = (sqrtS.unsqueeze(1) * VT_k).float()
            else:
                k = None
                U_k, V_k, _ = energy_base_rank_selector(U, S, VT, max_rank, energy_keep)

            # 去白化（SVD/eigh 白化：用 matmul）
            V_k = (V_k @ Lp_invT.T.to(V_k.dtype)).float()

        # 参数量检查：若不省参就跳过
        num_W  = out_c * in_c
        num_UV = out_c * k + k * in_c
        if num_UV >= num_W:
            print(f"[{j:02d}] {name}: keep full-rank (no compression gain) (k={k})")
            continue
        
        # Record the Frob. loss in this layer
        with torch.no_grad():
            W_recon = (U_k @ V_k).to(device=device,dtype=torch.float64)
            Y_true = H @ W.T
            Y_pred = Hp @ W_recon.T
            diff = Y_true - Y_pred
            fro_loss = torch.norm(diff, p="fro").item()
            stats[name]["fixed_rank"] = k
            stats[name]["fro_loss"] = fro_loss

        # 替换为两层线性
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
    print("Per-layer SVD compression done (fixed-rank).")
    return model, stats

# =============================================================
# 普通 SVD 压缩（无 whitening）：每层都“整 dataset 重新推理，早停到该层”
# =============================================================
@torch.no_grad()
def svd_compress_sliced_pairwise_energy(
    trainer,
    energy_keep: float,
    dev: str,
    ranks_json_path: Optional[str] = None,   # 新增/沿用：每层固定秩
    fixed_rank: Optional[int] = None,        # 新增：全局固定秩（当 json 没指定时用）
    capture_outputs: bool = False,
    calib_dataset=None,
    acts=None,
    stats={}
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

    print("Start per-layer SVD compression (fixed-rank, dataset-wide rollout + early-stop)…")

    for j, name in enumerate(linear_names):
        if math.isnan(stats[name]["fixed_rank"]):
            continue
        
        lin = get_module_by_name(model, name)
        if not isinstance(lin, nn.Linear):
            continue
        # 收集该层真实输入
        x_in_cpu = forward_until_via_rollout(trainer, name, calib_dataset, device)

        if isinstance(x_in_cpu, list):
            try: x_in_cpu = torch.cat(x_in_cpu, dim=0)
            except: x_in_cpu = x_in_cpu[0]

        in_c, out_c = lin.in_features, lin.out_features

        Hp = to_2d_features(x_in_cpu.to(device, non_blocking=True), in_c).double()
        H  = to_2d_features(acts[name], in_c).double().to(device)

        W = lin.weight.data.double().to(device)  # [out_c, in_c]
        max_rank = min(out_c, in_c)

        # ====== 选 k：优先用 ranks.json，其次 fixed_rank，最后才回退 energy_keep ======
        k_forced = None
        if name in loaded_ranks:
            if loaded_ranks[name] is None:
                print(f"[skip] {name}: forced skip by ranks.json")
                continue
            try:
                k_forced = int(loaded_ranks[name])
            except Exception:
                k_forced = None
        if k_forced is None and fixed_rank is not None:
            k_forced = int(fixed_rank)

        # ====== 第一层：右白化 W@L，SVD 截断，去白化 ======
        if j == 0:
            sigma_H = (H.T @ H).to(torch.float64)
            L, LinvT = svd_whitener(sigma_H, eps=1e-6)   # LinvT = L^{-T}
            L, LinvT = L.to(W.device, W.dtype), LinvT.to(W.device, W.dtype)

            Ww = W @ L  # 白化域
            U, S, VT = torch.linalg.svd(Ww, full_matrices=False)

            sigma_sq = S.pow(2)
            cum = torch.cumsum(sigma_sq, dim=0)
            k = torch.searchsorted(cum, sigma_sq.sum() * energy_keep).item() + 1
            k = max(1, min(k, max_rank))
            U_k, V_k, _ = energy_base_rank_selector(U, S, VT, max_rank, energy_keep)

            # 去白化（SVD/eigh 白化：用 matmul）
            V_k = (V_k @ LinvT.T.to(V_k.dtype)).to(W.dtype)

        # ====== 其余层：交叉白化 M* = W @ (H^T Hp @ L^{-T})，SVD 截断，去白化 ======
        else:
            sigma_Hp   = (Hp.T @ Hp).to(torch.float64)
            Lp, Lp_invT = svd_whitener(sigma_Hp, eps=1e-6)
            Lp, Lp_invT = Lp.to(W.device, W.dtype), Lp_invT.to(W.device, W.dtype)

            sigma_H_Hp = (H.T @ Hp).to(W.dtype)         # [in_c, in_c]
            Mstar = W @ (sigma_H_Hp @ Lp_invT)          # 右白化

            U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)

            if k_forced is not None:
                k = max(1, min(int(k_forced), max_rank))
                U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
                sqrtS = torch.sqrt(S_k)
                U_k = (U_k * sqrtS.unsqueeze(0)).float()
                V_k = (sqrtS.unsqueeze(1) * VT_k).float()
            else:
                k = None
                U_k, V_k, k = energy_base_rank_selector(U, S, VT, max_rank, energy_keep)

            # 去白化（SVD/eigh 白化：用 matmul）
            V_k = (V_k @ Lp_invT.T.to(V_k.dtype)).float()

        # 参数量检查：若不省参就跳过
        num_W  = out_c * in_c
        num_UV = out_c * k + k * in_c
        if num_UV >= num_W:
            print(f"[{j:02d}] {name}: keep full-rank (no compression gain) (k={k})")
            continue

        # Record the Frob. loss in this layer
        with torch.no_grad():
            W_recon = (U_k @ V_k).to(device=device,dtype=torch.float64)
            Y_true = H @ W.T
            Y_pred = Hp @ W_recon.T
            diff = Y_true - Y_pred
            fro_loss = torch.norm(diff, p="fro").item()
            stats[name]["energy_rank"] = k
            stats[name]["energy_fro_loss"] = fro_loss
        # 替换为两层线性
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
    print("Per-layer SVD compression done (fixed-rank).")
    return model, stats


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

    print("==== Profiling ====")
    acts = profile_inputs_physics(trainer, cali_data, args.DEV)

    print("==== Running per-layer SVD compression (dataset-wide rollout + early-stop) ====")
    # 1. 读取 CSV
    df_loaded = pd.read_csv("compression/compressed_ft_models/Poseidon-L/fluids.incompressible.BrownianBridge/SVDLLM_CrossSVD/fixed_rank/compression_stats.csv")
    # 2. 还原索引：因为你保存前做了 reset_index，现在要把 'layer_name' 重新设为 Index
    df_loaded.set_index('layer_name', inplace=True)
    # 3. 转换回字典：对应保存时的 orient='index'
    stats_reloaded = df_loaded.to_dict(orient='index')
    # 检查结果
    print(f"Loaded {len(stats_reloaded)} layers.")
    # 打印第一个看看是否一样
    first_key = next(iter(stats_reloaded))
    print(f"Sample: {first_key} -> {stats_reloaded[first_key]}")

    model, stats = svd_compress_sliced_pairwise_energy(
        trainer,
        energy_keep=args.energy_keep,
        dev=args.DEV,
        ranks_json_path=None, # "results/ranks.json",
        fixed_rank = None,
        capture_outputs=False,
        calib_dataset=cali_data,
        acts=acts,
        stats=stats_reloaded
    )
    print("Done.")
    df = pd.DataFrame.from_dict(stats, orient='index')
    df.index.name = 'layer_name'
    df.reset_index(inplace=True)

    if args.save_path is not None:
        os.makedirs(
            args.save_path,
            exist_ok=True
        )
        torch.save(model, f"{args.save_path}/compressed_model.pt")   # fp32
        print(f"Save to {args.save_path} Successfully")

        csv_path = os.path.join(args.save_path, 'compression_stats.csv')
        df.to_csv(csv_path, index=False)
        print(f"Stats saved to: {csv_path}")
