import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from MySVD.helper.whitener import *
from MySVD.helper.input_collector import *
from MySVD.helper.rank_selector import *
from MySVD.helper.fisher_collector import *
# =============================================================
# SVD 物理增强重构函数 (Full Rank Refinement)
# =============================================================
@torch.no_grad()
def svd_reconstruct_full_rank_physics_aware(
    trainer,
    dev: str,
    calib_dataset=None,
    act1=None, # 传入零阶信息 H (Base)
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None, # 传入 Sobolev Fisher
    eps_whiten: float = 1e-6,
):
    """
    不进行压缩，保持 Full Rank。
    利用 Sobolev Fisher (Ldict) 和输入对齐 (H/Hp) 对原始权重进行“物理清洗”。
    重构后的模型参数量不变，架构不变。
    """
    device = torch.device(dev)
    model: nn.Module = trainer.model.to(device).eval()
    trainer.set_ar_steps(None, output_all_steps=False)

    if calib_dataset is None:
        raise ValueError("Need calib_dataset")

    # 获取线性层顺序
    linear_names = linear_forward_order(trainer, calib_dataset)
    print(f"Start Physics-Aware Full-Rank Reconstruction on {len(linear_names)} linear layers...")

    for j, name in enumerate(linear_names):
        lin = get_module_by_name(model, name)
        if not isinstance(lin, nn.Linear):
            continue

        # =========================================================
        # 1. 准备统计量 (保持原逻辑)
        # =========================================================
        # 获取 Hp (当前层输入)
        x_in_cpu, _ = forward_until_via_rollout_jxs(trainer, name, calib_dataset, device, eps=1e-4)
        
        if isinstance(x_in_cpu, list): 
            try: x_in_cpu = torch.cat(x_in_cpu, dim=0)
            except: x_in_cpu = x_in_cpu[0]
            
        in_c, out_c = lin.in_features, lin.out_features
        
        # 构造 Hp (当前输入统计)
        Hp = as_2d_x(x_in_cpu, in_c).to(device, non_blocking=True).double()
        Hp = to_2d_features(Hp, in_c)

        # 构造 H (目标输入统计 - Base)
        H_zero = as_2d_x(act1[name], in_c)
        H = to_2d_features(H_zero.to(device, non_blocking=True).double(), in_c)

        # =========================================================
        # 2. 双侧白化矩阵计算
        # =========================================================
        W = lin.weight.data.double().to(device)
        
        # 右侧白化 (Input Whitening)
        if j == 0:
            sigma_H = (H.T @ H)
            Lr, Lr_invT = svd_whitener(sigma_H, eps=eps_whiten)
        else:
            sigma_Hp = (Hp.T @ Hp)
            Lr, Lr_invT = svd_whitener(sigma_Hp, eps=eps_whiten)
        
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        # 左侧白化 (Output Sobolev Weighting)
        if Ldict is not None and (name in Ldict):
            Ll, Ll_invT = Ldict[name]
            Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)
        else:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = torch.eye(out_c, device=W.device, dtype=W.dtype)

        # =========================================================
        # 3. SVD 分解与重构 (核心修改)
        # =========================================================
        # 构造加权后的权重矩阵 M_star
        if j == 0:
            # 第一层：直连
            Ww = (Ll @ W) @ Lr
            # SVD 分解
            U, S, VT = torch.linalg.svd(Ww, full_matrices=False)
        else:
            # 后续层：利用 H 和 Hp 的相关性进行对齐
            sigma_H_Hp = (H.T @ Hp).to(W.dtype)
            Mstar = (Ll @ W) @ (sigma_H_Hp @ Lr_invT)
            # SVD 分解
            U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)

        # ---------------------------------------------------------
        # [修改点] 不截断，保持 Full Rank
        # ---------------------------------------------------------
        # 直接使用完整的 U, S, VT 重构矩阵
        # Reconstructed Whitened Weight: M_recon = U * S * VT
        M_recon = U @ torch.diag(S) @ VT

        # =========================================================
        # 4. 逆变换 (De-whitening) 还原为物理权重
        # =========================================================
        # W_new = Ll^{-1} * M_recon * Lr^{-1}
        # 注意矩阵乘法顺序和转置:
        # 之前: Ww = Ll @ W @ Lr  =>  W = Ll_inv @ Ww @ Lr_inv
        
        # 右侧还原: M_recon @ Lr_invT.T
        W_temp = M_recon @ Lr_invT.T.to(M_recon.dtype)
        
        # 左侧还原: Ll_invT @ W_temp
        W_final = Ll_invT.to(W_temp.dtype) @ W_temp

        # =========================================================
        # 5. 原地更新权重 (In-place Update)
        # =========================================================
        # 确保形状一致 (有时候 SVD 会丢掉极小的全零维度，做个形状检查)
        if W_final.shape != lin.weight.shape:
            # 极罕见情况，如果发生通常是数值问题，这里做一个安全回退或Padding
            print(f"[Warning] Shape mismatch for {name}: {W_final.shape} vs {lin.weight.shape}. Keeping original.")
        else:
            # 更新权重
            lin.weight.data.copy_(W_final.float())
            print(f"[{j:02d}] {name}: Refined (Full Rank). Physics-Aware Updated.")

    model = model.cpu()
    print("Full-Rank Reconstruction complete.")
    return model