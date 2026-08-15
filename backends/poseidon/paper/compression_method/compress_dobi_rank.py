import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from MySVD.helper.whitener import *
from MySVD.helper.input_collector import *
from MySVD.helper.rank_selector import *
from MySVD.helper.fisher_collector import *
@torch.no_grad()
def svd_compress_sliced_pairwise_dobirank(
    trainer,
    energy_keep: float,
    dev: str,
    rank_dict: Optional[Dict[str, int]] = None,
    seq_len = 1024,
    capture_outputs: bool = False,
    calib_dataset=None,
    act1=None, # 传入零阶信息 H (Base)
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None, 
    eps_whiten: float = 1e-6,
):
    """
    主要逻辑：
    1. 优先使用 rank_dict 中的 k。
    2. 如果没有，退回到 energy_keep 自动截断。
    """
    device = torch.device(dev)
    model: nn.Module = trainer.model.to(device).eval()
    trainer.set_ar_steps(None, output_all_steps=False)

    if calib_dataset is None:
        raise ValueError("Need calib_dataset")
    
    # 确保 rank_dict 不为 None，方便后续 get
    if rank_dict is None:
        rank_dict = {}
            
    linear_names = linear_forward_order(trainer, calib_dataset)
    print(f"Start SVD compression on {len(linear_names)} linear layers...")

    for j, name in enumerate(linear_names):
        lin = get_module_by_name(model, name)
        if not isinstance(lin, nn.Linear):
            continue

        # =========================================================
        # 1. 构造输入统计量 H / Hp (仅使用 Base act1)
        # =========================================================
        x_in_cpu, _ = forward_until_via_rollout_jxs(trainer, name, calib_dataset, device, eps=1e-4)
        
        if isinstance(x_in_cpu, list): 
            try: x_in_cpu = torch.cat(x_in_cpu, dim=0)
            except: x_in_cpu = x_in_cpu[0]
            
        in_c, out_c = lin.in_features, lin.out_features
        
        # 构造 Hp (当前输入)
        Hp = as_2d_x(x_in_cpu, in_c).to(device, non_blocking=True).double()
        Hp = to_2d_features(Hp, in_c)

        # 构造 H (校准输入)
        if name in act1:
            H_zero = as_2d_x(act1[name], in_c)
            H = to_2d_features(H_zero.to(device, non_blocking=True).double(), in_c)
        else:
            # 如果校准数据缺失，回退到使用 Hp
            H = Hp

        # =========================================================
        # 2. 双侧白化 SVD
        # =========================================================
        W = lin.weight.data.double().to(device)
        max_rank = min(out_c, in_c)

        # 右侧白化 (Input Whitening)
        if j == 0:
            sigma_H = (H.T @ H)
            Lr, Lr_invT = svd_whitener(sigma_H, eps=eps_whiten)
        else:
            sigma_Hp = (Hp.T @ Hp)
            Lr, Lr_invT = svd_whitener(sigma_Hp, eps=eps_whiten)
        
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        # 左侧白化 (Fisher Output Weighting)
        if Ldict is not None and (name in Ldict):
            Ll, Ll_invT = Ldict[name]
            Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)
        else:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = torch.eye(out_c, device=W.device, dtype=W.dtype)

        # =========================================================
        # 3. 核心 SVD 分解
        # =========================================================
        if j == 0:
            Ww = (Ll @ W) @ Lr
            U, S, VT = torch.linalg.svd(Ww, full_matrices=False)
        else:
            sigma_H_Hp = (H.T @ Hp).to(W.dtype)
            Mstar = (Ll @ W) @ (sigma_H_Hp @ Lr_invT)
            U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)

        # =========================================================
        # [关键修改] 4. 秩的选择逻辑
        # =========================================================
        k_final = None

        # 优先级 A: rank_dict (精确指定)
        # 注意: 你的 json key 可能带 ".core" 后缀，也可能不带，这里做一个简单的兼容检查
        gamma_val = rank_dict.get(name)
        if gamma_val is not None:
            k_final = math.ceil(gamma_val)
        else:
            continue
        
        # 确保 k 合法 (1 <= k <= max_rank)
        k = max(1, min(k_final, max_rank))
        
        U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
        sqrtS = torch.sqrt(S_k)
        U_k = (U_k * sqrtS.unsqueeze(0)).float()
        V_k = (sqrtS.unsqueeze(1) * VT_k).float()
        
        log_msg = f"Dict/Fixed k={k}"

        # =========================================================
        # 5. 还原与替换
        # =========================================================
        # V_k (Right): De-whiten Input
        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        
        # U_k (Left): De-weight Output (Multiply by inverse Fisher)
        U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(W.dtype)

        # 参数量检查
        num_W  = out_c * in_c
        num_UV = out_c * k + k * in_c
        
        if num_UV >= num_W:
            print(f"[{j:02d}] {name}: Skipped (Full Rank better) k={k}")
            continue

        seq = nn.Sequential(
            nn.Linear(in_c, k, bias=False),
            nn.Linear(k, out_c, bias=(lin.bias is not None)),
        ).to(device)
        
        if lin.bias is not None:
            seq[1].bias.data.copy_(lin.bias.data)
        seq[0].weight.data.copy_(V_k)
        seq[1].weight.data.copy_(U_k)

        set_module_by_name(model, name, seq)
        print(f"[{j:02d}] {name}: Compressed ({log_msg}) -> {k}/{max_rank}")

    model = model.cpu()
    print("Compression complete.")
    return model