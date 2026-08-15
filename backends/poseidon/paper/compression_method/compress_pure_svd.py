import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from MySVD.helper.whitener import *
from MySVD.helper.input_collector import *
from MySVD.helper.rank_selector import *
from MySVD.helper.fisher_collector import *
@torch.no_grad()
def svd_compress_standard_puresvd(
    trainer,
    dev: str,
    rank_dict: dict,
    seq_len: int = 4096,
    # 以下参数保留是为了兼容接口，但在本函数中不再使用
    energy_keep: float = 0.999, 
    calib_dataset=None,
    act1=None,
    Ldict=None, 
    eps_whiten: float = 1e-6,
):
    """
    [Standard SVD Version]
    不使用白化，直接对 Weight 进行 SVD 分解。
    用于匹配 gamma 训练时的基底假设。
    """
    device = torch.device(dev)
    model: nn.Module = trainer.model.to(device).eval()
    
    # 确保 rank_dict 不为 None
    if rank_dict is None:
        rank_dict = {}
        print("[Warning] rank_dict is empty, no layers will be compressed via gamma.")

    print(f"Start Standard SVD compression (No Whitening) on model...")

    # 遍历模型所有模块
    # 注意：这里不再依赖 linear_forward_order 和 calib_dataset，因为不需要跑前向
    cnt_compressed = 0
    cnt_skipped = 0

    for name, lin in model.named_modules():
        if not isinstance(lin, nn.Linear):
            continue
            
        # 跳过不需要压缩的层（不在字典里的）
        # 如果你希望不在字典里也进行 energy_keep 压缩，可以把这个 check 去掉逻辑往下写
        gamma_val = rank_dict.get(name)
        if gamma_val is None:
            # 兼容：有时候字典里的 key 带有或不带 "module." 前缀，稍微 try 一下
            gamma_val = rank_dict.get(name.replace("module.", ""))
        
        if gamma_val is None:
            # print(f"Skipping {name} (not found in rank_dict)")
            continue

        in_c, out_c = lin.in_features, lin.out_features
        max_rank = min(in_c, out_c)
        W = lin.weight.data.double().to(device) # [Out, In]

        # =========================================================
        # 1. 核心 Standard SVD 分解 (直接对 W 分解)
        # =========================================================
        # W = U @ diag(S) @ V^T
        try:
            U, S, VT = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            print(f"[Error] SVD failed for {name}: {e}")
            continue

        # =========================================================
        # 2. 秩的选择逻辑 (Gamma -> k)
        # ========================================================
        k_target = math.ceil(gamma_val)
        # 边界保护
        k = max(1, min(k_target, max_rank))
        
        # 检查是否接近满秩，如果是则跳过，避免增加参数量或精度损失
        if k >= max_rank:
            cnt_skipped += 1
            continue

        # =========================================================
        # 3. 截断与重构
        # =========================================================
        U_k = U[:, :k].double()     # [Out, k]
        S_k = S[:k].double()        # [k]
        VT_k = VT[:k, :].double()   # [k, In]

        # 分配奇异值能量 (sqrt) 到两边，保持数值平衡
        sqrtS = torch.sqrt(S_k)
        
        # 构造新权重
        # Linear A (In -> k): Weight shape [k, In] -> sqrt(S) * V^T
        Weight_A = sqrtS.unsqueeze(1) * VT_k 
        
        # Linear B (k -> Out): Weight shape [Out, k] -> U * sqrt(S)
        Weight_B = U_k * sqrtS.unsqueeze(0)

        # =========================================================
        # 4. 替换层
        # =========================================================
        seq = nn.Sequential(
            nn.Linear(in_c, k, bias=False),
            nn.Linear(k, out_c, bias=(lin.bias is not None)),
        ).to(device)
        
        # 赋值权重
        seq[0].weight.data.copy_(Weight_A)
        seq[1].weight.data.copy_(Weight_B)
        
        # 处理 Bias (通常放在第二个 Linear)
        if lin.bias is not None:
            seq[1].bias.data.copy_(lin.bias.data)

        set_module_by_name(model, name, seq)
        cnt_compressed += 1
        
        # 计算保留的能量比例用于 Log
        energy_ratio = torch.sum(S_k**2) / torch.sum(S**2)
        print(f"Compressed {name}: gamma={gamma_val:.1f} -> k={k}/{max_rank} | Energy={energy_ratio:.4f}")

    model = model.cpu()
    print(f"Standard SVD Compression complete. Compressed: {cnt_compressed}, Skipped: {cnt_skipped}")
    return model