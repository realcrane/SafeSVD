import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from MySVD.helper.whitener import *
from MySVD.helper.input_collector import *
from MySVD.helper.rank_selector import *
from MySVD.helper.fisher_collector import *
from torch.utils.data import Subset

@torch.no_grad()
def svd_compress_sliced_pairwise_fixedrank(
    trainer,
    energy_keep: float,
    dev: str,
    ranks_json_path: Optional[str] = None,
    fixed_rank: Optional[int] = None,
    capture_outputs: bool = False,
    calib_dataset=None,
    act1=None, # 传入零阶信息 H (Base)
    act2=None, # [建议传 None] 不再需要输入导数
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None, # <--- 这里传入你算好的 Ldict
    eps_whiten: float = 1e-6,
):
    device = torch.device(dev)
    trainer.set_ar_steps(None, output_all_steps=False)
    model_teacher: nn.Module = trainer.model.to(device).eval()
    model_student = copy.deepcopy(model_teacher)
    
    if calib_dataset is None:
        raise ValueError("Need calib_dataset")

    loaded_ranks = {}
    if ranks_json_path is not None and os.path.exists(ranks_json_path):
        try:
            with open(ranks_json_path, "r") as f:
                loaded_ranks = json.load(f) or {}
            print(f"[fixed-rank] loaded {len(loaded_ranks)} ranks")
        except Exception:
            raise FileNotFoundError(f"No Path: {args.rank_path}")
            
    linear_names = [(name, m) for name, m in model_teacher.named_modules() if isinstance(m, nn.Linear)]
    print(f"Start SVD compression on {len(linear_names)} linear layers...")

    for j, (name, lin) in enumerate(tqdm(linear_names)):
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

        in_c, out_c = lin.in_features, lin.out_features
        if not isinstance(lin, nn.Linear):
            continue
        data_length = len(calib_dataset)
        batch_size = 512
        sigma_HHp = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
        sigma_HpHp = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
        accumulated_count = 0
        for idx in range(0, data_length, batch_size):
            end_idx = min(idx + batch_size, data_length)
            indices = list(range(idx, end_idx))
            batch_ds = Subset(calib_dataset, indices)

            H = input_catch(trainer, model_teacher, name, batch_ds, device)
            if isinstance(H, list): 
                try: H = torch.cat(H, dim=0)
                except: H = H[0]
            H = as_2d_x(H, in_c).to(device, non_blocking=True).double()
            H = to_2d_features(H, in_c)       

            Hp = input_catch(trainer, model_student, name, batch_ds, device)
            if isinstance(Hp, list): 
                try: Hp = torch.cat(Hp, dim=0)
                except: Hp = Hp[0]  
            Hp = as_2d_x(Hp, in_c).to(device, non_blocking=True).double()
            Hp = to_2d_features(Hp, in_c)

            sigma_HpHp.addmm_(Hp.T, Hp)
            sigma_HHp.addmm_(H.T, Hp)
            
            accumulated_count += H.shape[0]
            del H, Hp, batch_ds
            torch.cuda.empty_cache()

        W = lin.weight.data.double().to(device)
        max_rank = min(out_c, in_c)

        Lr, Lr_invT = svd_whitener(sigma_HpHp, eps=eps_whiten)
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        if Ldict is not None and (name in Ldict):
            Ll, Ll_invT = Ldict[name]
            Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)
        else:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = torch.eye(out_c, device=W.device, dtype=W.dtype)

        # 5. SVD 分解
        # 目标矩阵 M_star = Ll @ W @ Omega @ Lr_invT
        if j == 0:
            # 第一层：直连
            Ww = (Ll @ W) @ Lr
            U, S, VT = torch.linalg.svd(Ww, full_matrices=False)
        else:
            Mstar = (Ll @ W) @ (sigma_HHp @ Lr_invT)
            U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)

        if k_forced is not None:
            k = max(1, min(int(k_forced), max_rank))
            U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
            sqrtS = torch.sqrt(S_k)
            U_k = (U_k * sqrtS.unsqueeze(0)).float()
            V_k = (sqrtS.unsqueeze(1) * VT_k).float()
        else:
            U_k, V_k, k = energy_base_rank_selector(U, S, VT, max_rank, energy_keep)

        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        
        U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(W.dtype)
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

        set_module_by_name(model_student, name, seq)
        print(f"[{j:02d}] {name}: Compressed to rank {k}/{max_rank}")

    model_student = model_student.cpu()
    print("Compression complete.")
    return model_student



@torch.no_grad()
def svd_compress_sliced_pairwise_fixedenergy(
    trainer,
    energy_keep: float,
    dev: str,
    energy_json_path: Optional[str] = None, # <--- 修改：传入包含 energy ratio 的 json
    fixed_rank: Optional[int] = None,
    capture_outputs: bool = False,
    calib_dataset=None,
    act1=None, 
    act2=None, 
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None, 
    eps_whiten: float = 1e-6,
):
    device = torch.device(dev)
    trainer.set_ar_steps(None, output_all_steps=False)
    model_teacher: nn.Module = trainer.model.to(device).eval()
    model_student = copy.deepcopy(model_teacher)
    
    if calib_dataset is None:
        raise ValueError("Need calib_dataset")

    # ==========================================
    # 1. 读取能量比例字典 (Energy Ratio Dict)
    # ==========================================
    loaded_energies = {}
    if energy_json_path is not None and os.path.exists(energy_json_path):
        try:
            with open(energy_json_path, "r") as f:
                loaded_energies = json.load(f) or {}
            print(f"[dynamic-energy] loaded {len(loaded_energies)} layer energy ratios")
        except Exception:
            raise FileNotFoundError(f"No Path: {energy_json_path}")
            
    linear_names = [(name, m) for name, m in model_teacher.named_modules() if isinstance(m, nn.Linear)]
    print(f"Start SVD compression on {len(linear_names)} linear layers...")

    for j, (name, lin) in enumerate(tqdm(linear_names)):
        
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
        if not isinstance(lin, nn.Linear): continue
            
        data_length = len(calib_dataset)
        batch_size = 512
        sigma_HHp = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
        sigma_HpHp = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
        accumulated_count = 0
        
        for idx in range(0, data_length, batch_size):
            end_idx = min(idx + batch_size, data_length)
            indices = list(range(idx, end_idx))
            batch_ds = Subset(calib_dataset, indices)

            H = input_catch(trainer, model_teacher, name, batch_ds, device)
            if isinstance(H, list): 
                try: H = torch.cat(H, dim=0)
                except: H = H[0]
            H = as_2d_x(H, in_c).to(device, non_blocking=True).double()
            H = to_2d_features(H, in_c)       

            Hp = input_catch(trainer, model_student, name, batch_ds, device)
            if isinstance(Hp, list): 
                try: Hp = torch.cat(Hp, dim=0)
                except: Hp = Hp[0]  
            Hp = as_2d_x(Hp, in_c).to(device, non_blocking=True).double()
            Hp = to_2d_features(Hp, in_c)

            sigma_HpHp.addmm_(Hp.T, Hp)
            sigma_HHp.addmm_(H.T, Hp)
            
            accumulated_count += H.shape[0]
            del H, Hp, batch_ds
            torch.cuda.empty_cache()

        W = lin.weight.data.double().to(device)
        max_rank = min(out_c, in_c)

        Lr, Lr_invT = svd_whitener(sigma_HpHp, eps=eps_whiten)
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        if Ldict is not None and (name in Ldict):
            Ll, Ll_invT = Ldict[name]
            Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)
        else:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = torch.eye(out_c, device=W.device, dtype=W.dtype)

        # ==========================================
        # 5. SVD 分解与基于能量的动态截断
        # ==========================================
        if j == 0:
            # 第一层：直连
            Ww = (Ll @ W) @ Lr
            U, S, VT = torch.linalg.svd(Ww, full_matrices=False)
        else:
            Mstar = (Ll @ W) @ (sigma_HHp @ Lr_invT)
            U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)

        if target_ratio is not None:
            # [核心修改] 根据目标能量比例，动态计算发生 Input Shift 后的最佳 k
            S_sq = S ** 2
            total_energy = S_sq.sum().item()
            
            if total_energy == 0 or target_ratio == 0:
                k = 1
            else:
                cumsum_energy = torch.cumsum(S_sq, dim=0)
                # 找到累加能量首次大于等于 target_ratio 的索引
                k_idx = (cumsum_energy / total_energy >= target_ratio).nonzero(as_tuple=True)[0]
                
                if len(k_idx) > 0:
                    k = k_idx[0].item() + 1 # 索引从0开始，rank数量要+1
                else:
                    k = len(S) # 容错：兜底全秩
                    
            k = max(1, min(k, max_rank))
            U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
            sqrtS = torch.sqrt(S_k)
            U_k = (U_k * sqrtS.unsqueeze(0)).float()
            V_k = (sqrtS.unsqueeze(1) * VT_k).float()

        elif fixed_rank is not None:
            # 容错降级：如果 JSON 里没有这层，优先走固定的 k
            k = max(1, min(int(fixed_rank), max_rank))
            U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
            sqrtS = torch.sqrt(S_k)
            U_k = (U_k * sqrtS.unsqueeze(0)).float()
            V_k = (sqrtS.unsqueeze(1) * VT_k).float()
            
        else:
            # 最底层的容错：如果啥都没传，走原版的全局 energy_keep
            U_k, V_k, k = energy_base_rank_selector(U, S, VT, max_rank, energy_keep)

        # 权重复原
        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(W.dtype)
        
        num_W  = out_c * in_c
        num_UV = out_c * k + k * in_c
        
        if num_UV >= num_W or target_ratio == 1.0:
            print(f"[{j:02d}] {name}: Skipped (Full Rank better or Ratio=1.0) k={k}")
            continue

        # 替换模块
        seq = nn.Sequential(
            nn.Linear(in_c, k, bias=False),
            nn.Linear(k, out_c, bias=(lin.bias is not None)),
        ).to(device)
        
        if lin.bias is not None:
            seq[1].bias.data.copy_(lin.bias.data)
        seq[0].weight.data.copy_(V_k)
        seq[1].weight.data.copy_(U_k)

        set_module_by_name(model_student, name, seq)
        print(f"[{j:02d}] {name}: Compressed to rank {k}/{max_rank} (Energy Ratio: {target_ratio:.4f})")

    model_student = model_student.cpu()
    print("Compression complete.")
    return model_student