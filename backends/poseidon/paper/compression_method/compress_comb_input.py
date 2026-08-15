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
import os, json

@torch.no_grad()
def mysvd_comb(
    trainer,
    dev: str,
    ranks_json_path: Optional[str] = None,
    calib_dataset=None,
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    alpha: float = 0.5,
    eps_whiten: float = 1e-6,
    save_path: str = None,
    checkpoint_percent: float = 0.1,
):
    device = torch.device(dev)
    trainer.set_ar_steps(None, output_all_steps=False)
    model_teacher: nn.Module = trainer.model.to(device).eval()

    # --- 尝试加载 checkpoint ---
    ckpt_path = os.path.join(save_path, "compressing_model.pt") if save_path else None
    # MySVD_origin writes this name after compression finishes (mysvd.py).
    # Keep the resume guard aligned with that final artifact.
    final_path = os.path.join(save_path, "compressed_model_origin.pt") if save_path else None

    if ckpt_path and os.path.exists(ckpt_path) and (final_path and not os.path.exists(final_path)):
        print(f"==== Found checkpoint, resuming from {ckpt_path} ====")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_student = ckpt['model'].to(device).eval()
        compressed_layers = set(ckpt['compressed_layers'])
        print(f"  Already compressed {len(compressed_layers)} layers: {sorted(compressed_layers)[:5]}...")
    else:
        model_student = copy.deepcopy(model_teacher)
        compressed_layers = set()
    
    if calib_dataset is None:
        raise ValueError("Need calib_dataset")

    loaded_ranks = {}
    if ranks_json_path is not None and os.path.exists(ranks_json_path):
        try:
            with open(ranks_json_path, "r") as f:
                loaded_ranks = json.load(f) or {}
            print(f"[fixed-rank] loaded {len(loaded_ranks)} ranks")
        except Exception:
            raise FileNotFoundError(f"No Path: {ranks_json_path}")
            
    linear_names = [(name, m) for name, m in model_teacher.named_modules() if isinstance(m, nn.Linear)]
    total_layers = len(linear_names)
    checkpoint_interval = max(int(total_layers * checkpoint_percent), 1)

    print(f"Start SVD compression on {total_layers} linear layers (checkpoint every {checkpoint_interval} layers)...")
    layers_since_last_save = 0

    for j, (name, lin) in enumerate(tqdm(linear_names)):
        # --- 跳过已压缩的层 ---
        if name in compressed_layers:
            print(f"[{j:02d}] {name}: Skipped (already compressed)")
            continue

        k_forced = None
        if name in loaded_ranks:
          k = loaded_ranks[name]
          if k is None:
            compressed_layers.add(name)
            layers_since_last_save += 1
            continue
          k = int(k)
        else:
          compressed_layers.add(name)
          layers_since_last_save += 1
          continue

        in_c, out_c = lin.in_features, lin.out_features
        num_W  = out_c * in_c
        num_UV = out_c * k + k * in_c
        if num_UV >= num_W:
            print(f"[{j:02d}] {name}: Skipped (Full Rank better) k={k}")
            compressed_layers.add(name)
            layers_since_last_save += 1
            continue

        data_length = len(calib_dataset)
        batch_size = 512
        sigma_HHp = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
        sigma_HH = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
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
            sigma_HH.addmm_(H.T, H)
            sigma_HHp.addmm_(H.T, Hp)
            
            accumulated_count += H.shape[0]
            del H, Hp, batch_ds
            torch.cuda.empty_cache()

        W = lin.weight.data.double().to(device)
        max_rank = min(out_c, in_c)
        sigma_comb = alpha * sigma_HpHp + (1 - alpha) * sigma_HH

        Lr, Lr_invT = svd_whitener(sigma_comb, eps=eps_whiten)
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        if Ldict is not None and (name in Ldict):
            Ll, Ll_invT = Ldict[name]
            Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)
        else:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = torch.eye(out_c, device=W.device, dtype=W.dtype)


        C_comb_right = ((1 - alpha) * sigma_HH + alpha * sigma_HHp).to(W.dtype) @ Lr_invT
        Mstar = (Ll @ W) @ C_comb_right
        U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)
        # SVD
        U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
        sqrtS = torch.sqrt(S_k)
        U_k = (U_k * sqrtS.unsqueeze(0)).float()
        V_k = (sqrtS.unsqueeze(1) * VT_k).float()
        # Re-whiten
        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(W.dtype)

        seq = nn.Sequential(
            nn.Linear(in_c, k, bias=False),
            nn.Linear(k, out_c, bias=(lin.bias is not None)),
        ).to(device)
        
        if lin.bias is not None:
            seq[1].bias.data.copy_(lin.bias.data)
        seq[0].weight.data.copy_(V_k)
        seq[1].weight.data.copy_(U_k)

        set_module_by_name(model_student, name, seq)
        compressed_layers.add(name)
        layers_since_last_save += 1
        print(f"[{j:02d}] {name}: Compressed to rank {k}/{max_rank}")

        # --- 按百分比保存 checkpoint ---
        if save_path and layers_since_last_save >= checkpoint_interval:
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'model': model_student.cpu(),
                'compressed_layers': list(compressed_layers),
            }, ckpt_path)
            model_student = model_student.to(device)
            layers_since_last_save = 0
            pct = len(compressed_layers) / total_layers * 100
            print(f"  [Checkpoint] Saved at {pct:.1f}% ({len(compressed_layers)}/{total_layers} layers)")

    model_student = model_student.cpu()

    # --- 删除 checkpoint，压缩已完成 ---
    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"  [Checkpoint] Removed intermediate checkpoint {ckpt_path}")

    print("Compression complete.")
    return model_student



@torch.no_grad()
def mysvd_comb_ablation(
    trainer,
    dev: str,
    ranks_json_path: Optional[str] = None,
    calib_dataset=None,
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    alpha: float = 0.5,
    eps_whiten: float = 1e-6,
    uniform_rank = False,
    use_fisher = False,
    save_path: str = None,
    checkpoint_percent: float = 0.1,
):
    device = torch.device(dev)
    trainer.set_ar_steps(None, output_all_steps=False)
    model_teacher: nn.Module = trainer.model.to(device).eval()

    # --- 尝试加载 checkpoint ---
    ckpt_path = os.path.join(save_path, "compressing_model.pt") if save_path else None
    final_path = os.path.join(save_path, "compressed_model_origin.pt") if save_path else None

    if ckpt_path and os.path.exists(ckpt_path) and (final_path and not os.path.exists(final_path)):
        print(f"==== Found checkpoint, resuming from {ckpt_path} ====")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_student = ckpt['model'].to(device).eval()
        compressed_layers = set(ckpt['compressed_layers'])
        print(f"  Already compressed {len(compressed_layers)} layers: {sorted(compressed_layers)[:5]}...")
    else:
        model_student = copy.deepcopy(model_teacher)
        compressed_layers = set()
    
    if calib_dataset is None:
        raise ValueError("Need calib_dataset")

    loaded_ranks = {}
    if not uniform_rank:
        if not os.path.exists(ranks_json_path):
            raise FileNotFoundError(f"No Path: {ranks_json_path}")
            
        with open(ranks_json_path, "r") as f:
            loaded_ranks = json.load(f) or {}
        print(f"[fixed-rank] loaded {len(loaded_ranks)} ranks")

            
    linear_names = [(name, m) for name, m in model_teacher.named_modules() if isinstance(m, nn.Linear)]
    total_layers = len(linear_names)
    checkpoint_interval = max(int(total_layers * checkpoint_percent), 1)

    print(f"Start SVD compression on {total_layers} linear layers (checkpoint every {checkpoint_interval} layers)...")
    layers_since_last_save = 0

    for j, (name, lin) in enumerate(tqdm(linear_names)):
        # --- 跳过已压缩的层 ---
        if name in compressed_layers:
            print(f"[{j:02d}] {name}: Skipped (already compressed)")
            continue

        in_c, out_c = lin.in_features, lin.out_features

        if not uniform_rank:
            if name in loaded_ranks:
                k = loaded_ranks[name]
            else:
                compressed_layers.add(name)
                layers_since_last_save += 1
                continue
            if k is None: 
                compressed_layers.add(name)
                layers_since_last_save += 1
                continue
            k = int(k)
        else:
            k = int((out_c * in_c * 0.2) / (out_c + in_c))
            k = max(k, 1)

        num_W  = out_c * in_c
        num_UV = out_c * k + k * in_c
        if num_UV >= num_W:
            print(f"[{j:02d}] {name}: Skipped (Full Rank better) k={k}")
            compressed_layers.add(name)
            layers_since_last_save += 1
            continue

        data_length = len(calib_dataset)
        batch_size = 512
        sigma_HHp = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
        sigma_HH = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
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
            sigma_HH.addmm_(H.T, H)
            sigma_HHp.addmm_(H.T, Hp)
            
            accumulated_count += H.shape[0]
            del H, Hp, batch_ds
            torch.cuda.empty_cache()

        W = lin.weight.data.double().to(device)
        max_rank = min(out_c, in_c)
        sigma_comb = alpha * sigma_HpHp + (1 - alpha) * sigma_HH

        Lr, Lr_invT = svd_whitener(sigma_comb, eps=eps_whiten)
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        if not use_fisher:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = torch.eye(out_c, device=W.device, dtype=W.dtype)
        else:
            Ll, Ll_invT = Ldict[name]
            Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)


        C_comb_right = ((1 - alpha) * sigma_HH + alpha * sigma_HHp).to(W.dtype) @ Lr_invT
        Mstar = (Ll @ W) @ C_comb_right
        U, S, VT = torch.linalg.svd(Mstar, full_matrices=False)
        # SVD
        U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
        sqrtS = torch.sqrt(S_k)
        U_k = (U_k * sqrtS.unsqueeze(0)).float()
        V_k = (sqrtS.unsqueeze(1) * VT_k).float()
        # Re-whiten
        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(W.dtype)

        seq = nn.Sequential(
            nn.Linear(in_c, k, bias=False),
            nn.Linear(k, out_c, bias=(lin.bias is not None)),
        ).to(device)
        
        if lin.bias is not None:
            seq[1].bias.data.copy_(lin.bias.data)
        seq[0].weight.data.copy_(V_k)
        seq[1].weight.data.copy_(U_k)

        set_module_by_name(model_student, name, seq)
        compressed_layers.add(name)
        layers_since_last_save += 1
        print(f"[{j:02d}] {name}: Compressed to rank {k}/{max_rank}")

        # --- 按百分比保存 checkpoint ---
        if save_path and layers_since_last_save >= checkpoint_interval:
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'model': model_student.cpu(),
                'compressed_layers': list(compressed_layers),
            }, ckpt_path)
            model_student = model_student.to(device)
            layers_since_last_save = 0
            pct = len(compressed_layers) / total_layers * 100
            print(f"  [Checkpoint] Saved at {pct:.1f}% ({len(compressed_layers)}/{total_layers} layers)")

    model_student = model_student.cpu()

    # --- 删除 checkpoint，压缩已完成 ---
    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"  [Checkpoint] Removed intermediate checkpoint {ckpt_path}")

    print("Compression complete.")
    return model_student
