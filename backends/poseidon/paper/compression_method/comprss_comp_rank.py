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
def svd_compress_sliced_pairwise_comp_rank(
    trainer,
    dev: str,
    rank_ratio: float = 0.8,
    capture_outputs: bool = False,
    calib_dataset=None,
    act1=None,
    act2=None,
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    eps_whiten: float = 1e-6,
    skip_layer: bool = True
):
    device = torch.device(dev)
    trainer.set_ar_steps(None, output_all_steps=False)
    model_teacher: nn.Module = trainer.model.to(device).eval()
    model_student = copy.deepcopy(model_teacher)
    
    if calib_dataset is None:
        raise ValueError("Need calib_dataset")
            
    linear_names = linear_forward_order(trainer, calib_dataset)
    print(f"Start SVD compression on {len(linear_names)} linear layers...")

    for j, name in enumerate(tqdm(linear_names)):
        if skip_layer and "continuous_position_bias_mlp" in name: continue
        lin = get_module_by_name(model_teacher, name)
        in_c, out_c = lin.in_features, lin.out_features
        max_rank = min(out_c, in_c)
        k = int((out_c * in_c * rank_ratio) / (out_c + in_c))
        k = max(k, 1)
        if (out_c * k + k * in_c) >= (out_c * in_c): continue
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

        if j == 0:
            Ww = (Ll @ W) @ Lr
            U, S, VT = torch.linalg.svd(Ww, full_matrices=False)
        else:
            Mstar = (Ll @ W) @ (sigma_HHp @ Lr_invT)
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
        seq[0].weight.data.copy_(V_k)
        seq[1].weight.data.copy_(U_k)

        set_module_by_name(model_student, name, seq)
        print(f"[{j:02d}] {name}: Compressed to rank {k}/{max_rank}")

    model_student = model_student.cpu()
    print("Compression complete.")
    return model_student