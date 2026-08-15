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
def mysvd_clean_input(
    trainer,
    dev: str,
    ranks_json_path: Optional[str] = None,
    calib_dataset=None,
    profile_mat=None,
    Ldict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None, # <--- 这里传入你算好的 Ldict
    eps_whiten: float = 1e-6,
):
    device = torch.device(dev)
    trainer.set_ar_steps(None, output_all_steps=False)
    model = trainer.model
    model.eval()
    
    loaded_ranks = {}
    if ranks_json_path is not None and os.path.exists(ranks_json_path):
        try:
            with open(ranks_json_path, "r") as f:
                loaded_ranks = json.load(f) or {}
            print(f"[fixed-rank] loaded {len(loaded_ranks)} ranks")
        except Exception:
            raise FileNotFoundError(f"No Path: {args.rank_path}")
            
    linear_names = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    print(f"Start SVD compression on {len(linear_names)} linear layers...")

    for j, (name, lin) in enumerate(tqdm(linear_names)):
        if name in loaded_ranks:
            k = loaded_ranks[name]
            if k is None:
                print(f"[skip] {name}: forced skip by ranks.json")
                continue
            k = int(k)

        in_c, out_c = lin.in_features, lin.out_features
        num_W  = out_c * in_c
        num_UV = out_c * k + k * in_c
        if num_UV >= num_W:
            print(f"[{j:02d}] {name}: Skipped (Full Rank better) k={k}")
            continue
        if not isinstance(lin, nn.Linear): continue

        W = lin.weight.data.double().to(device)
        max_rank = min(out_c, in_c)

        Ll, Ll_invT = Ldict[name]
        Ll, Ll_invT = Ll.to(W.device, W.dtype), Ll_invT.to(W.device, W.dtype)

        Lr, Lr_invT = profile_mat[name]
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)

        M = Ll @ W @ Lr

        U, S, VT = torch.linalg.svd(M, full_matrices=False)

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

        set_module_by_name(model, name, seq)
        print(f"[{j:02d}] {name}: Compressed to rank {k}/{max_rank}")

    model = model.cpu()
    print("Compression complete.")
    return model