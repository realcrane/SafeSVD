import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
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

class SVDCollectorLayer(nn.Module):
    def __init__(self, orig_linear, device, eps_whiten=1e-6):
        super().__init__()
        self.in_c = orig_linear.in_features
        self.out_c = orig_linear.out_features
        self.weight = orig_linear.weight
        self.orig_bias = orig_linear.bias
        self.device = device
        self.eps = eps_whiten

        self.sigma_HpHp = torch.zeros(self.in_c, self.in_c, device=device, dtype=torch.float64)
        self.sigma_HHp  = torch.zeros(self.in_c, self.in_c, device=device, dtype=torch.float64)
        self.cached_inputs: list = []
        self.compressed_outputs: list = []
        self.replay_idx = 0

        self.teacher_H_current = None   # 只存当前 batch 的 H，用完即丢
        self.seq = None
        self.phase = "collect"

    def forward(self, x):
        if self.phase == "collect":
            Hp = x.reshape(-1, self.in_c).double()

            # teacher_H_current 由 teacher hook 在同一个 batch 填入
            H = self.teacher_H_current
            if H is not None:
                H = H.to(self.device).double()
                self.sigma_HHp.addmm_(H.T, Hp)
                self.teacher_H_current = None   # 用完即丢

            self.sigma_HpHp.addmm_(Hp.T, Hp)
            self.cached_inputs.append(x.detach())
            return F.linear(x, self.weight, self.orig_bias)

        elif self.phase == "replay":
            out = self.compressed_outputs[self.replay_idx]
            self.replay_idx += 1
            return out

        else:  # "compressed"
            return self.seq(x)

    def compress(self, rank_ratio, Ll=None, Ll_invT=None, is_first_layer=False):
        W = self.weight.data.double().to(self.device)
        in_c, out_c = self.in_c, self.out_c
        k = int((out_c * in_c * rank_ratio) / (out_c + in_c))
        k = max(k, 1)
        if (out_c * k + k * in_c) >= (out_c * in_c):
            self.phase = "compressed"
            self.seq = nn.Linear(in_c, out_c, bias=self.orig_bias is not None).to(self.device)
            self.seq.weight.data.copy_(self.weight.data)
            if self.orig_bias is not None: self.seq.bias.data.copy_(self.orig_bias.data)
            return -1

        Lr, Lr_invT = svd_whitener(self.sigma_HpHp, eps=self.eps)
        Lr, Lr_invT = Lr.to(W.device, W.dtype), Lr_invT.to(W.device, W.dtype)
        if Ll is None:
            Ll = torch.eye(out_c, device=W.device, dtype=W.dtype)
            Ll_invT = Ll.clone()

        if is_first_layer:
            U, S, VT = torch.linalg.svd((Ll @ W) @ Lr, full_matrices=False)
        else:
            U, S, VT = torch.linalg.svd((Ll @ W) @ (self.sigma_HHp.to(W.dtype) @ Lr_invT),
                                         full_matrices=False)

        U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
        sqrtS = torch.sqrt(S_k)
        U_k = (U_k * sqrtS.unsqueeze(0)).float()
        V_k = (sqrtS.unsqueeze(1) * VT_k).float()
        V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(W.dtype)
        U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(W.dtype)

        seq = nn.Sequential(
            nn.Linear(in_c, k, bias=False),
            nn.Linear(k, out_c, bias=(self.orig_bias is not None)),
        ).to(self.device)
        seq[0].weight.data.copy_(V_k)
        seq[1].weight.data.copy_(U_k)
        if self.orig_bias is not None:
            seq[1].bias.data.copy_(self.orig_bias.data)
        self.seq = seq

        # 预算 replay 输出
        self.compressed_outputs = []
        with torch.no_grad():
            for x in self.cached_inputs:
                self.compressed_outputs.append(seq(x.to(self.device)).cpu())
        self.cached_inputs.clear()
        self.phase = "replay"
        return k

    def reset_collect(self):
        self.sigma_HpHp.zero_()
        self.sigma_HHp.zero_()
        self.cached_inputs.clear()
        self.teacher_H_current = None
        self.batch_idx = 0
        self.phase = "collect"

    def finalize(self):
        self.cached_inputs.clear()
        self.compressed_outputs.clear()
        self.teacher_H_current = None
        self.phase = "compressed"


def paired_forward_batch(trainer, model_teacher, model_student,
                         collectors, batch, device):
    """
    一个 batch:
      1) teacher 前向 → hook 把每层 H 塞进对应 collector.teacher_H_current
      2) student 前向 → collector.forward 里配对累积协方差
    teacher 的 H 用完即丢，不累积
    """
    # --- teacher hook: 抓 H 塞给 collector ---
    handles = []
    for name, c in collectors.items():
        mod = get_module_by_name(model_teacher, name)
        def make_hook(collector):
            def hook_fn(_, inp):
                x = inp[0].detach()
                collector.teacher_H_current = x.reshape(-1, collector.in_c)
            return hook_fn
        handles.append(mod.register_forward_pre_hook(make_hook(c)))
    curr_inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    for inputs in [curr_inputs, prev_inputs]:
      if "pixel_values" in inputs:
          pv = inputs["pixel_values"]
          if pv.dim() == 3: 
              inputs["pixel_values"] = pv.unsqueeze(0)
          elif pv.dim() == 5: 
              inputs["pixel_values"] = pv[:, 0, ...]
    model_teacher(**curr_inputs)
    for h in handles: h.remove()

    # --- student 前向: collector 里自动配对 ---
    model_student(**curr_inputs)


@torch.no_grad()
def svd_compress_with_collector(trainer, dev, rank_ratio=0.8,
                                 calib_dataset=None, Ldict=None,
                                 eps_whiten=1e-6, skip_layer=True):
    device = torch.device(dev)
    model_teacher = trainer.model.to(device).eval()
    model_student = copy.deepcopy(model_teacher)
    linear_names = linear_forward_order(trainer, calib_dataset)

    # ===== 替换 student 目标层为 Collector =====
    collectors = {}
    for name in linear_names:
        if skip_layer and "continuous_position_bias_mlp" in name: continue
        lin = get_module_by_name(model_student, name)
        if not isinstance(lin, nn.Linear): continue
        c = SVDCollectorLayer(lin, device, eps_whiten)
        set_module_by_name(model_student, name, c)
        collectors[name] = c

    # ===== 第一轮: 逐 batch 交替前向，所有层同时 collect =====
    loader = torch.utils.data.DataLoader(calib_dataset, batch_size=512)
    for batch in loader:
        paired_forward_batch(trainer, model_teacher, model_student,
                            collectors, batch, device)
        torch.cuda.empty_cache()

    # ===== 逐层压缩 + re-collect =====
    names_list = list(collectors.keys())
    for j, name in enumerate(tqdm(names_list)):
        c = collectors[name]
        Ll, Ll_invT = None, None
        if Ldict and name in Ldict:
            Ll, Ll_invT = Ldict[name]
            Ll, Ll_invT = Ll.to(device), Ll_invT.to(device)

        k = c.compress(rank_ratio, Ll, Ll_invT, is_first_layer=(j == 0))
        max_rank = min(c.out_c, c.in_c)
        print(f"[{j:02d}] {name}: {k}/{max_rank}")

        if j + 1 < len(names_list):
            for future_name in names_list[j+1:]:
                collectors[future_name].reset_collect()

            # re-collect: teacher 也要重跑（H不变但需要逐batch配对）
            for batch in loader:
                paired_forward_batch(trainer, model_teacher, model_student,
                                    collectors, batch, device)
                torch.cuda.empty_cache()

        c.finalize()

    # ===== 替换为最终 seq =====
    for name, c in collectors.items():
        set_module_by_name(model_student, name, c.seq)

    return model_student.cpu()