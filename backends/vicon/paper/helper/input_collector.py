# =====================================================================
# VICON-specific: Collect per-layer activations via hooks
# =====================================================================
# 【改动说明】
#   1. profile_svd_vicon:        新增 use_trace_align 超参
#   2. collect_covariances_vicon: 新增 use_trace_align 超参
#   3. _trace_align_and_merge:   use_trace_align=False 时退化为简单平均
# =====================================================================
import os
import copy
import json
import math
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from VICON.src import utils
from VICON.src.dataset import all_datasets
import VICON.src.models as models
from VICON.src.trainer import Trainer
from MySVD.convert_attn import convert_attn_to_linear

class StopForwardException(Exception):
    pass

@torch.no_grad()
def collect_layer_input_vicon(
    trainer: Trainer,
    model: nn.Module,
    layer_name: str,
    calib_loaders: dict,
    device: torch.device,
    max_calib_batches: int = None,
) -> torch.Tensor:
    model = model.to(device).eval()
    old_model = trainer.model
    trainer.model = model
    target_module = model.get_submodule(layer_name)
    collected = []

    def pre_hook(module, inp):
        x = inp[0].detach().float()
        C = x.shape[-1]
        X = x.reshape(-1, C)
        collected.append(X.cpu())

    handle = target_module.register_forward_pre_hook(pre_hook)
    try:
        for dataset_type, loader in calib_loaders.items():
            for batch_cnt, batch in enumerate(loader):
                if max_calib_batches is not None and batch_cnt >= max_calib_batches:
                    break
                _, pairs, t_in, t_out, delta_t = batch
                pairs = trainer._move_to_device(pairs)
                data, _, _ = trainer._data_preprocess(pairs)
                _ = trainer._model_forward(data)
    finally:
        handle.remove()
    trainer.model = old_model
    if not collected:
        raise RuntimeError(f"No activations captured for layer: {layer_name}")
    return torch.cat(collected, dim=0)


def as_2d_x(x, in_c):
    if isinstance(x, list):
        x = torch.cat(x, dim=0)
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    if x.shape[-1] != in_c:
        raise ValueError(f"in_features={in_c}, but got shape={tuple(x.shape)}")
    return x


def to_2d_features(x: torch.Tensor, in_features: int) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.dim() == 2 and x.shape[1] == in_features:
        return x
    if x.shape[-1] == in_features:
        return x.reshape(-1, in_features)
    if x.dim() == 2 and x.shape[0] == in_features:
        return x.T
    raise ValueError(f"in_features={in_features}, but input shape={tuple(x.shape)}")


import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import math
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import json

from VICON.src import utils
from VICON.src.dataset import all_datasets
import VICON.src.models as models
from VICON.src.trainer import Trainer
from VICON.src.rollout_utils import rollout_one_traj
from VICON.src.rollout_strategy import gen_strategy_list
from MySVD.helper.whitener import *


# =====================================================================
# Trace 能量对齐工具函数
# =====================================================================

def _trace_align_and_merge(
    per_cat_mats: Dict[str, torch.Tensor],
    use_trace_align: bool = True,
) -> torch.Tensor:
    """
    合并多个类别的矩阵。

    Args:
        per_cat_mats:    {category_name: averaged_matrix}
        use_trace_align: True  → trace 能量对齐后取平均
                         False → 直接简单平均 (原始行为)
    """
    cats = list(per_cat_mats.keys())
    if len(cats) == 0:
        raise ValueError("No categories to merge")
    if len(cats) == 1:
        return per_cat_mats[cats[0]]

    if not use_trace_align:
        # 简单平均，不做任何缩放
        return sum(per_cat_mats[c] for c in cats) / len(cats)

    # trace 对齐
    traces = {c: torch.trace(per_cat_mats[c]).item() for c in cats}
    target_trace = max(traces.values())

    balanced = []
    for c in cats:
        tr = traces[c]
        if tr > 1e-15:
            balanced.append(per_cat_mats[c] * (target_trace / tr))
        else:
            balanced.append(per_cat_mats[c])

    return sum(balanced) / len(balanced)


# =====================================================================
# Step 1: Input Profiling — 按 dataset_type 分桶 (可选 trace 对齐)
# =====================================================================

@torch.no_grad()
def profile_svd_vicon(
    trainer, calib_loaders, cfg, dev,
    max_calib_batches=5,
    use_trace_align: bool = True,      # ← 新增超参
):
    device = torch.device(dev)
    model = trainer.model.to(device).eval()

    layers_dict = {}
    handles = []

    def hook(module, inp, out):
        x = inp[0].detach().float()
        C = x.shape[-1]
        X = x.reshape(-1, C)
        cat = getattr(model, '_current_dataset_type', 'default')
        if cat not in module._gram_per_cat:
            module._gram_per_cat[cat] = torch.zeros(C, C, device=x.device, dtype=torch.float64)
            module._gram_count_per_cat[cat] = 0
        module._gram_per_cat[cat] += (X.T @ X).double()
        module._gram_count_per_cat[cat] += X.shape[0]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._gram_per_cat = {}
            module._gram_count_per_cat = {}
            handles.append(module.register_forward_hook(hook))
            layers_dict[name] = module

    align_tag = "ON" if use_trace_align else "OFF"
    print(f"开始收集特征矩阵 (trace_align={align_tag}) ...")

    utils.set_seed(cfg.train_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.train_seed)

    for dataset_type, calib_loader in calib_loaders.items():
        print(f"  -> 正在收集数据集特征: {dataset_type}")
        model._current_dataset_type = dataset_type

        for batch_cnt, batch in enumerate(tqdm(
            calib_loader, total=max_calib_batches, desc=f"Processing {dataset_type}"
        )):
            if batch_cnt >= max_calib_batches:
                break
            _, pairs, t_in, t_out, delta_t = batch
            pairs = trainer._move_to_device(pairs)
            data, _, _ = trainer._data_preprocess(pairs)
            _ = trainer._model_forward(data)

    for h in handles:
        h.remove()

    profiling_mat = {}
    print("[Profiling] Computing SVD whitening matrices ...")
    for name, module in tqdm(layers_dict.items(), desc="Decomposing Layers"):
        per_cat_avg = {}
        for cat in module._gram_per_cat:
            cnt = module._gram_count_per_cat[cat]
            if cnt > 0:
                per_cat_avg[cat] = (module._gram_per_cat[cat] / cnt).double()

        if len(per_cat_avg) > 0:
            G = _trace_align_and_merge(per_cat_avg, use_trace_align=use_trace_align)
        else:
            G = torch.eye(module.in_features, device=device, dtype=torch.float64)

        Lr, Lr_invT = svd_whitener(G, eps=1e-6)
        profiling_mat[name] = (Lr.cpu().float(), Lr_invT.cpu().float())

        del module._gram_per_cat, module._gram_count_per_cat
        torch.cuda.empty_cache()

    model.cpu()
    torch.cuda.empty_cache()
    return profiling_mat


import torch
import torch.nn as nn
from typing import Dict
from tqdm import tqdm
from contextlib import contextmanager
from torch.utils.checkpoint import checkpoint


# =========================================================================
# Monkey-patch for ICON_CROPPED: non-inplace forward
# =========================================================================

def _make_safe_forward(model_instance):
    from VICON.src.model_utils import patchify, depatchify

    def safe_forward(self, x):
        cfg = self.cfg
        c_out = cfg.transformer.dim_channel
        dim_token = cfg.transformer.dim_token
        p_i = cfg.patch_num_in
        p_e = cfg.patch_num_out
        p_res = cfg.patch_resolution
        demo_num = cfg.demo_num

        init, end, c_mask = x
        bs, _, _, _, _ = init.shape
        feature_init = init.view(-1, *init.shape[-3:])
        feature_end = end.view(-1, *end.shape[-3:])

        feature_init = patchify(feature_init, patch_num=p_i)
        feature_end = patchify(feature_end, patch_num=p_e)
        feature = torch.cat((feature_init, feature_end), dim=1)
        feature = self.pre_proj(feature)

        if self.cfg.use_patch_pos_encoding:
            part_init = feature[:, : p_i * p_i] + self.patch_pos_encoding_init
            part_end  = feature[:, p_i * p_i :]  + self.patch_pos_encoding_end
            feature = torch.cat([part_init, part_end], dim=1)

        feature = feature.view(bs, -1, p_i * p_i + p_e * p_e, dim_token)

        if self.cfg.use_func_pos_encoding:
            part_init = feature[:, :, : p_i * p_i] + self.func_pos_encoding_init.view(1, -1, 1, dim_token)
            part_end  = feature[:, :, p_i * p_i :]  + self.func_pos_encoding_end.view(1, -1, 1, dim_token)
            feature = torch.cat([part_init, part_end], dim=2)

        feature = feature.view(bs, -1, dim_token)
        feature = self.transformer(feature, mask=self.mask)
        feature = feature.view(bs, demo_num, -1, dim_token)
        feature = feature[:, :, : p_i * p_i, :]

        feature = self.post_proj(feature)
        feature = feature.view(bs * demo_num, p_i, p_i, -1)
        feature = feature.view(bs * demo_num, p_e, p_i // p_e, p_e, p_i // p_e, -1)
        feature = feature.permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(bs * demo_num, p_e * p_e, -1)
        feature = depatchify(feature, patch_num=p_e, c=c_out, h=p_res, w=p_res)
        feature = feature.view(bs, demo_num, *feature.shape[-3:])

        return feature

    return safe_forward


def _make_safe_forward_checkpointed(model_instance):
    from VICON.src.model_utils import patchify, depatchify

    def safe_forward_ckpt(self, x):
        cfg = self.cfg
        c_out = cfg.transformer.dim_channel
        dim_token = cfg.transformer.dim_token
        p_i = cfg.patch_num_in
        p_e = cfg.patch_num_out
        p_res = cfg.patch_resolution
        demo_num = cfg.demo_num

        init, end, c_mask = x
        bs, _, _, _, _ = init.shape
        feature_init = init.view(-1, *init.shape[-3:])
        feature_end = end.view(-1, *end.shape[-3:])

        feature_init = patchify(feature_init, patch_num=p_i)
        feature_end = patchify(feature_end, patch_num=p_e)
        feature = torch.cat((feature_init, feature_end), dim=1)
        feature = self.pre_proj(feature)

        if self.cfg.use_patch_pos_encoding:
            part_init = feature[:, : p_i * p_i] + self.patch_pos_encoding_init
            part_end  = feature[:, p_i * p_i :]  + self.patch_pos_encoding_end
            feature = torch.cat([part_init, part_end], dim=1)

        feature = feature.view(bs, -1, p_i * p_i + p_e * p_e, dim_token)

        if self.cfg.use_func_pos_encoding:
            part_init = feature[:, :, : p_i * p_i] + self.func_pos_encoding_init.view(1, -1, 1, dim_token)
            part_end  = feature[:, :, p_i * p_i :]  + self.func_pos_encoding_end.view(1, -1, 1, dim_token)
            feature = torch.cat([part_init, part_end], dim=2)

        feature = feature.view(bs, -1, dim_token)

        mask = self.mask
        for layer in self.transformer.layers:
            feature = checkpoint(
                layer, feature, mask,
                None,
                use_reentrant=False,
            )
        if self.transformer.norm is not None:
            feature = self.transformer.norm(feature)

        feature = feature.view(bs, demo_num, -1, dim_token)
        feature = feature[:, :, : p_i * p_i, :]

        feature = self.post_proj(feature)
        feature = feature.view(bs * demo_num, p_i, p_i, -1)
        feature = feature.view(bs * demo_num, p_e, p_i // p_e, p_e, p_i // p_e, -1)
        feature = feature.permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(bs * demo_num, p_e * p_e, -1)
        feature = depatchify(feature, patch_num=p_e, c=c_out, h=p_res, w=p_res)
        feature = feature.view(bs, demo_num, *feature.shape[-3:])

        return feature

    return safe_forward_ckpt


@contextmanager
def _patched_forward(model, use_gradient_checkpointing=False):
    raw_model = model.module if hasattr(model, "module") else model
    cls_name = type(raw_model).__name__

    original_forward = None
    if cls_name == "ICON_CROPPED":
        original_forward = type(raw_model).forward
        if use_gradient_checkpointing:
            type(raw_model).forward = _make_safe_forward_checkpointed(raw_model)
            print(f"[Fisher] Patched {cls_name}.forward (non-inplace + gradient checkpointing)")
        else:
            type(raw_model).forward = _make_safe_forward(raw_model)
            print(f"[Fisher] Patched {cls_name}.forward (non-inplace)")

    inplace_states = {}
    for name, m in model.named_modules():
        if hasattr(m, 'inplace') and m.inplace:
            inplace_states[name] = True
            m.inplace = False

    try:
        yield
    finally:
        if original_forward is not None:
            type(raw_model).forward = original_forward
            print(f"[Fisher] Restored original {cls_name}.forward")
        for name, m in model.named_modules():
            if name in inplace_states:
                m.inplace = True


# =====================================================================
# collect_covariances_vicon — 按 dataset_type 分桶 (可选 trace 对齐)
# =====================================================================

@torch.no_grad()
def collect_covariances_vicon(
    trainer: Trainer,
    model_teacher: nn.Module,
    model_student: nn.Module,
    layer_name: str,
    cached_batches: list,             # list of (dataset_type, batch)
    device: torch.device,
    max_calib_batches: int = None,
    use_trace_align: bool = True,     # ← 新增超参
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    分批次计算协方差矩阵，按 dataset_type 分桶。
    use_trace_align=True  → trace 对齐后合并
    use_trace_align=False → 简单平均合并 (原始行为)
    """
    model_teacher = model_teacher.to(device).eval()
    model_student = model_student.to(device).eval()

    target_teacher = model_teacher.get_submodule(layer_name)
    target_student = model_student.get_submodule(layer_name)
    in_c = target_teacher.in_features

    cat_sigma_HH:   Dict[str, torch.Tensor] = {}
    cat_sigma_HHp:  Dict[str, torch.Tensor] = {}
    cat_sigma_HpHp: Dict[str, torch.Tensor] = {}
    cat_counts:     Dict[str, int] = {}

    teacher_inputs = []
    student_inputs = []

    def hook_teacher(module, inp):
        x = inp[0].detach().float()
        teacher_inputs.append(x.reshape(-1, x.shape[-1]))
        raise StopForwardException("Teacher collected, stopping forward pass.")

    def hook_student(module, inp):
        x = inp[0].detach().float()
        student_inputs.append(x.reshape(-1, x.shape[-1]))
        raise StopForwardException("Student collected, stopping forward pass.")

    h_t = target_teacher.register_forward_pre_hook(hook_teacher)
    h_s = target_student.register_forward_pre_hook(hook_student)

    old_model = trainer.model

    try:
        for dataset_type, batch in cached_batches:
            teacher_inputs.clear()
            student_inputs.clear()

            _, pairs, t_in, t_out, delta_t = batch
            pairs = trainer._move_to_device(pairs)
            data, _, _ = trainer._data_preprocess(pairs)

            trainer.model = model_teacher
            try:
                _ = trainer._model_forward(data)
            except StopForwardException:
                pass

            trainer.model = model_student
            try:
                _ = trainer._model_forward(data)
            except StopForwardException:
                pass

            if teacher_inputs and student_inputs:
                H_batch = torch.cat(teacher_inputs, dim=0).to(device, non_blocking=True).double()
                Hp_batch = torch.cat(student_inputs, dim=0).to(device, non_blocking=True).double()

                cat = dataset_type
                if cat not in cat_sigma_HH:
                    cat_sigma_HH[cat]   = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
                    cat_sigma_HHp[cat]  = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
                    cat_sigma_HpHp[cat] = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
                    cat_counts[cat] = 0

                cat_sigma_HH[cat]   += H_batch.T @ H_batch
                cat_sigma_HHp[cat]  += H_batch.T @ Hp_batch
                cat_sigma_HpHp[cat] += Hp_batch.T @ Hp_batch
                cat_counts[cat] += H_batch.shape[0]

                del H_batch, Hp_batch
                torch.cuda.empty_cache()

    finally:
        h_t.remove()
        h_s.remove()
        trainer.model = old_model

    # --- 合并 ---
    cats = list(cat_sigma_HH.keys())
    if not cats:
        z = torch.zeros((in_c, in_c), device=device, dtype=torch.float64)
        return z, z, z

    avg_HH   = {c: cat_sigma_HH[c]   / max(cat_counts[c], 1) for c in cats}
    avg_HHp  = {c: cat_sigma_HHp[c]  / max(cat_counts[c], 1) for c in cats}
    avg_HpHp = {c: cat_sigma_HpHp[c] / max(cat_counts[c], 1) for c in cats}

    if use_trace_align and len(cats) > 1:
        # 以 Teacher (HH) 的 trace 推导缩放系数
        traces_HH = {c: torch.trace(avg_HH[c]).item() for c in cats}
        target_trace = max(traces_HH.values())

        scale_factors = {}
        for c in cats:
            tr = traces_HH[c]
            scale_factors[c] = (target_trace / tr) if tr > 1e-15 else 1.0

        balanced_HH   = [avg_HH[c]   * scale_factors[c] for c in cats]
        balanced_HHp  = [avg_HHp[c]  * scale_factors[c] for c in cats]
        balanced_HpHp = [avg_HpHp[c] * scale_factors[c] for c in cats]
    else:
        # 简单平均，不缩放
        balanced_HH   = [avg_HH[c]   for c in cats]
        balanced_HHp  = [avg_HHp[c]  for c in cats]
        balanced_HpHp = [avg_HpHp[c] for c in cats]

    sigma_HH   = sum(balanced_HH)   / len(balanced_HH)
    sigma_HHp  = sum(balanced_HHp)  / len(balanced_HHp)
    sigma_HpHp = sum(balanced_HpHp) / len(balanced_HpHp)

    return sigma_HH, sigma_HHp, sigma_HpHp