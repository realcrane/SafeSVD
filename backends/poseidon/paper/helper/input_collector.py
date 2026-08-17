#coding:utf8
import os
import sys
import argparse
import json
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

import torch
import torch.nn as nn
from tqdm import tqdm
from safesvd.core.whitening import svd_whitener

# Poseidon / ScOT 依赖（仅用于取数据与构建 trainer；不做任何 whitening/profiling）
from poseidon.scOT.inference import get_trainer, rollout
from poseidon.scOT.problems.base import get_dataset

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# Profiling Origin Input
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io, os, logging
from transformers.utils import logging as hf_logging

@contextmanager
def quiet_everything():
    """
    静音：屏蔽 print、tqdm 进度条、HF/一般日志。
    用法:
        with quiet_everything():
            rollout(trainer, dataset, ...)
    """
    # 关进度条
    old_env = {k: os.environ.get(k) for k in ["TQDM_DISABLE", "HF_HUB_DISABLE_PROGRESS_BARS"]}
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # 压低 Transformers 日志级别
    old_hf_level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()

    # 压低 Python logging
    loggers = [
        logging.getLogger(),                         # root
        logging.getLogger("transformers"),
        logging.getLogger("poseidon"),          # 你项目里的 logger（如果有）
    ]
    old_levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)

    # 重定向 stdout/stderr
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        try:
            yield
        finally:
            # 恢复所有设置
            hf_logging.set_verbosity(old_hf_level)
            for lg, lvl in zip(loggers, old_levels):
                lg.setLevel(lvl)
            for k, v in old_env.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v

def to_2d_features(x: torch.Tensor, in_features: int) -> torch.Tensor:
    """把任意形状的层输入压成 [N, in_features]，并做一致性检查。"""
    x = torch.as_tensor(x)
    if x.dim() == 2 and x.shape[1] == in_features:
        return x
    if x.shape[-1] == in_features:
        return x.reshape(-1, in_features)
    if x.dim() == 2 and x.shape[0] == in_features:
        return x.T
    raise ValueError(f"in_features={in_features}, but input shape={tuple(x.shape)} 不匹配")

# =============================================================
# 工具函数（无 Whitening，无 LLM profiling，仅普通 SVD）
# =============================================================

def _concat_if_list(x):
    if isinstance(x, list):
        try:    return torch.cat(x, dim=0)
        except: return x
    return x

def _named_module_dict(model: nn.Module) -> Dict[int, str]:
    return {id(m): name for name, m in model.named_modules()}

def _is_leaf(m: nn.Module) -> bool:
    return len(list(m.children())) == 0

class _StopForward(Exception):
    """我们在目标层 post-hook 抛出的“本 batch 早停”异常"""
    pass

def _module_device(mod: nn.Module):
    for p in mod.parameters(recurse=False): return p.device
    for b in mod.buffers(recurse=False):    return b.device
    return None

def _apply_leaf_module(mod: nn.Module, x: torch.Tensor):
    dev = _module_device(mod)
    if dev is not None and x.device != dev:
        x = x.to(dev, non_blocking=True)
    return mod(x)

def _to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        t = [_to_device(v, device) for v in batch]
        return type(batch)(t) if not isinstance(batch, tuple) else tuple(t)
    return batch

def get_module_by_name(model: nn.Module, dotted: str) -> nn.Module:
    m = model
    for p in dotted.split('.'):
        m = getattr(m, p)
    return m

def set_module_by_name(model: nn.Module, dotted: str, new_mod: nn.Module) -> None:
    parent, attr = dotted.rsplit('.', 1)
    p = get_module_by_name(model, parent)
    setattr(p, attr, new_mod)

@contextmanager
def disable_ar_steps(trainer):
    old_steps = getattr(trainer, "ar_steps", None)
    old_oas   = getattr(trainer, "output_all_steps", False)
    if hasattr(trainer, "set_ar_steps"):
        trainer.set_ar_steps(None, output_all_steps=False)
    try:
        yield
    finally:
        if hasattr(trainer, "set_ar_steps"):
            trainer.set_ar_steps(old_steps, output_all_steps=old_oas)

def get_one_calib_batch(calib_dataset, device):
    """从 calib_dataset 取一个 batch（第一条），并搬到 device。"""
    for b in calib_dataset:
        return _to_device(b, device)
    raise ValueError("calib_dataset 不可迭代，无法获取一个 batch")

# -------------------------------------------------------------
# 前向顺序追踪（仅用于拿到 linear 的执行顺序）
# -------------------------------------------------------------
@torch.no_grad()
def linear_forward_order(trainer, calib_dataset) -> List[str]:
    """用单个 batch 获取按实际执行的 nn.Linear 顺序。"""
    device = next(trainer.model.parameters()).device if any(True for _ in trainer.model.parameters()) else torch.device('cpu')
    batch = get_one_calib_batch(calib_dataset, device)

    model = trainer.model.eval()
    id2name = _named_module_dict(model)
    order: List[str] = []
    seen = set()

    def hook(m, inp, out):
        nm = id2name.get(id(m))
        if nm and nm not in seen and isinstance(m, nn.Linear):
            seen.add(nm)
            order.append(nm)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(hook))

    # 优先尝试直接前向；失败回退到一次 rollout（不早停）
    try:
        if isinstance(batch, dict):
            _ = model(**batch)
        else:
            _ = model(batch)
    except Exception:
        _ = rollout(trainer, calib_dataset, model_path=getattr(trainer.model.config, 'name_or_path', ''))

    for h in hooks:
        h.remove()
    return order

# -------------------------------------------------------------
# 关键补丁：让 rollout(...) 时吞掉 _StopForward，并返回“0 loss”
# 从而实现“整 dataset 一次 rollout，但每个 batch 在目标层早停”
# -------------------------------------------------------------
@contextmanager
def swallow_stop_and_mute_metrics(trainer):
    """rollout 时吞掉我们在 hook 抛出的 _StopForward，并临时关闭 compute_metrics。"""
    def _dev_of(model):
        try:    return next(model.parameters()).device
        except: return torch.device("cpu")

    # 1) mute metrics
    old_metrics = getattr(trainer, "compute_metrics", None)
    trainer.compute_metrics = None

    # 2) patch prediction_step（兜底）
    orig_pred = getattr(trainer, "prediction_step", None)
    if orig_pred is not None:
        def patched_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            try:
                return orig_pred(model, inputs, prediction_loss_only, ignore_keys)
            except _StopForward:
                zero = torch.tensor(0.0, device=_dev_of(model))
                return (zero, None, None)
        trainer.prediction_step = patched_prediction_step.__get__(trainer, type(trainer))

    # 3) patch compute_loss（主路径）
    orig_cl = getattr(trainer, "compute_loss", None)
    if orig_cl is not None:
        def patched_compute_loss(self, model, inputs, return_outputs=False):
            try:
                return orig_cl(model, inputs, return_outputs=return_outputs)
            except _StopForward:
                zero = torch.tensor(0.0, device=_dev_of(model))
                return (zero, {}) if return_outputs else zero
        trainer.compute_loss = patched_compute_loss.__get__(trainer, type(trainer))

    try:
        yield
    finally:
        trainer.compute_metrics = old_metrics
        if orig_pred is not None:
            trainer.prediction_step = orig_pred
        if orig_cl is not None:
            trainer.compute_loss = orig_cl

def _ensure_batched_inputs(inputs: dict) -> dict:
    """
    规范化输入：
      - pixel_values: 保证 [B,C,H,W]
      - time: 无论是 float/int/Tensor，最终都变成 shape [B] 的 Tensor
      - 其它常见标量键同样扩成 [B]
    """
    if "pixel_values" not in inputs or not isinstance(inputs["pixel_values"], torch.Tensor):
        return inputs  # 若没有这个键，直接返回（或按需 raise）

    pv = inputs["pixel_values"]
    if pv.dim() == 3:                  # [C,H,W] -> [1,C,H,W]
        pv = pv.unsqueeze(0)
    elif pv.dim() != 4:
        raise ValueError(f"pixel_values 期望 3/4 维，收到 {tuple(pv.shape)}")
    inputs["pixel_values"] = pv

    B = pv.shape[0]
    device = pv.device
    dtype  = pv.dtype

    # ---- helper: 把值转成 [B] 的 1D Tensor（按需要广播/扩展）----
    def _to_len_B_1d(val, name: str):
        if val is None:
            return None
        t = torch.as_tensor(val, device=device, dtype=dtype)  # 支持 float/int/list/ndarray/Tensor
        if t.dim() == 0:
            t = t.expand(B)                                   # 标量 -> [B]
        elif t.dim() == 1:
            if t.numel() == 1 and B > 1:
                t = t.expand(B)                               # [1] -> [B]
            elif t.numel() != B:
                raise ValueError(f"{name} 长度为 {t.numel()} 与 batch 大小 {B} 不一致")
        elif t.dim() == 2 and t.shape[0] == B and t.shape[1] == 1:
            t = t.squeeze(1)                                  # [B,1] -> [B]
        else:
            raise ValueError(f"{name} 形状 {tuple(t.shape)} 与期望的 [B] 不兼容")
        return t

    # 规范化 time（关键修复点）
    if "time" in inputs:
        inputs["time"] = _to_len_B_1d(inputs["time"], "time")

    # 可选地把一些标量键也扩成 [B]
    for k in ("attention_mask", "mask", "conditioning"):
        if k in inputs:
            try:
                inputs[k] = _to_len_B_1d(inputs[k], k)
            except Exception:
                # 有些键可能是高维结构，这里跳过不强制
                pass

    return inputs

@torch.no_grad()
def forward_until_via_rollout(trainer, module_name: str, dataset, device: torch.device,
                              aggregate='concat'):
    """
    与 rollout 完全同路：一次 rollout 跑完整个 dataset，
    在目标层命中时抛 _StopForward 早停当前 batch，统计输入。
    """
    model = trainer.model.to(device).eval()
    target = get_module_by_name(model, module_name)

    xs = []

    def pre(m, inp):
        x = inp[0]
        if isinstance(x, (list, tuple)): x = x[0]
        xs.append(_concat_if_list(x).detach().cpu())
        # 命中后立刻早停当前 batch
        raise _StopForward

    h = target.register_forward_pre_hook(pre)
    try:
        with disable_ar_steps(trainer), swallow_stop_and_mute_metrics(trainer), quiet_everything():
            # 关键：一次 rollout 覆盖整个 dataset（就是你测评用的那条路）
            _ = rollout(trainer, dataset, model_path=getattr(trainer.model.config, 'name_or_path', ''))
    finally:
        h.remove()

    if not xs:
        raise RuntimeError(f"未在 {module_name} 捕获到任何输入。")

    if aggregate == 'list':
        return xs
    try:
        return torch.cat(xs, dim=0)  # 与 origin_act 同一统计口径
    except Exception:
        return xs

import torch

class _StopForward(Exception):
    pass

def _ensure_bchw(pv: torch.Tensor) -> torch.Tensor:
    # [C,H,W] -> [1,C,H,W]
    if pv.dim() == 3:
        pv = pv.unsqueeze(0)
    return pv

def central_diff_2d(x, dx=1.0, dy=1.0, bc="one-sided"):
    # x: [H,W]
    x = x.unsqueeze(0).unsqueeze(0)  # -> [1,1,H,W]
    B, C, H, W = x.shape
    dfdx = torch.empty_like(x)
    dfdy = torch.empty_like(x)
    if bc == "one-sided":
        dfdx[..., 1:-1] = (x[..., 2:] - x[..., :-2]) / (2*dx)
        dfdx[..., 0]    = (x[..., 1] - x[..., 0]) / dx
        dfdx[..., -1]   = (x[..., -1] - x[..., -2]) / dx

        dfdy[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2*dy)
        dfdy[..., 0, :]    = (x[..., 1, :] - x[..., 0, :]) / dy
        dfdy[..., -1, :]   = (x[..., -1, :] - x[..., -2, :]) / dy
    else:
        raise NotImplementedError
    return dfdx[0,0], dfdy[0,0]

from contextlib import contextmanager
import torch

@contextmanager
def patch_collator_dx(trainer, eps: float = 1e-4):
    """
    修改版 Collator 上下文管理器：
    不再计算空间导数，而是利用 batch['labels'] 计算时间演化方向 (Physics Dynamics)，
    并注入 eps * direction 作为扰动。
    """
    # 保存原始的 collator
    orig = trainer.data_collator

    def collate_with_physics_dt(features):
        # 1. 调用原始 collator 拿到 batch
        batch = orig(features)

        # 2. 检查必要的字段
        # 如果没有 labels，无法计算物理方向，则不做任何扰动
        if "pixel_values" not in batch or "labels" not in batch:
            return batch

        x_current = batch["pixel_values"]  # [B, C, H, W]
        x_future = batch["labels"]         # [B, C, H, W] 通常是下一时刻的真值

        # 3. 维度对齐
        # 某些 dataset 的 labels 可能带有时间维度 [B, T, C, H, W]，通常我们只需要第一帧
        if x_future.ndim > x_current.ndim:
            x_future = x_future.squeeze(1)
        
        # 确保数据类型一致 (Float)
        x_current = x_current.float()
        x_future = x_future.float()

        # 4. 计算物理演化方向 (Velocity / Time Derivative)
        # 必须确保在同一个 device 上计算 (通常 collator 在 CPU 上运行，但也可能在 GPU)
        x_future = x_future.to(x_current.device)
        
        # v = X_{next} - X_{now}
        velocity = x_future - x_current

        # 5. 归一化方向 (Normalization) [关键步骤]
        # 我们只想要“方向”，幅度由 eps 控制。
        # 如果不归一化，eps 就会乘以原本的物理步长，导致扰动过大或过小。
        
        # 拉平计算每张图的 norm
        B = x_current.shape[0]
        flat_v = velocity.reshape(B, -1)
        # 加一个极小值防止除零
        norm = torch.norm(flat_v, dim=1, keepdim=True) + 1e-8
        
        # 恢复形状 [B, 1, 1, 1] 以便广播
        view_shape = [B] + [1] * (x_current.ndim - 1)
        norm = norm.view(*view_shape)
        
        direction = velocity / norm

        # 6. 注入扰动
        # New Input = Current + eps * Physics_Direction
        batch["pixel_values"] = x_current + eps * direction

        return batch

    try:
        # 劫持：将 trainer 的 collator 替换为我们的物理扰动版
        trainer.data_collator = collate_with_physics_dt
        yield
    finally:
        # 还原：退出 with 块后恢复原始 collator
        trainer.data_collator = orig

@torch.no_grad()
def input_catch(trainer, model, module_name, calib_dataset, device, eps=1e-4, aggregate="concat"):
    trainer.model = model
    target = get_module_by_name(model, module_name)

    def collect(ds):
        xs = []
        def pre(m, inp):
            x = inp[0]; x = x[0] if isinstance(x,(list,tuple)) else x
            xs.append(_concat_if_list(x).detach().cpu())
            raise _StopForward
        h = target.register_forward_pre_hook(pre)
        try:
            with disable_ar_steps(trainer), swallow_stop_and_mute_metrics(trainer), quiet_everything():
                _ = rollout(trainer, ds, model_path=getattr(trainer.model.config,'name_or_path',''))
        finally:
            h.remove()
        return xs if aggregate=="list" else torch.cat(xs, dim=0)
    H = collect(calib_dataset)
    return H


@torch.no_grad()
def forward_until_via_rollout_jxs(trainer, module_name, calib_dataset, device, eps=1e-4, aggregate="concat"):
    model = trainer.model.to(device).eval()
    target = get_module_by_name(model, module_name)

    def collect(ds):
        xs = []
        def pre(m, inp):
            x = inp[0]; x = x[0] if isinstance(x,(list,tuple)) else x
            xs.append(_concat_if_list(x).detach().cpu())
            raise _StopForward
        h = target.register_forward_pre_hook(pre)
        try:
            with disable_ar_steps(trainer), swallow_stop_and_mute_metrics(trainer), quiet_everything():
                _ = rollout(trainer, ds, model_path=getattr(trainer.model.config,'name_or_path',''))
        finally:
            h.remove()
        return xs if aggregate=="list" else torch.cat(xs, dim=0)

    # 第一次：原始
    H = collect(calib_dataset)

    # 第二次：只改 collator，数据集本体不变
    with patch_collator_dx(trainer, eps=eps):
        H_eps = collect(calib_dataset)
    # print((H_eps - H).abs().max().item())          # 标量 float

    if aggregate=="list":
        return H, [(he - h)/eps for h,he in zip(H,H_eps)]
    return H, (H_eps - H)/eps

def as_2d_x(x, in_c):
    """
    把 x 规范到 [N, in_c] 的 2D Tensor：
    - 支持 Tensor 或 List[Tensor]
    - 若维度>2，展平成 [*, in_c]
    """
    if isinstance(x, list):
        x = torch.cat(x, dim=0)
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    if x.shape[-1] != in_c:
        raise ValueError(f"in_features={in_c}, but got shape={tuple(x.shape)}")
    return x




import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def is_relevant_linear(name: str, module: nn.Module) -> bool:
    """
    当前先保留所有 nn.Linear。
    如果后面确认某些 norm/head 是冗余支路，可在这里加 keyword 过滤。
    """
    return isinstance(module, nn.Linear)

def forward_and_capture_Z_all_linear_with_grad(
    trainer,
    batch: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    跑一次 model(**batch)，记录所有 nn.Linear 的输出张量 Z_l（保持在 autograd 图里）。
    不 reshape、不 detach。
    返回:
      Y: 最终输出 [B, ...]
      Z_by_layer: {layer_name: Z_l}，每个 Z_l 都 requires_grad=True
    """
    model = trainer.model
    model.eval()
    device = next(iter(batch.values())).device
    model.to(device)

    # 确保参数和中间量都能建图
    for p in model.parameters():
        p.requires_grad_(True)

    Z_by_layer: Dict[str, torch.Tensor] = {}
    handles = []

    def make_fwd_hook(name):
        def hook(mod, inp, out):
            if isinstance(out, (tuple, list)):
                z = out[0]
            else:
                z = out
            Z_by_layer[name] = z
            return out  # 不改动 forward 流
        return hook

    for name, module in model.named_modules():
        if is_relevant_linear(name, module):
            handles.append(module.register_forward_hook(make_fwd_hook(name)))

    outputs = model(**batch)

    if hasattr(outputs, "output"):
        Y = outputs.output
    elif hasattr(outputs, "logits"):
        Y = outputs.logits
    elif isinstance(outputs, (tuple, list)):
        Y = outputs[0]
    else:
        for h in handles:
            h.remove()
        raise RuntimeError(f"Don't know how to get Y from outputs: {type(outputs)}")

    for h in handles:
        h.remove()

    return Y, Z_by_layer

@torch.no_grad()
def jvp_for_layer_fd(
    trainer,
    batch: Dict[str, Any],
    layer_name: str,
    v_sample: torch.Tensor,
    Y_base_sample: torch.Tensor,
    eps: float = 1e-3,
    sample_idx: int = 0,
) -> torch.Tensor:
    """
    对指定 linear 层和指定样本，算有限差分 JVP:
        J_l v ≈ (Y_pert - Y_base) / eps

    参数:
      v_sample: 该层对 sample_idx 的方向向量，shape = Z_l[sample_idx].shape
      Y_base_sample: 未扰动时这一样本的输出，shape = Y[sample_idx].shape
    返回:
      Jv: 与 Y_base_sample 同 shape 的张量
    """
    model = trainer.model
    model.eval()
    device = next(iter(batch.values())).device
    model.to(device)

    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(f"{layer_name} not found in model.named_modules()")
    target_module = modules[layer_name]

    # 在 hook 里只对 sample_idx 加 eps * v
    def hook_fn(mod, inp, out):
        if isinstance(out, (tuple, list)):
            z = out[0]
            rest = out[1:]
        else:
            z = out
            rest = None

        v_full = torch.zeros_like(z)
        v_full[sample_idx] = v_sample

        z_pert = z + eps * v_full

        if rest is not None:
            return (z_pert,) + rest
        else:
            return z_pert

    h = target_module.register_forward_hook(hook_fn)

    outputs_pert = model(**batch)
    if hasattr(outputs_pert, "output"):
        Y_pert = outputs_pert.output
    elif hasattr(outputs_pert, "logits"):
        Y_pert = outputs_pert.logits
    elif isinstance(outputs_pert, (tuple, list)):
        Y_pert = outputs_pert[0]
    else:
        h.remove()
        raise RuntimeError(f"Don't know how to get Y from outputs: {type(outputs_pert)}")

    h.remove()

    Y_pert_sample = Y_pert[sample_idx : sample_idx + 1]
    Jv = (Y_pert_sample - Y_base_sample) / eps
    return Jv

def vjp_for_layer(
    Y_sample: torch.Tensor,
    Z_layer: torch.Tensor,
    u: torch.Tensor,
    sample_idx: int = 0,
) -> torch.Tensor:
    """
    计算给定层的 VJP: J_l^T u

    参数:
      Y_sample: 基础前向时该样本的输出，shape = Y[sample_idx].shape
      Z_layer: 该层所有样本的激活 Z_l，shape=[B,...]，来自 forward_and_capture_Z_all_linear_with_grad
      u: 与 Y_sample 同 shape 的向量 (比如 Jv 的结果)
    返回:
      J^T u 对应该样本的结果，shape = Z_l[sample_idx].shape
    """
    # autograd.grad 的 outputs / grad_outputs 需要 batch 形状匹配
    Y_sample = Y_sample.unsqueeze(0)  # [1, ...]
    u = u.unsqueeze(0)                # [1, ...]

    grad_Z = torch.autograd.grad(
        outputs=Y_sample,
        inputs=Z_layer,
        grad_outputs=u,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0]

    return grad_Z[sample_idx]

def top_eigvec_JtJ_for_layer(
    trainer,
    calib_loader,
    layer_name: str,
    sample_idx: int = 0,
    num_iter: int = 10,
    eps: float = 1e-3,
):
    """
    用 power iteration 估计某一 linear 层的 J_l^T J_l 主特征向量（在该层激活空间的“最敏感方向”）。
    """
    # 取一个 batch（可以扩展成多 batch 累积）
    batch = next(iter(calib_loader))
    batch = trainer._prepare_inputs(batch)

    # 先跑一遍有 grad 的前向，拿到 Y_base 和所有 Z_l
    Y_base, Z_by_layer = forward_and_capture_Z_all_linear_with_grad(trainer, batch)
    if layer_name not in Z_by_layer:
        raise KeyError(f"{layer_name} not captured")
    Z_l = Z_by_layer[layer_name]

    if sample_idx >= Z_l.shape[0]:
        raise ValueError(f"sample_idx {sample_idx} out of range for Z_l with B={Z_l.shape[0]}")

    Y_b = Y_base[sample_idx]          # [..]
    z_shape = Z_l[sample_idx].shape
    d = Z_l[sample_idx].numel()

    # 初始化 v
    v = torch.randn(d, device=Z_l.device)
    v = v / (v.norm() + 1e-9)

    for _ in range(num_iter):
        # 1) Jv: 有限差分（用 no_grad + hook）
        v_sample = v.view(z_shape)
        Jv = jvp_for_layer_fd(
            trainer=trainer,
            batch=batch,
            layer_name=layer_name,
            v_sample=v_sample,
            Y_base_sample=Y_b.unsqueeze(0),
            eps=eps,
            sample_idx=sample_idx,
        )[0]  # shape = Y_b.shape

        # 2) J^T(Jv): 在原图上做 VJP
        JtJv = vjp_for_layer(
            Y_sample=Y_b,
            Z_layer=Z_l,
            u=Jv,
            sample_idx=sample_idx,
        )

        # 3) 归一化，得到新 v
        v_new = JtJv.view(-1)
        nrm = v_new.norm()
        if nrm.item() == 0:
            break
        v = v_new / nrm

    # 返回该层在 sample_idx 上的“最敏感方向”
    return v.view(z_shape).detach().cpu()


@torch.no_grad()
def profile_inputs_only(
    trainer,
    calib_loader,
    dev: str,
    flatten: bool = True,
    to_cpu: bool = True,
    only_linear: bool = True,
    aggregate: str = "concat",
    **kwargs # 吞掉 eps 等不再需要的参数
):
    """
    简化版 Profiler：仅收集每一层的输入 H (Base Activations)。
    移除了物理扰动和梯度计算，以大幅节省内存。
    """
    device = torch.device(dev)
    model = trainer.model.to(device).eval()

    # 辅助函数：判断是否收集该层
    def is_target_module(m):
        return isinstance(m, nn.Linear) if only_linear else True

    # -------- 内部函数：执行一次收集 --------
    def collect_activations() -> Dict[str, List[torch.Tensor]]:
        handles = []
        buckets: Dict[str, List[torch.Tensor]] = {}

        # 定义 Hook
        def make_pre_hook(layer_name: str):
            def _pre(module, inp):
                x = inp[0]
                if isinstance(x, (list, tuple)):
                    x = x[0]
                if flatten:
                    # [B, ..., C] -> [N, C]
                    x = x.reshape(-1, x.shape[-1])      
                
                # 关键：立刻搬到 CPU，防止 GPU 显存溢出
                x = x.detach().to('cpu', copy=True)
                
                buckets.setdefault(layer_name, []).append(x)
            return _pre

        # 注册 Hook
        for name, module in model.named_modules():
            if is_target_module(module):
                handles.append(module.register_forward_pre_hook(make_pre_hook(name)))

        try:
            # 这里的 rollout 只做前向推理，不涉及任何 loss 计算或反向传播
            # 也不涉及 patch_collator 的修改
            with disable_ar_steps(trainer), swallow_stop_and_mute_metrics(trainer), quiet_everything():
                _ = rollout(trainer, calib_loader, model_path=getattr(trainer.model.config, 'name_or_path', ''))
        finally:
            for h in handles:
                h.remove()

        torch.cuda.empty_cache()
        return buckets

    # -------- 开始执行 --------
    print("Collecting Base H (Physics State)...")
    H_lists = collect_activations()

    # -------- 聚合结果 --------
    H_by_layer: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {}
    
    print(f"Aggregating {len(H_lists)} layers...")
    
    for name, batch_list in H_lists.items():
        if aggregate == "concat":
            try:
                # 将 List[Tensor] 合并为一个巨大的 Tensor
                # 注意：这一步如果是超大模型依然可能耗费大量 CPU 内存
                H_by_layer[name] = torch.cat(batch_list, dim=0)
            except Exception as e:
                print(f"[Warning] Concatenation failed for {name}: {e}. Keeping as list.")
                H_by_layer[name] = batch_list
        else:
            H_by_layer[name] = batch_list

    # 清理模型引用
    model = model.cpu()
    torch.cuda.empty_cache()
    
    # 返回 (H, None) 以保持与主代码解包逻辑 H, _ = ... 兼容
    return H_by_layer, None
