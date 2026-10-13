"""Poseidon MLP recognition/reconstruction copied from SAFE_SVD helpers."""

from __future__ import annotations

import re
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .input_collector import _concat_if_list, get_module_by_name, capture_rollout, input_catch

MLPKind = str

def infer_two_linear_mlp(name: str) -> Optional[Tuple[MLPKind, str, str, str]]:
    """Return (kind, block_name, first_linear_name, second_linear_name)."""
    swin_ffn_fc1 = re.match(r"^(encoder|decoder)\.layers\.\d+\.blocks\.\d+\.intermediate\.dense$", name)
    swin_ffn_fc2 = re.match(r"^(encoder|decoder)\.layers\.\d+\.blocks\.\d+\.output\.dense$", name)
    convnext_fc1 = re.match(r"^residual_blocks\.\d+\.\d+\.pwconv1$", name)
    convnext_fc2 = re.match(r"^residual_blocks\.\d+\.\d+\.pwconv2$", name)
    cpb_fc1 = re.match(
        r"^(encoder|decoder)\.layers\.\d+\.blocks\.\d+\.attention\.self\.continuous_position_bias_mlp\.0$",
        name,
    )
    cpb_fc2 = re.match(
        r"^(encoder|decoder)\.layers\.\d+\.blocks\.\d+\.attention\.self\.continuous_position_bias_mlp\.2$",
        name,
    )

    if swin_ffn_fc1:
        block = name[: -len(".intermediate.dense")]
        return "swin_ffn", block, name, f"{block}.output.dense"
    if swin_ffn_fc2:
        block = name[: -len(".output.dense")]
        return "swin_ffn", block, f"{block}.intermediate.dense", name
    if convnext_fc1:
        block = name[: -len(".pwconv1")]
        return "convnext", block, name, f"{block}.pwconv2"
    if convnext_fc2:
        block = name[: -len(".pwconv2")]
        return "convnext", block, f"{block}.pwconv1", name
    if cpb_fc1:
        block = name[: -len(".0")]
        return "continuous_position_bias_mlp", block, name, f"{block}.2"
    if cpb_fc2:
        block = name[: -len(".2")]
        return "continuous_position_bias_mlp", block, f"{block}.0", name
    return None


def is_two_linear_mlp_member(name: str) -> bool:
    return infer_two_linear_mlp(name) is not None


def _linear_like_forward_no_hooks(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        if x.dtype != module.weight.dtype:
            x = x.to(module.weight.dtype)
        return F.linear(x, module.weight, module.bias)
    if isinstance(module, nn.Sequential):
        for child in module:
            x = _linear_like_forward_no_hooks(child, x)
        return x
    raise TypeError(
        f"Unsupported first MLP module type {type(module)!r}; expected nn.Linear "
        "or the low-rank nn.Sequential used by SAFE-SVD."
    )


def _second_linear_input(
    model: nn.Module,
    kind: MLPKind,
    block_name: str,
    first_name: str,
    x: torch.Tensor,
) -> torch.Tensor:
    block = get_module_by_name(model, block_name)
    first = get_module_by_name(model, first_name)
    if kind == "swin_ffn":
        x = _linear_like_forward_no_hooks(first, x)
        return block.intermediate.intermediate_act_fn(x)
    if kind == "convnext":
        x = _linear_like_forward_no_hooks(first, x)
        return block.act(x)
    if kind == "continuous_position_bias_mlp":
        x = _linear_like_forward_no_hooks(first, x)
        return block[1](x)
    raise ValueError(f"Unknown MLP kind: {kind}")


def first_linear_to_second_input(
    model: nn.Module,
    kind: MLPKind,
    block_name: str,
    first_name: str,
    x: torch.Tensor,
) -> torch.Tensor:
    return _second_linear_input(model, kind, block_name, first_name, x)



@torch.no_grad()
def two_linear_mlp_input_catch(trainer, model, module_name, calib_dataset, device, aggregate="concat", *, rollout_fn=None):
    spec = infer_two_linear_mlp(module_name)
    if spec is None:
        raise ValueError(f"{module_name} is not a two-linear MLP member")
    kind, block_name, first_name, second_name = spec
    def transform(args):
        values = args[0]
        if isinstance(values, (list, tuple)):
            values = values[0]
        values = _concat_if_list(values).detach().to(device)
        if module_name == second_name:
            values = _second_linear_input(model, kind, block_name, first_name, values)
        return values.detach().cpu()
    captured = capture_rollout(
        trainer, model, first_name, calib_dataset, device,
        transform=transform, rollout_fn=rollout_fn,
    )
    return captured if aggregate == "list" else torch.cat(captured, dim=0)


@torch.no_grad()
def mlp_aware_input_catch(trainer, model, module_name, calib_dataset, device, aggregate="concat", *, rollout_fn=None):
    capture = two_linear_mlp_input_catch if is_two_linear_mlp_member(module_name) else input_catch
    return capture(trainer, model, module_name, calib_dataset, device, aggregate=aggregate, rollout_fn=rollout_fn)
