from typing import Optional, Tuple
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from SAFE_SVD.helper.input_collector import (
    _concat_if_list,
    _StopForward,
    disable_ar_steps,
    get_module_by_name,
    quiet_everything,
    rollout,
    swallow_stop_and_mute_metrics,
)


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
def two_linear_mlp_input_catch(
    trainer,
    model: nn.Module,
    module_name: str,
    calib_dataset,
    device: torch.device,
    aggregate: str = "concat",
):
    """
    Capture inputs for Poseidon two-Linear MLP members.

    For the first Linear, this is the usual pre-hook input. For the second
    Linear, we hook the first Linear input once and compute first+activation
    locally, avoiding a second forward path to the second Linear.
    """
    spec = infer_two_linear_mlp(module_name)
    if spec is None:
        raise ValueError(f"{module_name} is not a supported two-Linear MLP member")

    kind, block_name, first_name, second_name = spec
    if module_name not in (first_name, second_name):
        raise ValueError(f"{module_name} does not match inferred MLP pair {first_name}, {second_name}")

    trainer.model = model
    model = model.to(device).eval()
    first = get_module_by_name(model, first_name)
    xs = []

    def pre(_m, inp):
        x = inp[0]
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = _concat_if_list(x).detach().to(device)
        if module_name == second_name:
            x = _second_linear_input(model, kind, block_name, first_name, x)
        xs.append(x.detach().cpu())
        raise _StopForward

    h = first.register_forward_pre_hook(pre)
    try:
        with disable_ar_steps(trainer), swallow_stop_and_mute_metrics(trainer), quiet_everything():
            _ = rollout(trainer, calib_dataset, model_path=getattr(trainer.model.config, "name_or_path", ""))
    finally:
        h.remove()

    if not xs:
        raise RuntimeError(f"No inputs were captured for MLP member {module_name}.")
    if aggregate == "list":
        return xs
    return torch.cat(xs, dim=0)


@torch.no_grad()
def mlp_aware_input_catch(
    trainer,
    model: nn.Module,
    module_name: str,
    calib_dataset,
    device: torch.device,
    fallback,
    aggregate: str = "concat",
):
    if is_two_linear_mlp_member(module_name):
        return two_linear_mlp_input_catch(
            trainer,
            model,
            module_name,
            calib_dataset,
            device,
            aggregate=aggregate,
        )
    return fallback(trainer, model, module_name, calib_dataset, device, aggregate=aggregate)
