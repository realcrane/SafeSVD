from typing import Optional, Tuple
import re

import torch
import torch.nn as nn

from SAFE_SVD.helper.input_collector import (
    _StopForward,
    disable_ar_steps,
    get_module_by_name,
    quiet_everything,
    rollout,
    swallow_stop_and_mute_metrics,
)


def infer_attention_member(name: str) -> Optional[Tuple[str, str, str, str, str, str, str]]:
    """
    Return (role, attention_name, self_name, query_name, key_name, value_name, output_name).
    """
    patterns = {
        "query": r"^(encoder|decoder)\.layers\.\d+\.blocks\.\d+\.attention\.self\.query$",
        "key": r"^(encoder|decoder)\.layers\.\d+\.blocks\.\d+\.attention\.self\.key$",
        "value": r"^(encoder|decoder)\.layers\.\d+\.blocks\.\d+\.attention\.self\.value$",
        "output": r"^(encoder|decoder)\.layers\.\d+\.blocks\.\d+\.attention\.output\.dense$",
    }
    role = None
    for candidate, pattern in patterns.items():
        if re.match(pattern, name):
            role = candidate
            break
    if role is None:
        return None

    if role == "output":
        attention_name = name[: -len(".output.dense")]
    else:
        attention_name = name.rsplit(".self.", 1)[0]
    self_name = f"{attention_name}.self"
    return (
        role,
        attention_name,
        self_name,
        f"{self_name}.query",
        f"{self_name}.key",
        f"{self_name}.value",
        f"{attention_name}.output.dense",
    )


def is_attention_member(name: str) -> bool:
    return infer_attention_member(name) is not None


def _detach_to_device(x, device):
    if torch.is_tensor(x):
        return x.detach().to(device, non_blocking=True)
    if isinstance(x, tuple):
        return tuple(_detach_to_device(v, device) for v in x)
    if isinstance(x, list):
        return [_detach_to_device(v, device) for v in x]
    if isinstance(x, dict):
        return {k: _detach_to_device(v, device) for k, v in x.items()}
    return x


@torch.no_grad()
def attention_self_input_catch(
    trainer,
    model: nn.Module,
    self_name: str,
    calib_dataset,
    device: torch.device,
):
    trainer.model = model
    model = model.to(device).eval()
    target = get_module_by_name(model, self_name)
    inputs = []

    def pre(_m, inp):
        inputs.append(_detach_to_device(inp, device))
        raise _StopForward

    h = target.register_forward_pre_hook(pre)
    try:
        with disable_ar_steps(trainer), swallow_stop_and_mute_metrics(trainer), quiet_everything():
            _ = rollout(trainer, calib_dataset, model_path=getattr(trainer.model.config, "name_or_path", ""))
    finally:
        h.remove()

    if not inputs:
        raise RuntimeError(f"No attention.self inputs were captured for {self_name}.")
    return inputs


@torch.no_grad()
def attention_context_from_cached_inputs(
    model: nn.Module,
    self_name: str,
    cached_inputs,
    device: torch.device,
):
    model = model.to(device).eval()
    self_module = get_module_by_name(model, self_name)
    contexts = []
    for args in cached_inputs:
        args = _detach_to_device(args, device)
        outputs = self_module(*args)
        context = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        contexts.append(context.detach())
    return contexts
