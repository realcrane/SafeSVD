"""Poseidon attention recognition/reconstruction copied from SAFE_SVD helpers."""

from __future__ import annotations

import re
from typing import Optional, Tuple
import torch
import torch.nn as nn
from ..collector import tree_to as _detach_to_device
from .input_collector import get_module_by_name, capture_rollout

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



@torch.no_grad()
def attention_self_input_catch(trainer, model, self_name, calib_dataset, device, *, rollout_fn=None,
                               capture_device=None):
    return capture_rollout(trainer, model, self_name, calib_dataset, device,
                           rollout_fn=rollout_fn, capture_device=capture_device)
