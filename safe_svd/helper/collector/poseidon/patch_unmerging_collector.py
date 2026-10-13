"""Poseidon PatchUnmerging recognition/reconstruction copied from SAFE_SVD helpers."""

from __future__ import annotations

import math
import re
from typing import Optional, Tuple
import torch
import torch.nn as nn
from ..collector import tree_to as _detach_to_device
from .input_collector import get_module_by_name, capture_rollout

def infer_patch_unmerging_member(name: str) -> Optional[Tuple[str, str, str, str]]:
    """Return (role, block_name, upsample_name, mixup_name)."""
    upsample = re.match(r"^decoder\.layers\.\d+\.upsample\.upsample$", name)
    mixup = re.match(r"^decoder\.layers\.\d+\.upsample\.mixup$", name)
    if upsample:
        block = name[: -len(".upsample")]
        return "upsample", block, name, f"{block}.mixup"
    if mixup:
        block = name[: -len(".mixup")]
        return "mixup", block, f"{block}.upsample", name
    return None


def is_patch_unmerging_member(name: str) -> bool:
    return infer_patch_unmerging_member(name) is not None


@torch.no_grad()
def patch_unmerging_mixup_inputs_from_args(
    model: nn.Module,
    block_name: str,
    cached_args,
    device: torch.device,
):
    model = model.to(device).eval()
    block = get_module_by_name(model, block_name)
    mixup_inputs = []

    for args in cached_args:
        input_feature, output_dimensions, time = _detach_to_device(args, device)
        output_height, output_width = output_dimensions
        batch_size, seq_len, hidden_size = input_feature.shape
        input_height = input_width = math.floor(seq_len**0.5)

        input_feature = block.upsample(input_feature)
        input_feature = input_feature.reshape(
            batch_size, input_height, input_width, 2, 2, hidden_size // 2
        )
        input_feature = input_feature.permute(0, 1, 3, 2, 4, 5)
        input_feature = input_feature.reshape(
            batch_size, 2 * input_height, 2 * input_width, hidden_size // 2
        )
        input_feature = block.maybe_crop(input_feature, output_height, output_width)
        input_feature = input_feature.reshape(batch_size, -1, hidden_size // 2)
        input_feature = block.norm(input_feature, time)
        mixup_inputs.append(input_feature.detach())

    return mixup_inputs



@torch.no_grad()
def patch_unmerging_args_catch(trainer, model, block_name, calib_dataset, device, *, rollout_fn=None):
    return capture_rollout(trainer, model, block_name, calib_dataset, device, rollout_fn=rollout_fn)
