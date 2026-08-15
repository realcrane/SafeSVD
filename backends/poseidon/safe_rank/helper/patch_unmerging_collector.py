from typing import Optional, Tuple
import math
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


def _detach_to_device(x, device):
    if torch.is_tensor(x):
        return x.detach().to(device, non_blocking=True)
    if isinstance(x, tuple):
        return tuple(_detach_to_device(v, device) for v in x)
    if isinstance(x, list):
        return [_detach_to_device(v, device) for v in x]
    return x


@torch.no_grad()
def patch_unmerging_args_catch(
    trainer,
    model: nn.Module,
    block_name: str,
    calib_dataset,
    device: torch.device,
):
    trainer.model = model
    model = model.to(device).eval()
    target = get_module_by_name(model, block_name)
    args_list = []

    def pre(_m, inp):
        args_list.append(_detach_to_device(inp, device))
        raise _StopForward

    h = target.register_forward_pre_hook(pre)
    try:
        with disable_ar_steps(trainer), swallow_stop_and_mute_metrics(trainer), quiet_everything():
            _ = rollout(trainer, calib_dataset, model_path=getattr(trainer.model.config, "name_or_path", ""))
    finally:
        h.remove()

    if not args_list:
        raise RuntimeError(f"No PatchUnmerging inputs were captured for {block_name}.")
    return args_list


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
