"""MySVD_fast-compatible cross-layer Poseidon input reuse.

MySVD_fast does not independently run a rollout for every linear layer.  It
reuses first-MLP inputs, attention.self inputs, and PatchUnmerging arguments
for their dependent linear layers.  SAFE-SVD keeps the generic covariance and
factorisation code, while this module supplies those same input pairs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Subset

def _mlp_spec(name: str) -> tuple[str, str, str, str] | None:
    match = re.match(r"^(encoder|decoder)\.layers\.\d+\.blocks\.\d+\.(intermediate\.dense|output\.dense)$", name)
    if match:
        block, member = name.rsplit(".", 2)[0], match.group(2)
        first, second = f"{block}.intermediate.dense", f"{block}.output.dense"
        return "swin_ffn", block, first, second
    match = re.match(r"^residual_blocks\.\d+\.\d+\.(pwconv1|pwconv2)$", name)
    if match:
        block = name.rsplit(".", 1)[0]
        return "convnext", block, f"{block}.pwconv1", f"{block}.pwconv2"
    match = re.match(r"^(encoder|decoder)\.layers\.\d+\.blocks\.\d+\.attention\.self\.continuous_position_bias_mlp\.(0|2)$", name)
    if match:
        block = name.rsplit(".", 1)[0]
        return "continuous_position_bias_mlp", block, f"{block}.0", f"{block}.2"
    return None


def _attention_spec(name: str) -> tuple[str, str, str] | None:
    match = re.match(r"^(encoder|decoder)\.layers\.\d+\.blocks\.\d+\.attention\.(self\.(query|key|value)|output\.dense)$", name)
    if not match:
        return None
    attention_name = name.rsplit(".self.", 1)[0] if ".self." in name else name[: -len(".output.dense")]
    return (match.group(3) or "output"), attention_name, f"{attention_name}.self"


def _patch_spec(name: str) -> tuple[str, str, str] | None:
    if re.match(r"^decoder\.layers\.\d+\.upsample\.upsample$", name):
        block = name[: -len(".upsample")]
        return "upsample", block, f"{block}.mixup"
    if re.match(r"^decoder\.layers\.\d+\.upsample\.mixup$", name):
        block = name[: -len(".mixup")]
        return "mixup", block, f"{block}.upsample"
    return None


def _tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    return value


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    return value


def _linear_forward(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return torch.nn.functional.linear(values.to(module.weight.dtype), module.weight, module.bias)
    if isinstance(module, nn.Sequential):
        for child in module:
            values = _linear_forward(child, values)
        return values
    raise TypeError(f"Expected a Linear or low-rank Sequential, got {type(module)!r}.")


def _as_2d(values: torch.Tensor, in_features: int) -> torch.Tensor:
    """Use MySVD_fast's feature-layout normalization before Gram accumulation."""
    values = torch.as_tensor(values)
    if values.ndim == 2 and values.shape[1] == in_features:
        return values
    if values.shape[-1] == in_features:
        return values.reshape(-1, in_features)
    if values.ndim == 2 and values.shape[0] == in_features:
        return values.T
    raise ValueError(f"Expected {in_features} input features, got {tuple(values.shape)}.")


def _input_capture_tools():
    """Import after the runtime has placed the local scOT checkout on sys.path."""
    from backends.poseidon.paper.helper.input_collector import (
        _StopForward,
        disable_ar_steps,
        quiet_everything,
        swallow_stop_and_mute_metrics,
    )

    return _StopForward, disable_ar_steps, quiet_everything, swallow_stop_and_mute_metrics


@dataclass
class MySVDfastCapturePairs:
    """A stateful capture callback with MySVD_fast's reuse dependencies."""

    trainer: Any
    calibration_data: Any
    helpers: dict[str, Any]
    device: str
    batch_size: int
    mlp_cache: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    attention_cache: dict[str, list[tuple[tuple[Any, ...], tuple[Any, ...]]]] = field(default_factory=dict)
    patch_cache: dict[str, list[tuple[tuple[Any, ...], tuple[Any, ...]]]] = field(default_factory=dict)

    @property
    def target(self) -> torch.device:
        return torch.device(self.device)

    def __call__(self, teacher: nn.Module, student: nn.Module, layer_name: str) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        mlp = _mlp_spec(layer_name)
        if mlp:
            yield from self._mlp_pairs(teacher, student, layer_name, mlp)
            return
        attention = _attention_spec(layer_name)
        if attention:
            yield from self._attention_pairs(teacher, student, layer_name, attention)
            return
        patch = _patch_spec(layer_name)
        if patch:
            yield from self._patch_pairs(teacher, student, layer_name, patch)
            return
        yield from self._direct_pairs(teacher, student, layer_name)

    def release_cached_inputs(self, layer_name: str) -> None:
        """Match MySVD_fast's cleanup when a dependent layer is not compressed."""
        mlp = _mlp_spec(layer_name)
        if mlp is not None and layer_name == mlp[3]:
            self.mlp_cache.pop(layer_name, None)
        attention = _attention_spec(layer_name)
        if attention is not None and attention[0] == "output":
            self.attention_cache.pop(attention[1], None)
        patch = _patch_spec(layer_name)
        if patch is not None and patch[0] == "mixup":
            self.patch_cache.pop(layer_name, None)

    def _subsets(self) -> Iterator[Any]:
        for start in range(0, len(self.calibration_data), self.batch_size):
            stop = min(start + self.batch_size, len(self.calibration_data))
            yield Subset(self.calibration_data, list(range(start, stop)))

    def _module(self, model: nn.Module, name: str) -> nn.Module:
        return self.helpers["get_module_by_name"](model, name)

    def _direct(self, model: nn.Module, layer_name: str, subset: Any) -> torch.Tensor:
        layer = self._module(model, layer_name)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"{layer_name} is expected to be nn.Linear.")
        captured = self.helpers["input_catch"](self.trainer, model, layer_name, subset, self.target)
        return _as_2d(captured, layer.in_features).cpu()

    def _direct_pairs(self, teacher: nn.Module, student: nn.Module, layer_name: str) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for subset in self._subsets():
            yield self._direct(teacher, layer_name, subset), self._direct(student, layer_name, subset)

    def _arguments(self, model: nn.Module, module_name: str, subset: Any) -> tuple[Any, ...]:
        self.trainer.model = model
        model.to(self.target).eval()
        captured: list[tuple[Any, ...]] = []

        stop_forward, disable_ar, quiet, swallow_stop = _input_capture_tools()

        def pre_hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            captured.append(_tree_to_cpu(args))
            raise stop_forward

        handle = self._module(model, module_name).register_forward_pre_hook(pre_hook)
        try:
            with disable_ar(self.trainer), swallow_stop(self.trainer), quiet():
                self.helpers["rollout"](self.trainer, subset)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"Expected one {module_name} capture per subset, got {len(captured)}.")
        return captured[0]

    def _mlp_output(self, model: nn.Module, kind: str, block_name: str, first_name: str, values: torch.Tensor) -> torch.Tensor:
        values = _linear_forward(self._module(model, first_name), values.to(self.target))
        block = self._module(model, block_name)
        if kind == "swin_ffn":
            return block.intermediate.intermediate_act_fn(values)
        if kind == "convnext":
            return block.act(values)
        if kind == "continuous_position_bias_mlp":
            return block[1](values)
        raise ValueError(f"Unsupported MLP kind {kind!r}.")

    def _mlp_pairs(self, teacher: nn.Module, student: nn.Module, layer_name: str, spec: tuple[str, str, str, str]) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        kind, block_name, first_name, second_name = spec
        target_layer = self._module(teacher, layer_name)
        assert isinstance(target_layer, nn.Linear)
        if layer_name == second_name and second_name in self.mlp_cache:
            for teacher_values, student_values in self.mlp_cache.pop(second_name):
                yield (
                    _as_2d(self._mlp_output(teacher, kind, block_name, first_name, teacher_values), target_layer.in_features).cpu(),
                    _as_2d(self._mlp_output(student, kind, block_name, first_name, student_values), target_layer.in_features).cpu(),
                )
            return
        cached: list[tuple[torch.Tensor, torch.Tensor]] = []
        for subset in self._subsets():
            teacher_values = self._direct(teacher, first_name, subset)
            student_values = self._direct(student, first_name, subset)
            if layer_name == first_name:
                cached.append((teacher_values, student_values))
                yield teacher_values, student_values
            else:
                yield (
                    _as_2d(self._mlp_output(teacher, kind, block_name, first_name, teacher_values), target_layer.in_features).cpu(),
                    _as_2d(self._mlp_output(student, kind, block_name, first_name, student_values), target_layer.in_features).cpu(),
                )
        if layer_name == first_name:
            self.mlp_cache[second_name] = cached

    def _attention_pairs(self, teacher: nn.Module, student: nn.Module, layer_name: str, spec: tuple[str, str, str]) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        role, attention_name, self_name = spec
        pairs = self.attention_cache.get(attention_name)
        if pairs is None:
            pairs = [(self._arguments(teacher, self_name, subset), self._arguments(student, self_name, subset)) for subset in self._subsets()]
            if role == "query":
                self.attention_cache[attention_name] = pairs
        layer = self._module(teacher, layer_name)
        assert isinstance(layer, nn.Linear)
        if role == "output":
            self.attention_cache.pop(attention_name, None)
            for teacher_args, student_args in pairs:
                teacher_values = self._module(teacher, self_name)(*_tree_to_device(teacher_args, self.target))
                student_values = self._module(student, self_name)(*_tree_to_device(student_args, self.target))
                teacher_values = teacher_values[0] if isinstance(teacher_values, (tuple, list)) else teacher_values
                student_values = student_values[0] if isinstance(student_values, (tuple, list)) else student_values
                yield _as_2d(teacher_values, layer.in_features).cpu(), _as_2d(student_values, layer.in_features).cpu()
            return
        for teacher_args, student_args in pairs:
            yield _as_2d(teacher_args[0], layer.in_features).cpu(), _as_2d(student_args[0], layer.in_features).cpu()

    def _patch_mixup(self, model: nn.Module, block_name: str, args: tuple[Any, ...]) -> torch.Tensor:
        input_feature, output_dimensions, time = _tree_to_device(args, self.target)
        block = self._module(model, block_name)
        output_height, output_width = output_dimensions
        batch_size, sequence_length, hidden_size = input_feature.shape
        input_height = input_width = math.floor(sequence_length**0.5)
        values = block.upsample(input_feature).reshape(batch_size, input_height, input_width, 2, 2, hidden_size // 2)
        values = values.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, 2 * input_height, 2 * input_width, hidden_size // 2)
        values = block.maybe_crop(values, output_height, output_width).reshape(batch_size, -1, hidden_size // 2)
        return block.norm(values, time)

    def _patch_pairs(self, teacher: nn.Module, student: nn.Module, layer_name: str, spec: tuple[str, str, str]) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        role, block_name, paired_name = spec
        layer = self._module(teacher, layer_name)
        assert isinstance(layer, nn.Linear)
        if role == "mixup" and layer_name in self.patch_cache:
            for teacher_args, student_args in self.patch_cache.pop(layer_name):
                yield (
                    _as_2d(self._patch_mixup(teacher, block_name, teacher_args), layer.in_features).cpu(),
                    _as_2d(self._patch_mixup(student, block_name, student_args), layer.in_features).cpu(),
                )
            return
        pairs = [(self._arguments(teacher, block_name, subset), self._arguments(student, block_name, subset)) for subset in self._subsets()]
        if role == "upsample":
            self.patch_cache[paired_name] = pairs
        for teacher_args, student_args in pairs:
            yield _as_2d(teacher_args[0], layer.in_features).cpu(), _as_2d(student_args[0], layer.in_features).cpu()
