"""Poseidon cache-aware route; all compression helpers live in this package."""

from __future__ import annotations

from torch.utils.data import Subset
from ..collector import CaptureContext, CapturedPairs, cache_pair, paired_values, feature_rows, tree_to
from .mlp_module_collector import (
    infer_two_linear_mlp, first_linear_to_second_input, mlp_aware_input_catch,
)
from .attention_module_collector import (
    infer_attention_member, attention_context_from_cached_inputs, attention_self_input_catch,
)
from .patch_unmerging_collector import (
    infer_patch_unmerging_member, patch_unmerging_mixup_inputs_from_args, patch_unmerging_args_catch,
)

def collect_poseidon(context: CaptureContext, layer_name: str) -> CapturedPairs:
    """Route by layer structure using the locally copied Poseidon helpers.

    options requires trainer; rollout_fn may provide the model's inference
    callable. Calibration slices default to 64. Cache entries contain upstream
    inputs/arguments, never stale downstream outputs.
    """
    if context.options.get("trainer") is None:
        raise ValueError("Poseidon route_options requires trainer")
    trainer = context.options["trainer"]
    rollout_fn = context.options.get("rollout_fn")
    batch_size = context.options.get("capture_batch_size", 64)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("capture_batch_size must be a positive integer")
    features = context.teacher.get_submodule(layer_name).in_features
    category = context.options.get("dataset_category", "poseidon")
    mlp = infer_two_linear_mlp(layer_name)
    attention = infer_attention_member(layer_name)
    patch = infer_patch_unmerging_member(layer_name)
    entry = context.cache.get(layer_name)
    if attention is not None:
        role, _, self_name, _, _, _, output_name = attention
        entry = context.cache.get(output_name)

    if mlp is not None and layer_name == mlp[3] and entry is not None:
        kind, block_name, first_name, _ = mlp
        transform = first_linear_to_second_input
        for h, hp in entry["pairs"]:
            h = transform(context.teacher, kind, block_name, first_name, tree_to(h, context.device))
            hp = transform(context.student, kind, block_name, first_name, tree_to(hp, context.device))
            yield category, feature_rows(h, features), feature_rows(hp, features)
        return
    if attention is not None and entry is not None:
        if role in {"key", "value"}:
            for args, args_p in entry["pairs"]:
                yield category, feature_rows(args[0], features), feature_rows(args_p[0], features)
            if role == "value":
                entry["context_ready"] = True
            return
        if role == "output" and entry["context_ready"]:
            transform = attention_context_from_cached_inputs
            for args, args_p in entry["pairs"]:
                h = transform(context.teacher, self_name, [args], context.device)[0]
                hp = transform(context.student, self_name, [args_p], context.device)[0]
                yield category, feature_rows(h, features), feature_rows(hp, features)
            return
    if patch is not None and patch[0] == "mixup" and entry is not None:
        transform = patch_unmerging_mixup_inputs_from_args
        for args, args_p in entry["pairs"]:
            h = transform(context.teacher, patch[1], [args], context.device)[0]
            hp = transform(context.student, patch[1], [args_p], context.device)[0]
            yield category, feature_rows(h, features), feature_rows(hp, features)
        return

    cached = []
    for start in range(0, len(context.calibration_data), batch_size):
        subset = Subset(context.calibration_data, list(range(start, min(start + batch_size, len(context.calibration_data)))))
        if attention is not None and role == "query":
            helper = attention_self_input_catch
            teacher_args = helper(trainer, context.teacher, self_name, subset, context.device, rollout_fn=rollout_fn)
            student_args = helper(trainer, context.student, self_name, subset, context.device, rollout_fn=rollout_fn)
            for args, args_p in paired_values(teacher_args, student_args):
                cached.append(cache_pair(context, args, args_p))
                yield category, feature_rows(args[0], features), feature_rows(args_p[0], features)
        elif patch is not None and patch[0] == "upsample":
            helper = patch_unmerging_args_catch
            teacher_args = helper(trainer, context.teacher, patch[1], subset, context.device, rollout_fn=rollout_fn)
            student_args = helper(trainer, context.student, patch[1], subset, context.device, rollout_fn=rollout_fn)
            for args, args_p in paired_values(teacher_args, student_args):
                cached.append(cache_pair(context, args, args_p))
                yield category, feature_rows(args[0], features), feature_rows(args_p[0], features)
        else:
            # Includes the source's shortcut for uncached second MLP members.
            helper = mlp_aware_input_catch
            h = helper(trainer, context.teacher, layer_name, subset, context.device, rollout_fn=rollout_fn)
            hp = helper(trainer, context.student, layer_name, subset, context.device, rollout_fn=rollout_fn)
            h, hp = feature_rows(h, features), feature_rows(hp, features)
            if mlp is not None and layer_name == mlp[2]:
                cached.append(cache_pair(context, h, hp))
            yield category, h, hp
    if mlp is not None and layer_name == mlp[2]:
        context.cache[mlp[3]] = {"pairs": cached}
    elif attention is not None and role == "query":
        context.cache[output_name] = {"pairs": cached, "context_ready": False}
    elif patch is not None and patch[0] == "upsample":
        if not context.ranks or context.ranks.get(patch[3]) is not None:
            context.cache[patch[3]] = {"pairs": cached}


