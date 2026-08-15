"""Accelerated sequential SAFE-SVD compression for an already-loaded model."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Subset
from tqdm import tqdm

from SAFE_SVD.helper.attention_module_collector import (
    attention_context_from_cached_inputs,
    attention_self_input_catch,
    infer_attention_member,
    is_attention_member,
)
from SAFE_SVD.helper.input_collector import (
    get_module_by_name,
    input_catch,
    set_module_by_name,
    to_2d_features,
)
from SAFE_SVD.helper.mlp_module_collector import (
    first_linear_to_second_input,
    infer_two_linear_mlp,
    is_two_linear_mlp_member,
    mlp_aware_input_catch,
)
from SAFE_SVD.helper.patch_unmerging_collector import (
    infer_patch_unmerging_member,
    is_patch_unmerging_member,
    patch_unmerging_args_catch,
    patch_unmerging_mixup_inputs_from_args,
)
from SAFE_SVD.helper.perf_logger import make_perf_logger
from SAFE_SVD.helper.whitener import (
    fisher_cholesky_whitener,
    resolve_whitening_eps,
    svd_whitener,
)


def _tree_detach_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, tuple):
        return tuple(_tree_detach_to_cpu(x) for x in obj)
    if isinstance(obj, list):
        return [_tree_detach_to_cpu(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _tree_detach_to_cpu(v) for k, v in obj.items()}
    return obj


def _tree_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, tuple):
        return tuple(_tree_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_tree_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _tree_to_device(v, device) for k, v in obj.items()}
    return obj


def _extend_cache(cache_list, values, offload_to_cpu=False):
    cache_list.extend(_tree_detach_to_cpu(values) if offload_to_cpu else values)


def _append_cache(cache_list, value, offload_to_cpu=False):
    cache_list.append(_tree_detach_to_cpu(value) if offload_to_cpu else value)


def _stable_svd(Mstar, layer_name):
    try:
        return torch.linalg.svd(Mstar, full_matrices=False)
    except RuntimeError as exc:
        if "linalg.svd" not in str(exc):
            raise
        print(
            f"[SVD fallback] {layer_name}: torch.linalg.svd failed on "
            f"{Mstar.device}/{Mstar.dtype}; retrying on CPU/float64."
        )
        original_device = Mstar.device
        original_dtype = Mstar.dtype
        Mstar_cpu = torch.nan_to_num(Mstar.detach().cpu().double())
        U, S, VT = torch.linalg.svd(Mstar_cpu, full_matrices=False)
        return (
            U.to(device=original_device, dtype=original_dtype, non_blocking=True),
            S.to(device=original_device, dtype=original_dtype, non_blocking=True),
            VT.to(device=original_device, dtype=original_dtype, non_blocking=True),
        )


def _accumulate_tensor_pair(
    H,
    Hp,
    device,
    in_c,
    sigma_HHp,
    sigma_HH,
    sigma_HpHp,
    accum_device=None,
    accum_dtype=torch.float64,
):
    accum_device = accum_device or device
    H = to_2d_features(H, in_c).to(device, non_blocking=True)
    H = to_2d_features(H, in_c)
    Hp = to_2d_features(Hp, in_c).to(device, non_blocking=True)
    Hp = to_2d_features(Hp, in_c)

    count = H.shape[0]
    H_acc = H.to(device=accum_device, dtype=accum_dtype, non_blocking=True)
    Hp_acc = Hp.to(device=accum_device, dtype=accum_dtype, non_blocking=True)

    sigma_HpHp.addmm_(Hp_acc.T, Hp_acc)
    sigma_HH.addmm_(H_acc.T, H_acc)
    sigma_HHp.addmm_(H_acc.T, Hp_acc)
    del H_acc, Hp_acc
    del H, Hp
    return count


def _accumulate_cached_inputs(
    cache_entry,
    device,
    in_c,
    sigma_HHp,
    sigma_HH,
    sigma_HpHp,
    accum_device=None,
    accum_dtype=torch.float64,
):
    accumulated_count = 0
    cache_type = cache_entry.get("type", "direct")

    if cache_type == "mlp_lazy":
        for H, Hp in zip(cache_entry["teacher_first_inputs"], cache_entry["student_first_inputs"]):
            H2 = first_linear_to_second_input(
                cache_entry["model_teacher"],
                cache_entry["kind"],
                cache_entry["block_name"],
                cache_entry["first_name"],
                H.to(device, non_blocking=True),
            )
            Hp2 = first_linear_to_second_input(
                cache_entry["model_student"],
                cache_entry["kind"],
                cache_entry["block_name"],
                cache_entry["first_name"],
                Hp.to(device, non_blocking=True),
            )
            accumulated_count += _accumulate_tensor_pair(
                H2, Hp2, device, in_c, sigma_HHp, sigma_HH, sigma_HpHp,
                accum_device=accum_device, accum_dtype=accum_dtype
            )
            del H2, Hp2
            torch.cuda.empty_cache()
        return accumulated_count

    if cache_type == "attention_context_lazy":
        self_name = cache_entry["self_name"]
        for teacher_args, student_args in zip(cache_entry["teacher_self_inputs"], cache_entry["student_self_inputs"]):
            teacher_args = _tree_to_device(teacher_args, device)
            student_args = _tree_to_device(student_args, device)
            H = attention_context_from_cached_inputs(
                cache_entry["model_teacher"], self_name, [teacher_args], device
            )[0]
            Hp = attention_context_from_cached_inputs(
                cache_entry["model_student"], self_name, [student_args], device
            )[0]
            accumulated_count += _accumulate_tensor_pair(
                H, Hp, device, in_c, sigma_HHp, sigma_HH, sigma_HpHp,
                accum_device=accum_device, accum_dtype=accum_dtype
            )
            del H, Hp
            torch.cuda.empty_cache()
        return accumulated_count

    if cache_type == "patch_unmerging_lazy":
        block_name = cache_entry["block_name"]
        for teacher_args, student_args in zip(cache_entry["teacher_args"], cache_entry["student_args"]):
            teacher_args = _tree_to_device(teacher_args, device)
            student_args = _tree_to_device(student_args, device)
            H = patch_unmerging_mixup_inputs_from_args(
                cache_entry["model_teacher"], block_name, [teacher_args], device
            )[0]
            Hp = patch_unmerging_mixup_inputs_from_args(
                cache_entry["model_student"], block_name, [student_args], device
            )[0]
            accumulated_count += _accumulate_tensor_pair(
                H, Hp, device, in_c, sigma_HHp, sigma_HH, sigma_HpHp,
                accum_device=accum_device, accum_dtype=accum_dtype
            )
            del H, Hp
            torch.cuda.empty_cache()
        return accumulated_count

    for H, Hp in zip(cache_entry["teacher_inputs"], cache_entry["student_inputs"]):
        accumulated_count += _accumulate_tensor_pair(
            H, Hp, device, in_c, sigma_HHp, sigma_HH, sigma_HpHp,
            accum_device=accum_device, accum_dtype=accum_dtype
        )
    return accumulated_count


def _hidden_inputs_from_attention_cache(cache_entry):
    return {
        "teacher_inputs": [args[0] for args in cache_entry["teacher_self_inputs"]],
        "student_inputs": [args[0] for args in cache_entry["student_self_inputs"]],
    }

@torch.no_grad()
def compress(
    trainer,
    calibration_dataset,
    ranks,
    *,
    device_name: str,
    left_metrics=None,
    alpha: float = 0.7,
    eps_whiten: float = 1e-12,
    eps_whiten_mode: str = "absolute",
    whitening_mode: str = "origin",
    save_path: Path | None = None,
    checkpoint_percent: float = 0.1,
    low_gpu: bool = False,
    overwrite_checkpoint: bool = False,
    perf_csv_path: str | None = None,
    perf_run_id: str | None = None,
):
    """Return a compressed CPU model using MLP, attention, and PatchUnmerging caches."""
    if whitening_mode not in {"origin", "new"}:
        raise ValueError(
            f"Unknown whitening_mode={whitening_mode!r}; expected 'origin' or 'new'."
        )
    dev = device_name
    calib_dataset = calibration_dataset
    Ldict = left_metrics
    allow_zero_rank = True
    overwrite_cache = overwrite_checkpoint
    accum_dtype = torch.float64
    device = torch.device(dev)
    # low-gpu keeps covariance/SVD on GPU for throughput, but switches to
    # float32 and offloads reusable activation caches to CPU.
    accum_device = device
    accum_dtype = torch.float32 if low_gpu else accum_dtype
    perf = make_perf_logger(
        perf_csv_path,
        device=device,
        run_id=perf_run_id,
        phase="compression",
        method="safe_svd_low_gpu" if low_gpu else "safe_svd_accelerated",
        alpha=alpha,
    )
    trainer.set_ar_steps(None, output_all_steps=False)
    model_teacher: nn.Module = trainer.model.to(device).eval()

    # --- 尝试加载 checkpoint ---
    ckpt_path = os.path.join(save_path, "compressing_model.pt") if save_path else None
    final_path = os.path.join(save_path, "compressed_model.pt") if save_path else None

    if ckpt_path and os.path.exists(ckpt_path) and (final_path and not os.path.exists(final_path)) and not overwrite_cache:
        print(f"==== Found checkpoint, resuming from {ckpt_path} ====")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_student = ckpt['model'].to(device).eval()
        compressed_layers = set(ckpt['compressed_layers'])
        print(f"  Already compressed {len(compressed_layers)} layers: {sorted(compressed_layers)[:5]}...")
    else:
        if ckpt_path and os.path.exists(ckpt_path) and overwrite_cache:
            print(f"==== Ignoring existing checkpoint because --overwrite_cache is set: {ckpt_path} ====")
        model_student = copy.deepcopy(model_teacher)
        compressed_layers = set()

    if calib_dataset is None:
        raise ValueError("Need calib_dataset")

    # dict(ranks) -> {layer_name: int | None}; avoids re-reading rank JSON in this process.
    loaded_ranks = dict(ranks)

    linear_names = [(name, m) for name, m in model_teacher.named_modules() if isinstance(m, nn.Linear)]
    total_layers = len(linear_names)
    checkpoint_interval = max(int(total_layers * checkpoint_percent), 1)

    mlp_layers = sum(1 for name, _ in linear_names if is_two_linear_mlp_member(name))
    attention_layers = sum(1 for name, _ in linear_names if is_attention_member(name))
    patch_unmerging_layers = sum(1 for name, _ in linear_names if is_patch_unmerging_member(name))
    print(
        f"Start accelerated SAFE-SVD compression on {total_layers} linear layers "
        f"({mlp_layers} two-linear MLP members use pair-cache collection; "
        f"{attention_layers} attention QKV/output members use transformer-block cache; "
        f"{patch_unmerging_layers} PatchUnmerging members use reshape/norm cache; "
        f"{'GPU float32 covariance with CPU activation-cache offload; ' if low_gpu else ''}"
        f"checkpoint every {checkpoint_interval} layers)..."
    )
    layers_since_last_save = 0
    mlp_pair_cache = {}
    attention_cache = {}
    patch_unmerging_cache = {}

    def release_unconsumed_pair_cache(layer_name: str) -> None:
        """Release a predecessor cache when its consumer layer is skipped.

        The caches below are only useful until the paired consumer is processed:
        MLP first -> second, attention query -> output, and PatchUnmerging
        upsample -> mixup.  A skipped consumer used to ``continue`` before the
        normal cleanup below, keeping its teacher/student activations alive for
        the rest of a high-GPU run.
        """
        released = []

        mlp_spec = infer_two_linear_mlp(layer_name)
        if mlp_spec is not None:
            _, _, _, second_name = mlp_spec
            if layer_name == second_name and mlp_pair_cache.pop(layer_name, None) is not None:
                released.append("MLP")

        attention_spec = infer_attention_member(layer_name)
        if attention_spec is not None:
            attention_role, attention_name, *_ = attention_spec
            if attention_role == "output" and attention_cache.pop(attention_name, None) is not None:
                released.append("attention")

        patch_spec = infer_patch_unmerging_member(layer_name)
        if patch_spec is not None:
            patch_role, _, _, mixup_name = patch_spec
            if patch_role == "mixup" and patch_unmerging_cache.pop(mixup_name, None) is not None:
                released.append("PatchUnmerging")

        if released:
            print(f"[Cache cleanup] {layer_name}: released unconsumed {'/'.join(released)} cache.")
            torch.cuda.empty_cache()

    for j, (name, lin) in enumerate(tqdm(linear_names)):
        # --- 跳过已压缩的层 ---
        if name in compressed_layers:
            release_unconsumed_pair_cache(name)
            print(f"[{j:02d}] {name}: Skipped (already compressed)")
            continue

        if name in loaded_ranks:
          k = loaded_ranks[name]
          if k is None:
            release_unconsumed_pair_cache(name)
            compressed_layers.add(name)
            layers_since_last_save += 1
            continue
          k = int(k)
        else:
          release_unconsumed_pair_cache(name)
          compressed_layers.add(name)
          layers_since_last_save += 1
          continue

        in_c, out_c = lin.in_features, lin.out_features
        if allow_zero_rank and k == 0:
            release_unconsumed_pair_cache(name)
            with perf.stage("compression/zero_rank_layer", note=f"layer={name};out={out_c};in={in_c}"):
                zero_lin = get_module_by_name(model_student, name)
                if not isinstance(zero_lin, nn.Linear):
                    raise TypeError(f"{name} is expected to be nn.Linear, got {type(zero_lin)}")
                zero_lin.weight.data.zero_()
                if lin.bias is not None and zero_lin.bias is not None:
                    zero_lin.bias.data.copy_(lin.bias.data)
            compressed_layers.add(name)
            layers_since_last_save += 1
            print(f"[{j:02d}] {name}: Zero-rank compressed (weight zeroed, bias kept)")

            if save_path and layers_since_last_save >= checkpoint_interval:
                with perf.stage("compression/save_checkpoint", note=f"layers={len(compressed_layers)}/{total_layers}"):
                    os.makedirs(save_path, exist_ok=True)
                    torch.save({
                        'model': model_student.cpu(),
                        'compressed_layers': list(compressed_layers),
                    }, ckpt_path)
                    model_student = model_student.to(device)
                    layers_since_last_save = 0
                    pct = len(compressed_layers) / total_layers * 100
                    print(f"  [Checkpoint] Saved at {pct:.1f}% ({len(compressed_layers)}/{total_layers} layers)")
            continue

        num_W  = out_c * in_c
        num_UV = out_c * k + k * in_c
        if num_UV >= num_W:
            release_unconsumed_pair_cache(name)
            with perf.stage("compression/skip_full_rank_layer", note=f"layer={name};rank={k};out={out_c};in={in_c}"):
                print(f"[{j:02d}] {name}: Skipped (Full Rank better) k={k}")
                compressed_layers.add(name)
                layers_since_last_save += 1
            continue

        mlp_spec = infer_two_linear_mlp(name)
        cache_entry = None
        if mlp_spec is not None:
            kind, _block_name, first_name, second_name = mlp_spec
            if name == second_name:
                cache_entry = mlp_pair_cache.pop(name, None)
                if cache_entry is not None:
                    print(
                        f"[MLP pair-cache hit] {name}: reusing activation({first_name}) "
                        f"cached from {cache_entry['num_batches']} batches; no forward-to-fc2 rollout."
                    )
                else:
                    print(
                        f"[MLP shortcut active] {name}: collecting input at {first_name} "
                        f"and computing {kind} first-linear+activation locally."
                    )
        cache_teacher_first_inputs = []
        cache_student_first_inputs = []
        cache_second_name = None
        if mlp_spec is not None:
            kind, block_name, first_name, second_name = mlp_spec
            second_rank = loaded_ranks.get(second_name)
            if name == first_name and second_rank is not None:
                cache_second_name = second_name

        attention_spec = infer_attention_member(name)
        attention_role = attention_name = attention_self_name = None
        attention_self_inputs_teacher = []
        attention_self_inputs_student = []
        if attention_spec is not None:
            (
                attention_role,
                attention_name,
                attention_self_name,
                attention_query_name,
                attention_key_name,
                attention_value_name,
                attention_output_name,
            ) = attention_spec
            if attention_role == "query":
                print(
                    f"[Attention cache store] {name}: collecting {attention_self_name} "
                    "inputs once for query/key/value/output reuse."
                )
            elif attention_role in ("key", "value"):
                attn_entry = attention_cache.get(attention_name)
                if attn_entry is not None:
                    cache_entry = _hidden_inputs_from_attention_cache(attn_entry)
                    print(
                        f"[Attention cache hit] {name}: reusing cached input from "
                        f"{attention_query_name}; no forward-to-{attention_role} rollout."
                    )
            elif attention_role == "output":
                attn_entry = attention_cache.get(attention_name)
                if attn_entry is not None and attn_entry.get("context_ready", False):
                    cache_entry = {
                        "type": "attention_context_lazy",
                        "model_teacher": model_teacher,
                        "model_student": model_student,
                        "self_name": attention_self_name,
                        "teacher_self_inputs": attn_entry["teacher_self_inputs"],
                        "student_self_inputs": attn_entry["student_self_inputs"],
                        "num_batches": len(attn_entry["teacher_self_inputs"]),
                    }
                    print(
                        f"[Attention cache hit] {name}: reusing cached self-attention context; "
                        "lazy per-batch context compute, no forward-to-output rollout."
                    )

        patch_unmerging_spec = infer_patch_unmerging_member(name)
        patch_role = patch_block_name = patch_mixup_name = None
        patch_teacher_args = []
        patch_student_args = []
        if patch_unmerging_spec is not None:
            patch_role, patch_block_name, patch_upsample_name, patch_mixup_name = patch_unmerging_spec
            if patch_role == "upsample":
                second_rank = loaded_ranks.get(patch_mixup_name)
                if second_rank is not None:
                    print(
                        f"[PatchUnmerging cache store] {name}: collecting {patch_block_name} "
                        "inputs for mixup reuse."
                    )
            elif patch_role == "mixup":
                patch_entry = patch_unmerging_cache.pop(name, None)
                if patch_entry is not None:
                    cache_entry = patch_entry
                    print(
                        f"[PatchUnmerging cache hit] {name}: reusing cached reshape/crop/norm "
                        f"output from {patch_upsample_name}; no forward-to-mixup rollout."
                    )

        data_length = len(calib_dataset)
        batch_size = 64
        sigma_HHp = torch.zeros((in_c, in_c), device=accum_device, dtype=accum_dtype)
        sigma_HH = torch.zeros((in_c, in_c), device=accum_device, dtype=accum_dtype)
        sigma_HpHp = torch.zeros((in_c, in_c), device=accum_device, dtype=accum_dtype)
        with perf.stage("compression/collect_layer_covariance", note=f"layer={name};rank={k};out={out_c};in={in_c}"):
            if cache_entry is not None:
                accumulated_count = _accumulate_cached_inputs(
                    cache_entry, device, in_c, sigma_HHp, sigma_HH, sigma_HpHp,
                    accum_device=accum_device, accum_dtype=accum_dtype
                )
                torch.cuda.empty_cache()
            else:
                accumulated_count = 0
                for idx in range(0, data_length, batch_size):
                    end_idx = min(idx + batch_size, data_length)
                    indices = list(range(idx, end_idx))
                    batch_ds = Subset(calib_dataset, indices)

                    if patch_role == "upsample":
                        teacher_args = patch_unmerging_args_catch(
                            trainer, model_teacher, patch_block_name, batch_ds, device
                        )
                        student_args = patch_unmerging_args_catch(
                            trainer, model_student, patch_block_name, batch_ds, device
                        )
                        _extend_cache(patch_teacher_args, teacher_args, offload_to_cpu=low_gpu)
                        _extend_cache(patch_student_args, student_args, offload_to_cpu=low_gpu)
                        H = torch.cat([args[0] for args in teacher_args], dim=0)
                        Hp = torch.cat([args[0] for args in student_args], dim=0)
                    elif attention_role == "query":
                        teacher_self_inputs = attention_self_input_catch(
                            trainer, model_teacher, attention_self_name, batch_ds, device
                        )
                        student_self_inputs = attention_self_input_catch(
                            trainer, model_student, attention_self_name, batch_ds, device
                        )
                        _extend_cache(attention_self_inputs_teacher, teacher_self_inputs, offload_to_cpu=low_gpu)
                        _extend_cache(attention_self_inputs_student, student_self_inputs, offload_to_cpu=low_gpu)
                        H = torch.cat([args[0] for args in teacher_self_inputs], dim=0)
                        Hp = torch.cat([args[0] for args in student_self_inputs], dim=0)
                    else:
                        H = mlp_aware_input_catch(trainer, model_teacher, name, batch_ds, device, input_catch)
                        Hp = mlp_aware_input_catch(trainer, model_student, name, batch_ds, device, input_catch)

                    if isinstance(H, list):
                        try: H = torch.cat(H, dim=0)
                        except: H = H[0]
                    H = to_2d_features(H, in_c).to(device, non_blocking=True)
                    H = to_2d_features(H, in_c)

                    if isinstance(Hp, list):
                        try: Hp = torch.cat(Hp, dim=0)
                        except: Hp = Hp[0]
                    Hp = to_2d_features(Hp, in_c).to(device, non_blocking=True)
                    Hp = to_2d_features(Hp, in_c)

                    if cache_second_name is not None:
                        _append_cache(cache_teacher_first_inputs, H, offload_to_cpu=low_gpu)
                        _append_cache(cache_student_first_inputs, Hp, offload_to_cpu=low_gpu)

                    H_acc = H.to(device=accum_device, dtype=accum_dtype, non_blocking=True)
                    Hp_acc = Hp.to(device=accum_device, dtype=accum_dtype, non_blocking=True)

                    sigma_HpHp.addmm_(Hp_acc.T, Hp_acc)
                    sigma_HH.addmm_(H_acc.T, H_acc)
                    sigma_HHp.addmm_(H_acc.T, Hp_acc)

                    accumulated_count += H.shape[0]
                    del H, Hp, H_acc, Hp_acc, batch_ds
                    torch.cuda.empty_cache()

        with perf.stage("compression/whiten_svd_replace_layer", note=f"layer={name};rank={k};samples={accumulated_count}"):
            W = lin.weight.data.to(device=device, dtype=accum_dtype)
            compute_dtype = W.dtype
            max_rank = min(out_c, in_c)
            sigma_comb = sigma_HpHp.mul(alpha).add(sigma_HH, alpha=(1 - alpha))

            if whitening_mode == "origin":
                left_factor, left_inverse = fisher_cholesky_whitener(
                    sigma_comb, accumulated_count
                )
                # The Fisher helper returns Ll with Ll.T @ Ll = sigma_comb.
                # Sequential compression needs Lr @ Lr.T = sigma_comb.
                Lr, Lr_invT = left_factor.T, left_inverse
                del left_factor, left_inverse
            else:
                # Equation (21) defines Sigma_c_cov as an expectation.  The
                # accumulated Gram is therefore converted to its empirical
                # mean before whitening.
                covariance_mean = sigma_comb / accumulated_count
                resolved_eps = resolve_whitening_eps(
                    covariance_mean,
                    eps_whiten,
                    eps_whiten_mode,
                    sample_count=accumulated_count,
                )
                Lr, Lr_invT = svd_whitener(
                    covariance_mean, eps=resolved_eps
                )
            Lr, Lr_invT = Lr.to(W.device, compute_dtype), Lr_invT.to(W.device, compute_dtype)
            del sigma_comb, sigma_HpHp

            if Ldict is not None and (name in Ldict):
                Ll, Ll_invT = Ldict[name]
                Ll, Ll_invT = Ll.to(W.device, compute_dtype), Ll_invT.to(W.device, compute_dtype)
            else:
                Ll = torch.eye(out_c, device=W.device, dtype=compute_dtype)
                Ll_invT = torch.eye(out_c, device=W.device, dtype=compute_dtype)


            right_cov = sigma_HH.mul(1 - alpha).add(sigma_HHp, alpha=alpha)
            # Equation (21) also defines Sigma_cov as an expectation.  It
            # must use the same mean convention as Sigma_c_cov above.
            right_cov.div_(accumulated_count)
            del sigma_HHp, sigma_HH
            C_comb_right = right_cov.to(device=W.device, dtype=compute_dtype) @ Lr_invT
            del right_cov, Lr
            Mstar = (Ll @ W) @ C_comb_right
            del C_comb_right, W, Ll
            U, S, VT = _stable_svd(Mstar, name)
            del Mstar
            # SVD
            U_k, S_k, VT_k = U[:, :k], S[:k], VT[:k, :]
            sqrtS = torch.sqrt(S_k)
            U_k = (U_k * sqrtS.unsqueeze(0)).float()
            V_k = (sqrtS.unsqueeze(1) * VT_k).float()
            del U, S, VT, S_k, VT_k, sqrtS
            # Re-whiten
            V_k = (V_k @ Lr_invT.T.to(V_k.dtype)).to(compute_dtype)
            U_k = (Ll_invT.to(U_k.dtype) @ U_k).to(compute_dtype)
            del Lr_invT, Ll_invT

            seq = nn.Sequential(
                nn.Linear(in_c, k, bias=False),
                nn.Linear(k, out_c, bias=(lin.bias is not None)),
            ).to(device)

            if lin.bias is not None:
                seq[1].bias.data.copy_(lin.bias.data)
            seq[0].weight.data.copy_(V_k)
            seq[1].weight.data.copy_(U_k)

            set_module_by_name(model_student, name, seq)
            del U_k, V_k, seq
        if patch_role == "upsample" and patch_mixup_name is not None:
            second_rank = loaded_ranks.get(patch_mixup_name)
            if second_rank is not None:
                patch_unmerging_cache[patch_mixup_name] = {
                    "type": "patch_unmerging_lazy",
                    "model_teacher": model_teacher,
                    "model_student": model_student,
                    "block_name": patch_block_name,
                    "teacher_args": patch_teacher_args,
                    "student_args": patch_student_args,
                    "num_batches": len(patch_teacher_args),
                }
                print(
                    f"[PatchUnmerging cache store] {name} -> {patch_mixup_name}: "
                    f"cached {len(patch_teacher_args)} block input batches for lazy mixup input compute."
                )
        if attention_role == "query":
            attention_cache[attention_name] = {
                "teacher_self_inputs": attention_self_inputs_teacher,
                "student_self_inputs": attention_self_inputs_student,
            }
            print(
                f"[Attention cache store] {name}: cached {len(attention_self_inputs_teacher)} "
                "teacher/student attention.self input batches for K/V/output."
            )
        elif attention_role == "value" and attention_name in attention_cache:
            attn_entry = attention_cache[attention_name]
            attn_entry["context_ready"] = True
            print(
                f"[Attention context store] {name} -> {attention_output_name}: "
                f"cached {len(attn_entry['teacher_self_inputs'])} attention.self input batches for lazy context."
            )
        elif attention_role == "output":
            attention_cache.pop(attention_name, None)
        if cache_second_name is not None:
            mlp_pair_cache[cache_second_name] = {
                "type": "mlp_lazy",
                "model_teacher": model_teacher,
                "model_student": model_student,
                "kind": kind,
                "block_name": block_name,
                "first_name": first_name,
                "teacher_first_inputs": cache_teacher_first_inputs,
                "student_first_inputs": cache_student_first_inputs,
                "num_batches": len(cache_teacher_first_inputs),
            }
            print(
                f"[MLP pair-cache store] {name} -> {cache_second_name}: "
                f"cached {len(cache_teacher_first_inputs)} first-layer input batches for lazy activation compute."
            )
        compressed_layers.add(name)
        layers_since_last_save += 1
        print(f"[{j:02d}] {name}: Compressed to rank {k}/{max_rank}")

        # --- 按百分比保存 checkpoint ---
        if save_path and layers_since_last_save >= checkpoint_interval:
            with perf.stage("compression/save_checkpoint", note=f"layers={len(compressed_layers)}/{total_layers}"):
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    'model': model_student.cpu(),
                    'compressed_layers': list(compressed_layers),
                }, ckpt_path)
                model_student = model_student.to(device)
                layers_since_last_save = 0
                pct = len(compressed_layers) / total_layers * 100
                print(f"  [Checkpoint] Saved at {pct:.1f}% ({len(compressed_layers)}/{total_layers} layers)")

    model_student = model_student.cpu()

    # --- 删除 checkpoint，压缩已完成 ---
    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"  [Checkpoint] Removed intermediate checkpoint {ckpt_path}")

    print("Accelerated compression complete.")
    return model_student
