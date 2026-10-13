"""SAFE-SVD main flow, organized like the scripts in physic_model_comp.

The caller loads the model and selects calibration data first. This entry
then runs input profiling, Fisher collection, rank selection, and compression.
Cache paths must belong to the same checkpoint and calibration configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from safe_svd.helper.collector.input_collector import collect_inputs
from safe_svd.helper.collector.fisher_collector import collect_fisher
from safe_svd.helper.collector.vicon_attention import convert_attn_to_linear
from safe_svd.rank import allocate_ranks
from safe_svd.compress import compress


def main(
    model,
    calibration_data,
    *,
    model_type,
    rank_ratio,
    device="cuda",
    params=None,
    trainer=None,
    rollout_fn=None,
    dataset_name=None,
    use_fisher="y",
    rank_selector="safe",
    alpha=0.7,
    compress_mode="comb",
    gradient_loss=None,
    balance_ratio=10.0,
    decay_alpha=1.0,
    order_control=0,
    calibration_seed=42,
    batches_per_dset=16,
    max_calib_batches=None,
    train_seed=None,
    fisher_batch_size=16,
    capture_batch_size=64,
    use_trace_balance=None,
    whitening_mode=None,
    eps_whiten=1e-6,
    eps_whiten_mode="absolute",
    input_accumulation_dtype=None,
    compute_dtype=torch.float64,
    offload_cache=False,
    fisher_loss_fn=None,
    profile_cache=None,
    fisher_cache=None,
    rank_cache=None,
    save_path=None,
    overwrite_cache=False,
    verbose=True,
):
    """Return the compressed nn.Module directly.

    MPP: calibration_data is train_dataset; supply params.
    VICON: calibration_data is calib_loaders; supply trainer.
    Poseidon: calibration_data is calib_dataset or its Fisher DataLoader;
    supply trainer, dataset_name, and optionally rollout_fn.

    Like the original scripts, explicit cache paths are loaded when present
    and saved after their stage. overwrite_cache=True forces recomputation.
    Final saving follows the originals: MPP/Poseidon save a complete model;
    VICON saves its state_dict plus rank_dict.json for the low-rank topology.
    """
    # ===================================================================
    # 0. Configuration
    # ===================================================================
    if model_type not in {"mpp", "vicon", "poseidon"}:
        raise ValueError("model_type must be mpp, vicon, or poseidon")
    if not 0 < rank_ratio <= 1 or not 0 <= alpha <= 1:
        raise ValueError("Invalid rank_ratio or alpha")
    if rank_selector not in {"safe", "legacy"}:
        raise ValueError("rank_selector must be safe or legacy")
    if compress_mode not in {"comb", "clean_input"}:
        raise ValueError("compress_mode must be comb or clean_input")
    if use_fisher not in {True, False, "y", "n"}:
        raise ValueError("use_fisher must be y/n or bool")
    use_fisher = use_fisher == "y" if isinstance(use_fisher, str) else use_fisher
    if use_trace_balance is None:
        use_trace_balance = model_type != "poseidon"
    if use_trace_balance not in {True, False, "y", "n"}:
        raise ValueError("use_trace_balance must be y/n or bool")
    trace_balance = use_trace_balance == "y" if isinstance(use_trace_balance, str) else use_trace_balance
    whitening_mode = whitening_mode or ("origin" if model_type == "poseidon" else "new")
    if whitening_mode not in {"origin", "new"}:
        raise ValueError("whitening_mode must be origin or new")
    whitening = "fisher_cholesky" if whitening_mode == "origin" else "spectral"
    aggregation = "category_mean" if model_type == "vicon" else "sample_mean"
    gradient_loss = gradient_loss or ("mse" if model_type == "poseidon" else "l1")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; set device='cpu'")

    profile_cache = Path(profile_cache).expanduser() if profile_cache is not None else None
    fisher_cache = Path(fisher_cache).expanduser() if fisher_cache is not None else None
    rank_cache = Path(rank_cache).expanduser() if rank_cache is not None else None
    save_path = Path(save_path).expanduser() if save_path is not None else None
    final_path = None
    if save_path is not None:
        filename = "compressed_model.pth" if model_type == "vicon" else "compressed_model.pt"
        final_path = save_path / filename
        if final_path.exists() and not overwrite_cache:
            raise FileExistsError(f"Model already exists: {final_path}; choose a new save_path or overwrite_cache=True")

    # ===================================================================
    # 1. Prepare the loaded model and calibration data
    # ===================================================================
    if model_type == "mpp":
        if params is None:
            raise ValueError("MPP requires params")
        route_options = {
            "params": params,
            "batches_per_dset": batches_per_dset,
            "calibration_seed": calibration_seed,
        }
    elif model_type == "vicon":
        if trainer is None:
            raise ValueError("VICON requires trainer")
        convert_attn_to_linear(model, verbose=verbose)
        route_options = {
            "trainer": trainer,
            "max_calib_batches": max_calib_batches,
            "train_seed": train_seed,
        }
    else:
        if trainer is None:
            raise ValueError("Poseidon requires trainer")
        fisher_loader = None
        if isinstance(calibration_data, DataLoader):
            fisher_loader = calibration_data
            calibration_data = calibration_data.dataset
        route_options = {
            "trainer": trainer,
            "rollout_fn": rollout_fn,
            "dataset_name": dataset_name or "poseidon",
            "fisher_loader": fisher_loader,
            "fisher_batch_size": fisher_batch_size,
            "capture_batch_size": capture_batch_size,
            "order_control": order_control,
            "offload_cache": offload_cache,
        }
    route_options["gradient_loss"] = gradient_loss
    route_options["balance_ratio"] = balance_ratio
    route_options["decay_alpha"] = decay_alpha
    model = model.to(dev).eval()

    # ===================================================================
    # 2. Input Profiling: Rdict and input_sample_counts
    # ===================================================================
    if verbose:
        print("==== Step 1: Input Profiling ====", flush=True)
    if profile_cache is not None and profile_cache.is_file() and not overwrite_cache:
        cached_profile = torch.load(profile_cache, map_location="cpu", weights_only=True)
        if "profile_mat" in cached_profile:
            Rdict = cached_profile["profile_mat"]
            input_sample_counts = cached_profile["sample_counts"]
        else:
            Rdict = cached_profile  # Original MPP/VICON factors-only cache.
            input_sample_counts = {}
        if verbose:
            print(f"Loaded input cache: {profile_cache}")
    else:
        Rdict, input_sample_counts = collect_inputs(
            model,
            calibration_data,
            model_type=model_type,
            device=dev,
            route_options=route_options,
            whitening=whitening,
            eps=eps_whiten,
            eps_mode=eps_whiten_mode,
            aggregation=aggregation,
            trace_balance=trace_balance,
            accumulation_dtype=input_accumulation_dtype,
        )
        if profile_cache is not None:
            profile_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"profile_mat": Rdict, "sample_counts": input_sample_counts}, profile_cache)

    # ===================================================================
    # 3. Fisher Collection: Ldict
    # ===================================================================
    if verbose:
        print("==== Step 2: Fisher Collection ====", flush=True)
    Ldict = None
    if use_fisher:
        if fisher_cache is not None and fisher_cache.is_file() and not overwrite_cache:
            Ldict = torch.load(fisher_cache, map_location="cpu", weights_only=True)
            if verbose:
                print(f"Loaded Fisher cache: {fisher_cache}")
        else:
            Ldict = collect_fisher(
                model,
                calibration_data,
                model_type=model_type,
                device=dev,
                route_options=route_options,
                aggregation=aggregation,
                trace_balance=trace_balance,
                loss_fn=fisher_loss_fn,
            )
            if fisher_cache is not None:
                fisher_cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(Ldict, fisher_cache)

    # ===================================================================
    # 4. Rank Selection: rank_dict
    # ===================================================================
    if verbose:
        print("==== Step 3: Rank Selection ====", flush=True)
    if rank_cache is not None and rank_cache.is_file() and not overwrite_cache:
        rank_dict = json.loads(rank_cache.read_text())
        if verbose:
            print(f"Loaded rank cache: {rank_cache}")
    else:
        rank_dict = allocate_ranks(
            model,
            left_whiteners=Ldict,
            input_whiteners=Rdict,
            rank_ratio=rank_ratio,
            device=dev,
            selector=rank_selector,
        )
        if rank_cache is not None:
            rank_cache.parent.mkdir(parents=True, exist_ok=True)
            rank_cache.write_text(json.dumps(rank_dict, indent=2) + "\n")

    # ===================================================================
    # 5. Compression: teacher/student layer-by-layer replacement
    # ===================================================================
    if verbose:
        print("==== Step 4: Compression ====", flush=True)
    compressed_model = compress(
        model,
        calibration_data,
        ranks=rank_dict,
        model_type=model_type,
        device=dev,
        route_options=route_options,
        left_metrics=Ldict,
        input_metrics=Rdict,
        alpha=alpha,
        mode="clean" if compress_mode == "clean_input" else "combined",
        whitening=whitening,
        eps=eps_whiten,
        eps_mode=eps_whiten_mode,
        aggregation=aggregation,
        trace_balance=trace_balance,
        compute_dtype=compute_dtype,
    )

    # ===================================================================
    # 6. Save: use the original per-model checkpoint format
    # ===================================================================
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        if model_type == "vicon":
            torch.save(compressed_model.state_dict(), final_path)
        else:
            torch.save(compressed_model, final_path)
        (save_path / "rank_dict.json").write_text(json.dumps(rank_dict, indent=2) + "\n")
        if verbose:
            print(f"Success! Saved to {final_path}")
    return compressed_model


run_safe_svd = main
