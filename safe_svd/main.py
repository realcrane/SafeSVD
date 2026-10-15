"""Standalone SAFE-SVD command-line entry point.

Run directly with --model mpp/vicon/poseidon to load model and calibration
data, then run input profiling, Fisher collection, rank selection and compression.
Cache paths must belong to the same checkpoint and calibration configuration.
All CLI and Python API parameters are documented in safe_svd/PARAMETERS.md.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow both python -m safe_svd.main and python safe_svd/main.py.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_svd.helper.perf_logger import make_perf_logger
from safe_svd.helper.collector.input_collector import collect_inputs
from safe_svd.helper.collector.fisher_collector import collect_fisher
from safe_svd.helper.collector.vicon_attention import convert_attn_to_linear
from safe_svd.rank import allocate_ranks
from safe_svd.compress import compress

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    fisher_batch_size=8,
    capture_batch_size=64,
    use_trace_balance=None,
    whitening_mode=None,
    eps_whiten=1e-6,
    eps_whiten_mode=None,
    input_accumulation_dtype=None,
    compute_dtype=torch.float64,
    offload_cache=True,
    fisher_loss_fn=None,
    profile_cache=None,
    fisher_cache=None,
    rank_cache=None,
    save_path=None,
    overwrite_cache=False,
    checkpoint_percent=0.1,
    verbose=True,
    pipeline_stage="all",
    perf_csv_path=None,
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
    With save_path, checkpoint every checkpoint_percent of Linear layers.
    Resume automatically from compressing_model.pt unless overwrite_cache=True.
    Completed profiling/Fisher/rank caches are reused.
    """
    # ===================================================================
    # 0. Configuration
    # ===================================================================
    if pipeline_stage not in {"all", "rank_selection", "compression"}:
        raise ValueError("pipeline_stage must be all, rank_selection, or compression")
    if model_type not in {"mpp", "vicon", "poseidon"}:
        raise ValueError("model_type must be mpp, vicon, or poseidon")
    if not 0 < rank_ratio <= 1 or not 0 <= alpha <= 1:
        raise ValueError("Invalid rank_ratio or alpha")
    if not 0 < checkpoint_percent <= 1:
        raise ValueError("checkpoint_percent must be in (0,1]")
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
    whitening_mode = whitening_mode or "new"
    if whitening_mode not in {"origin", "new"}:
        raise ValueError("whitening_mode must be origin or new")
    whitening = "fisher_cholesky" if whitening_mode == "origin" else "spectral"
    eps_whiten_mode = eps_whiten_mode or (
        "sum_equivalent" if model_type == "poseidon" and whitening_mode == "new" else "absolute"
    )
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
        if pipeline_stage != "rank_selection" and final_path.exists() and not overwrite_cache:
            raise FileExistsError(f"Model already exists: {final_path}; choose a new save_path or overwrite_cache=True")

    if pipeline_stage == "compression":
        if profile_cache is None or not profile_cache.is_file():
            raise FileNotFoundError("compression stage requires an existing profile_cache")
        if use_fisher and (fisher_cache is None or not fisher_cache.is_file()):
            raise FileNotFoundError("compression stage requires an existing fisher_cache")
        if rank_cache is None or not rank_cache.is_file():
            raise FileNotFoundError("compression stage requires an existing rank_cache")
        if overwrite_cache:
            raise ValueError("compression-only cannot recompute caches; use pipeline_stage='all'")
    perf = make_perf_logger(perf_csv_path, device=dev, method="safe_svd", model_name=model_type,
                           rank_ratio=rank_ratio, alpha=alpha, dataset=dataset_name)

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
    with perf.stage("input_collection"):
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
            Rdict = None
        needs_counts = model_type == "poseidon" and whitening_mode == "new" and (
            rank_cache is None or not rank_cache.is_file() or overwrite_cache
        )
        if Rdict is not None and needs_counts and not input_sample_counts:
            if pipeline_stage == "compression":
                raise ValueError("New Poseidon rank selection needs sample counts")
            Rdict = None
        if Rdict is None:
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
    with perf.stage("fisher_collection"):
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
                    progress=verbose,
                )
                if fisher_cache is not None:
                    fisher_cache.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(Ldict, fisher_cache)

    # ===================================================================
    # 4. Rank Selection: rank_dict
    # ===================================================================
    if verbose:
        print("==== Step 3: Rank Selection ====", flush=True)
    with perf.stage("rank_selection"):
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
                energy_scales=input_sample_counts if model_type == "poseidon" and whitening_mode == "new" else None,
            )
            if rank_cache is not None:
                rank_cache.parent.mkdir(parents=True, exist_ok=True)
                rank_cache.write_text(json.dumps(rank_dict, indent=2) + "\n")

    if pipeline_stage == "rank_selection":
        if verbose:
            print(f"Rank selection completed: {rank_cache}")
        return None

    # ===================================================================
    # 5. Compression: teacher/student layer-by-layer replacement
    # ===================================================================
    if verbose:
        print("==== Step 4: Compression ====", flush=True)
    if dev.type == "cuda":
        # Match the existing runtime's fresh-stage boundary before copying
        # the teacher: clear unused buffers without changing model precision.
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
    checkpoint_path = save_path / "compressing_model.pt" if save_path is not None else None
    initial_student, completed_layers = None, set()
    total_layers = sum(isinstance(layer, nn.Linear) for layer in model.modules())
    checkpoint_every = max(int(total_layers * checkpoint_percent), 1)
    compression_config = {
        "model_type": model_type, "dataset_name": dataset_name,
        "alpha": alpha, "compress_mode": compress_mode,
        "whitening": whitening, "eps": eps_whiten, "eps_mode": eps_whiten_mode,
        "aggregation": aggregation, "trace_balance": trace_balance,
        "compute_dtype": str(compute_dtype), "use_fisher": use_fisher,
        "gradient_loss": gradient_loss, "balance_ratio": balance_ratio,
        "decay_alpha": decay_alpha, "order_control": order_control,
        "calibration_seed": calibration_seed, "batches_per_dset": batches_per_dset,
        "max_calib_batches": max_calib_batches, "train_seed": train_seed,
        "calibration_size": len(calibration_data) if hasattr(calibration_data, "__len__") else None,
    }
    if checkpoint_path is not None and checkpoint_path.is_file() and not overwrite_cache:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "compressed_layers" not in checkpoint:
            raise ValueError(f"Invalid compression checkpoint: {checkpoint_path}")
        if "rank_dict" in checkpoint and checkpoint["rank_dict"] != rank_dict:
            raise ValueError("Checkpoint ranks differ from current ranks; use a new save_path or --overwrite_cache (Python: overwrite_cache=True)")
        if "compression_config" in checkpoint and checkpoint["compression_config"] != compression_config:
            changed = [key for key, value in compression_config.items()
                       if checkpoint["compression_config"].get(key) != value]
            raise ValueError(f"Checkpoint compression settings differ: {changed}; use a new save_path or --overwrite_cache (Python: overwrite_cache=True)")
        completed_layers = set(checkpoint["compressed_layers"])
        if "model" in checkpoint:
            initial_student = checkpoint["model"]
            if not isinstance(initial_student, nn.Module):
                raise ValueError("Checkpoint model must be an nn.Module")
        else:
            # VICON stores weights and topology, as in its final output format.
            initial_student = copy.deepcopy(model).cpu()
            for name, original in model.named_modules():
                if name not in completed_layers or not isinstance(original, nn.Linear):
                    continue
                rank = rank_dict.get(name)
                if rank is None or rank == 0 or rank * (original.in_features + original.out_features) >= original.weight.numel():
                    continue
                replacement = nn.Sequential(
                    nn.Linear(original.in_features, rank, bias=False),
                    nn.Linear(rank, original.out_features, bias=original.bias is not None),
                ).to(device="cpu", dtype=original.weight.dtype)
                parent, _, attribute = name.rpartition(".")
                setattr(initial_student.get_submodule(parent) if parent else initial_student, attribute, replacement)
            initial_student.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        if verbose:
            print(f"==== Resuming from {checkpoint_path}: {len(completed_layers)}/{total_layers} layers completed ====", flush=True)

    def save_compression_checkpoint(student, done):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        student_device = next(student.parameters()).device
        try:
            student.cpu()
            checkpoint = {
                "compressed_layers": sorted(done), "rank_dict": rank_dict,
                "compression_config": compression_config,
            }
            if model_type == "vicon":
                checkpoint["model_state_dict"] = student.state_dict()
            else:
                checkpoint["model"] = student
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
        finally:
            student.to(student_device)
            temporary.unlink(missing_ok=True)
        if verbose:
            print(f"[Checkpoint] Saved {len(done)}/{total_layers} layers to {checkpoint_path}", flush=True)

    if checkpoint_path is not None and verbose:
        print(f"Checkpoint every {checkpoint_every} layers ({checkpoint_percent:.0%}): {checkpoint_path}", flush=True)
    with perf.stage("compression"):
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
            checkpoint_fn=save_compression_checkpoint if checkpoint_path is not None else None,
            checkpoint_every=checkpoint_every,
            initial_student=initial_student,
            completed_layers=completed_layers,
            progress=verbose,
        )

    # ===================================================================
    # 6. Save: use the original per-model checkpoint format
    # ===================================================================
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        # The final filename marks success, so never leave a partial final file.
        temporary = final_path.with_name(final_path.name + ".tmp")
        rank_output = save_path / "rank_dict.json"
        rank_temporary = rank_output.with_name(rank_output.name + ".tmp")
        try:
            torch.save(compressed_model.state_dict() if model_type == "vicon" else compressed_model, temporary)
            rank_temporary.write_text(json.dumps(rank_dict, indent=2) + "\n")
            rank_temporary.replace(rank_output)
            temporary.replace(final_path)
        finally:
            temporary.unlink(missing_ok=True)
            rank_temporary.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
        if verbose:
            print(f"Success! Saved to {final_path}")
    return compressed_model


run_safe_svd = main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified SAFE-SVD: load -> input -> Fisher -> rank -> compress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model defaults: MPP ratio=0.4, samples=16, seed=42, selector=legacy;\n"
               "VICON ratio=0.8, samples=2048, seed=2026, selector=legacy;\n"
               "Poseidon ratio=0.2, trajectories=64, seed=default, selector=safe.\n"
               "Full parameter definitions and memory guidance: safe_svd/PARAMETERS.md",
    )
    parser.add_argument("--model", "--model_type", dest="model_type", choices=("mpp", "vicon", "poseidon"), required=True,
                        help="Select the model/data loading and collection route")
    parser.add_argument("--DEV", "--device", dest="DEV", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument("--ckpt_path", "--model_path", dest="ckpt_path", type=Path,
                        help="Checkpoint file for MPP/VICON, config+weights directory for Poseidon")
    parser.add_argument("--data_root", "--data_path", dest="data_root", type=Path, help="Dataset root directory")
    parser.add_argument("--rank_ratio", type=float, help="Retained Linear weight budget fraction in (0,1], not per-layer rank")
    parser.add_argument("--rank_selector", choices=("legacy", "safe"), help="Original greedy or strict-budget greedy allocation")
    parser.add_argument("--alpha", type=float, default=0.7, help="Student-input correction weight in comb mode, [0,1] (default: 0.7)")
    parser.add_argument("--use_fisher", choices=("y", "n"), default="y", help="Use Fisher left metrics in rank/compression (default: y)")
    parser.add_argument("--fisher_loss", "--gradient_loss", dest="gradient_loss", choices=("l1", "mse", "smooth_l1"), default="l1",
                        help="Physical/Sobolev error function; base model loss is unchanged (default: l1)")
    parser.add_argument("--balance_ratio", type=float, default=10.0, help="Physical loss strength relative to base loss (default: 10)")
    parser.add_argument("--decay_alpha", type=float, default=1.0, help="MPP/Poseidon order-k loss multiplier decay_alpha**(k-1) (default: 1)")
    parser.add_argument("--use_trace_balance", "--use_trace_align", dest="use_trace_balance", choices=("y", "n"),
                        help="Balance covariance traces across categories (MPP/VICON: y; Poseidon: n)")
    parser.add_argument("--compress_mode", choices=("comb", "clean_input"), default="comb",
                        help="Sequential teacher/student capture or original-input-only SVD (default: comb)")
    parser.add_argument("--whitening_mode", choices=("origin", "new"),
                        help="Cholesky-relative-jitter or spectral whitening (default: new)")
    parser.add_argument("--whitening_eps", "--eps_whiten", dest="eps_whiten", type=float, default=1e-6,
                        help="Positive regularizer for new spectral whitening (default: 1e-6)")
    parser.add_argument("--whitening_eps_mode", "--eps_whiten_mode", dest="eps_whiten_mode",
                        choices=("absolute", "relative", "sum_equivalent"),
                        help="Regularizer scale (Poseidon new: sum_equivalent; others: absolute)")
    parser.add_argument("--calibration_seed", default=None, metavar="default|SEED",
                        help="Sampling seed; Poseidon default selects fixed prefix, integer selects random trajectories")
    parser.add_argument("--batch_size", type=int, help="MPP YAML batch override; VICON default 128; Poseidon inference default 256")
    parser.add_argument("--fisher_batch_size", type=int, default=8, help="Poseidon backward batch size (default: 16)")
    parser.add_argument("--capture_batch_size", type=int, default=8, help="Poseidon compression capture slice size in examples, not trajectories (default: 64)")
    parser.add_argument("--pipeline_stage", choices=("all", "rank_selection", "compression"), default="all",
                        help="All stages, stop after ranks, or compress with existing caches (default: all)")
    parser.add_argument("--profile_cache", type=Path, help="Input whitening factors and sample-count cache (.pt)")
    parser.add_argument("--fisher_cache", type=Path, help="Fisher left-factor cache (.pt)")
    parser.add_argument("--rank_cache", "--rank_path", dest="rank_cache", type=Path, help="Rank allocation cache (.json)")
    parser.add_argument("--cache_dir", type=Path, help="Root for generated input/Fisher caches")
    parser.add_argument("--rank_root", type=Path, help="Root for generated rank caches")
    parser.add_argument("--save_path", type=Path, help="Output directory for compressed model and rank_dict.json")
    parser.add_argument("--overwrite_cache", action="store_true", help="Recompute caches, ignore compression checkpoint and allow replacing final output")
    parser.add_argument("--checkpoint_percent", type=float, default=0.1,
                        help="Save compression checkpoint every fraction of Linear layers, in (0,1] (default: 0.1)")
    parser.add_argument("--dry_run", action="store_true", help="Print paths/configuration without importing model runtimes or loading assets")

    # MPP arguments, retaining the names in MySVD/mysvd.py.
    parser.add_argument("--yaml_config", type=Path, help="MPP model/data YAML file")
    parser.add_argument("--config", default="basic_config", help="MPP YAML section (default: basic_config)")
    parser.add_argument("--calibration_num", type=int, help="Samples per MPP file or VICON category (MPP: 16; VICON: 2048)")

    # VICON arguments; compose its original Hydra configuration explicitly.
    parser.add_argument("--config_dir", type=Path, help="VICON Hydra configuration directory")
    parser.add_argument("--config_name", default="default", help="VICON Hydra config name (default: default)")
    parser.add_argument("--config_override", action="append", default=[], metavar="KEY=VALUE", help="Repeatable VICON Hydra override")
    parser.add_argument("--random_calibration", choices=("y", "n"), default="n", help="VICON seeded random calibration selection (default: n)")
    parser.add_argument("--rank_type", default="MySVD_Rank", choices=("MySVD_Rank",), help="VICON cache label; algorithm is chosen by rank_selector")
    parser.add_argument("--dataset_workers", type=int, help="VICON calibration DataLoader workers (default: cfg.dataset_workers)")

    # Poseidon arguments, retaining SAFE_SVD/safe_svd.py's trajectory semantics.
    parser.add_argument("--model_name", default="Poseidon-L", help="Poseidon cache label; architecture comes from checkpoint (default: Poseidon-L)")
    parser.add_argument("--dataset", default="fluids.incompressible.PiecewiseConstants", help="Poseidon dataset registry name (default: fluids.incompressible.PiecewiseConstants)")
    parser.add_argument("--traj_num", type=int, default=64, help="Poseidon complete calibration trajectories (default: 64)")
    parser.add_argument("--use_phyloss", default="no_phyloss", help="Compatibility argument only; does not change checkpoint or Fisher loss")
    parser.add_argument("--final_time", type=int, default=20, help="Poseidon max_num_time_steps for non-elliptic data (default: 20)")
    parser.add_argument("--order_control", type=int, default=2, help="Poseidon maximum spatial derivative order; 0 still includes physical value error (default: 2)")
    parser.add_argument("--compression_backend", choices=("GPU", "low-gpu"), default="GPU",
                        help="Compression precision: GPU=float64; low-gpu=float32. Poseidon caches are CPU in both modes")
    parser.add_argument("--gpu_profile", action="store_true", help="Write per-stage timing/memory to save_path/gpu_profile.csv")
    args = parser.parse_args(argv)

    # Model-specific defaults match the original model scripts.
    if args.rank_ratio is None:
        args.rank_ratio = {"mpp": 0.4, "vicon": 0.8, "poseidon": 0.2}[args.model_type]
    if args.rank_selector is None:
        args.rank_selector = "safe" if args.model_type == "poseidon" else "legacy"
    if args.calibration_num is None:
        args.calibration_num = 2048 if args.model_type == "vicon" else 16
    if args.whitening_mode is None:
        args.whitening_mode = "new"
    if args.eps_whiten_mode is None:
        args.eps_whiten_mode = (
            "sum_equivalent" if args.model_type == "poseidon" and args.whitening_mode == "new" else "absolute"
        )
    if args.use_trace_balance is None:
        args.use_trace_balance = "n" if args.model_type == "poseidon" else "y"
    if args.calibration_seed is None:
        args.calibration_seed = {"mpp": 42, "vicon": 2026, "poseidon": None}[args.model_type]
    elif args.calibration_seed == "default":
        if args.model_type != "poseidon":
            parser.error("calibration_seed=default is only supported by Poseidon")
        args.calibration_seed = None
    else:
        try:
            args.calibration_seed = int(args.calibration_seed)
        except ValueError:
            parser.error("calibration_seed must be default or a non-negative integer")
    if args.calibration_seed is not None and args.calibration_seed < 0:
        parser.error("calibration_seed must be non-negative")
    if not 0 < args.rank_ratio <= 1 or not 0 <= args.alpha <= 1:
        parser.error("rank_ratio must be in (0,1], alpha in [0,1]")
    if not 0 < args.checkpoint_percent <= 1:
        parser.error("checkpoint_percent must be in (0,1]")
    if args.calibration_num <= 0 or args.traj_num <= 0 or args.final_time <= 0:
        parser.error("Calibration sizes and final_time must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("batch_size must be positive")
    if args.fisher_batch_size <= 0 or args.capture_batch_size <= 0:
        parser.error("fisher_batch_size and capture_batch_size must be positive")
    if args.dataset_workers is not None and args.dataset_workers < 0:
        parser.error("dataset_workers must be non-negative")
    if args.model_type == "mpp" and args.gradient_loss == "smooth_l1":
        parser.error("MPP original Fisher supports l1 or mse")
    if args.model_type == "vicon" and args.calibration_num < (args.batch_size or 128):
        parser.error("VICON calibration_num must be at least batch_size")
    if args.eps_whiten <= 0 or args.order_control < 0:
        parser.error("whitening_eps must be positive and order_control non-negative")
    return args


def resolve_cli_paths(args):
    # Model code comes only from official checkouts inside this repository.
    # User-supplied file paths are relative to the invocation directory.
    if args.model_type == "mpp":
        args.runtime_root = PROJECT_ROOT / "mpp/multiple_physics_pretraining"
        prepared_config = PROJECT_ROOT / "data/official/configs/mpp_avit_L.yaml"
        default_config = prepared_config if prepared_config.is_file() else args.runtime_root / "config/mpp_avit_L_config.yaml"
        default_checkpoint = PROJECT_ROOT / "models/MPP/MPP_AViT_L"
        default_data = PROJECT_ROOT / "data/official/mpp"
        calibration_tag = f"cali{args.calibration_num}_seed{args.calibration_seed}"
    elif args.model_type == "vicon":
        args.runtime_root = PROJECT_ROOT / "vicon/VICON"
        if not (args.runtime_root / "src").is_dir():
            args.runtime_root = PROJECT_ROOT / "VICON"
        default_config = args.runtime_root / "configs"
        default_checkpoint = PROJECT_ROOT / "models/VICON/model.pth"
        default_data = PROJECT_ROOT / "data/VICON"
        calibration_tag = f"cali{args.calibration_num}"
        if args.random_calibration == "y":
            calibration_tag += f"_random_seed{args.calibration_seed}"
    else:
        args.runtime_root = PROJECT_ROOT / "poseidon"
        default_config = None
        default_checkpoint = PROJECT_ROOT / "models/poseidon" / args.dataset
        default_data = PROJECT_ROOT / "poseidon/camlab-ethz/down_streams"
        calibration_tag = f"traj{args.traj_num}"
        if args.calibration_seed is not None:
            calibration_tag += f"_random_seed{args.calibration_seed}"
        calibration_tag += f"_ftime{args.final_time}"
    args.ckpt_path = args.ckpt_path.expanduser().resolve() if args.ckpt_path is not None else default_checkpoint
    args.data_root = args.data_root.expanduser().resolve() if args.data_root is not None else default_data
    if args.model_type == "mpp":
        args.yaml_config = args.yaml_config.expanduser().resolve() if args.yaml_config is not None else default_config
    elif args.model_type == "vicon":
        args.config_dir = args.config_dir.expanduser().resolve() if args.config_dir is not None else default_config
    trace_tag = "trace" if args.use_trace_balance == "y" else "notrace"
    whitening_tag = f"{args.whitening_mode}_eps{args.eps_whiten_mode}{args.eps_whiten}"
    if args.model_type == "poseidon" and args.whitening_mode == "new":
        whitening_tag += "_rankscale"
    dataset_tag = args.dataset if args.model_type == "poseidon" else args.model_type
    input_tag = f"{calibration_tag}_{trace_tag}_{whitening_tag}"
    if args.batch_size is not None:
        input_tag += f"_bs{args.batch_size}"
    fisher_tag = f"{input_tag}_{args.gradient_loss}_br{args.balance_ratio}_decay{args.decay_alpha}"
    if args.model_type == "poseidon":
        fisher_tag += f"_order{args.order_control}_derivonly"
        if args.use_fisher == "y":
            fisher_tag += f"_fbs{args.fisher_batch_size}"
    checkpoint_tag = args.model_name if args.model_type == "poseidon" else args.ckpt_path.stem
    args.cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else PROJECT_ROOT / "artifacts/cache" / dataset_tag / checkpoint_tag
    args.profile_cache = args.profile_cache.expanduser().resolve() if args.profile_cache else args.cache_dir / "input" / input_tag / "profiling_mat.pt"
    args.fisher_cache = args.fisher_cache.expanduser().resolve() if args.fisher_cache else args.cache_dir / "fisher" / fisher_tag / "Ldict.pt"
    args.rank_root = args.rank_root.expanduser().resolve() if args.rank_root else PROJECT_ROOT / "artifacts/ranks" / dataset_tag / checkpoint_tag
    rank_tag = f"{fisher_tag}_{args.rank_selector}_{args.use_fisher}"
    if args.model_type == "vicon":
        rank_tag += f"_{args.rank_type}"
    args.rank_cache = args.rank_cache.expanduser().resolve() if args.rank_cache else args.rank_root / rank_tag / f"rank_{args.rank_ratio}.json"
    args.save_path = args.save_path.expanduser().resolve() if args.save_path else (
        PROJECT_ROOT / "artifacts/compressed" / dataset_tag / checkpoint_tag / str(args.rank_ratio) /
        f"{rank_tag}_alpha{args.alpha}_{args.compress_mode}_{args.compression_backend}"
    )
    return args


def load_model_and_data(args):
    """Load a repository-local official model and its calibration data."""
    package_root = args.runtime_root.resolve()
    if not package_root.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"Model source must be inside this repository: {package_root}")
    device = torch.device(args.DEV)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --DEV cpu")
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    params, trainer, rollout_fn = None, None, None
    args.train_seed = None
    args.batches_per_dset = 16
    args.max_calib_batches = None

    # ===================================================================
    # MPP: YAML -> AViT checkpoint -> training datasets
    # ===================================================================
    if args.model_type == "mpp":
        if not (package_root / "models/avit.py").is_file():
            raise FileNotFoundError(f"Official MPP source is missing: {package_root}/models/avit.py. "
                                    "Install the full PolymathicAI/multiple_physics_pretraining checkout under mpp/.")
        sys.path.insert(0, str(package_root.parent))
        from multiple_physics_pretraining.utils.YParams import YParams
        from multiple_physics_pretraining.models.avit import build_avit
        from multiple_physics_pretraining.data_utils.datasets import get_data_loader
        if not args.ckpt_path.is_file():
            raise ValueError("MPP --ckpt_path must be a checkpoint file")
        params = YParams(str(args.yaml_config), args.config)
        params.use_ddp = False
        params.log_to_wandb = False
        if args.batch_size is not None:
            params["batch_size"] = args.batch_size
        data_root = args.data_root
        for field in ("train_data_paths", "valid_data_paths"):
            rebased = []
            for path_spec in getattr(params, field):
                path_spec = list(path_spec)
                path = Path(path_spec[0]).expanduser()
                if not path.is_absolute():
                    path = data_root / path
                path_spec[0] = str(path)
                rebased.append(path_spec)
            setattr(params, field, rebased)
        print(f"==== Loading MPP model: {args.ckpt_path} ====", flush=True)
        model = build_avit(params).to(device)
        checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)))
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        print("==== Loading MPP train_dataset ====", flush=True)
        _, calibration_data, _ = get_data_loader(
            params, params.train_data_paths, distributed=False, split="train", rank=0,
            train_offset=params.embedding_offset,
        )
        # This division is the actual sampling rule in the original MPP main.
        args.batches_per_dset = args.calibration_num / params.batch_size
        print(f"MPP calibration batch limit per file: {args.batches_per_dset}")

    # ===================================================================
    # VICON: Hydra config -> ICON checkpoint/trainer -> calib_loaders
    # ===================================================================
    elif args.model_type == "vicon":
        if not (package_root / "src").is_dir():
            raise FileNotFoundError(f"Official VICON source is missing: {package_root}/src. "
                                    "Install the full Eydcao/VICON checkout under vicon/VICON/.")
        # This checkout also uses absolute imports such as model_utils/utils.
        sys.path.insert(0, str(package_root.parent))
        sys.path.insert(0, str(package_root / "src"))
        from hydra import compose, initialize_config_dir
        from omegaconf import open_dict
        import VICON.src.models as models
        import VICON.src.trainer as trainer_module
        from VICON.src.dataset import all_datasets
        from safe_svd.helper.collector.vicon_calibration_dataset import build_calibration_loaders
        with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base=None):
            cfg = compose(config_name=args.config_name, overrides=args.config_override)
        with open_dict(cfg):
            if args.dataset_workers is not None:
                cfg.dataset_workers = args.dataset_workers
            dataset_configs = cfg.datasets.types if hasattr(cfg.datasets, "types") else cfg.datasets
            for name, dataset_config in dataset_configs.items():
                for field in ("folder", "path", "data_path"):
                    if not hasattr(dataset_config, field):
                        continue
                    path = Path(getattr(dataset_config, field)).expanduser()
                    explicit = any(override.lstrip("+").startswith((f"datasets.types.{name}.{field}=", f"datasets.{name}.{field}="))
                                   for override in args.config_override)
                    if args.data_root is not None and not explicit:
                        relative = Path(path.name) if path.is_absolute() or ".." in path.parts else path
                        path = args.data_root / relative
                    elif not path.is_absolute():
                        path = Path.cwd() / path
                    setattr(dataset_config, field, str(path.resolve()))
        constructor = models.ICON_CROPPED if cfg.model.type == "crop" else models.ICON_UNCROPPED
        model = constructor(cfg.model)
        # Original Trainer has a module-global default device; honor --DEV.
        trainer_module.device = device
        trainer = trainer_module.Trainer(model, cfg.model, cfg.opt, cfg.loss, trainable_mode=cfg.trainable_mode)
        trainer.device = device
        checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        trainer.model.load_state_dict({key.removeprefix("module."): value for key, value in state_dict.items()})
        model = trainer.model.to(device).eval()
        print("==== Loading VICON calibration datasets ====", flush=True)
        dataset_seed = args.calibration_seed if args.random_calibration == "y" else cfg.train_seed
        if "rollout_cfg" in inspect.signature(all_datasets).parameters:
            datasets = all_datasets(cfg.datasets, 0, dataset_seed, "train", cfg.rollout)
        else:
            datasets = all_datasets(cfg.datasets, 0, dataset_seed, "train")
        batch_size = args.batch_size or 128
        calibration_data = build_calibration_loaders(
            datasets, calibration_num=args.calibration_num, batch_size=batch_size,
            num_workers=cfg.dataset_workers, random_calibration=args.random_calibration == "y",
            calibration_seed=args.calibration_seed,
        )
        args.max_calib_batches = int(args.calibration_num / batch_size)
        args.train_seed = cfg.train_seed

    # ===================================================================
    # Poseidon: complete training trajectories -> get_trainer/checkpoint
    # ===================================================================
    else:
        if not (package_root / "scOT").is_dir():
            raise FileNotFoundError(f"Official Poseidon source is missing: {package_root}/scOT. "
                                    "Install the camlab-ethz/poseidon checkout under poseidon/.")
        sys.path.insert(0, str(package_root))
        from scOT.inference import get_trainer, rollout
        from scOT.problems.base import get_dataset
        from safe_svd.helper.collector.poseidon.calibration_dataset import build_calibration_dataset
        dataset_kwargs = {"data_path": str(args.data_root)}
        if "elliptic" not in args.dataset:
            dataset_kwargs.update(fix_input_to_time_step=None, time_step_size=1, max_num_time_steps=args.final_time)
        print(f"==== Loading Poseidon calibration trajectories: {args.dataset} ====", flush=True)
        calibration_data = build_calibration_dataset(
            get_dataset, dataset_name=args.dataset, num_trajectories=args.traj_num,
            random_calibration=args.calibration_seed is not None,
            calibration_seed=args.calibration_seed or 0, **dataset_kwargs,
        )
        print(f"==== Loading Poseidon model: {args.ckpt_path} ====", flush=True)
        trainer = get_trainer(model_path=str(args.ckpt_path), batch_size=args.batch_size or 256,
                              dataset=calibration_data, workers=1)
        # Match the fork's effective zero-worker, metadata-preserving rollout.
        trainer.args.dataloader_num_workers = 0
        trainer.args.remove_unused_columns = False
        model = trainer.model.to(device).eval()
        rollout_fn = rollout
    return model, calibration_data, params, trainer, rollout_fn


def cli_main(argv=None):
    args = resolve_cli_paths(parse_args(argv))
    print("==== Unified SAFE-SVD configuration ====", flush=True)
    print(f"Model: {args.model_type}; device: {args.DEV}; rank_selector: {args.rank_selector}; rank_ratio: {args.rank_ratio}")
    print(f"Official model package: {args.runtime_root}")
    if args.model_type == "poseidon":
        print(f"Dataset: {args.dataset}; traj_num: {args.traj_num}; final_time: {args.final_time}; calibration_seed: {args.calibration_seed if args.calibration_seed is not None else 'default'}")
        print(f"Batches: rollout={args.batch_size or 256}; Fisher={args.fisher_batch_size}; capture={args.capture_batch_size}; cache=cpu; compression_dtype={'float32' if args.compression_backend == 'low-gpu' else 'float64'}")
    else:
        print(f"Calibration_num: {args.calibration_num}; calibration_seed: {args.calibration_seed}; trace: {args.use_trace_balance}")
    print(f"Whitening: {args.whitening_mode}/{args.eps_whiten_mode}/{args.eps_whiten}; Fisher: {args.use_fisher}/{args.gradient_loss}; stage: {args.pipeline_stage}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Data root: {args.data_root}")
    print(f"Input cache: {args.profile_cache}")
    print(f"Fisher cache: {args.fisher_cache}")
    print(f"Rank cache: {args.rank_cache}")
    print(f"Output: {args.save_path}")
    print(f"Compression checkpoint: {args.save_path / 'compressing_model.pt'}; interval={args.checkpoint_percent:.0%}; resume={not args.overwrite_cache}")
    if args.dry_run:
        return 0
    filename = "compressed_model.pth" if args.model_type == "vicon" else "compressed_model.pt"
    if args.pipeline_stage != "rank_selection" and (args.save_path / filename).is_file() and not args.overwrite_cache:
        raise FileExistsError(f"Model already exists: {args.save_path / filename}; choose a new save_path")
    model, calibration_data, params, trainer, rollout_fn = load_model_and_data(args)
    # Record resolved settings before the expensive stages, including failures.
    # This does not overwrite or invalidate any existing calibration cache.
    args.save_path.mkdir(parents=True, exist_ok=True)
    run_config_path = args.save_path / "run_config.json"
    config_file = args.ckpt_path / "config.json"
    run_config = {
        "argv": list(sys.argv[1:] if argv is None else argv),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch_version": str(torch.__version__),
        "checkpoint_config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest() if config_file.is_file() else None,
        "cache_exists_before_run": {name: getattr(args, name).is_file() for name in ("profile_cache", "fisher_cache", "rank_cache")},
        "compression_checkpoint_exists_before_run": (args.save_path / "compressing_model.pt").is_file(),
        "compression_dtype": "float32" if args.compression_backend == "low-gpu" else "float64",
        "poseidon_factor_dtype": "float32" if args.model_type == "poseidon" else None,
        "capture_cache_device": "cpu",
    }
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n")
    main(
        model, calibration_data,
        model_type=args.model_type,
        rank_ratio=args.rank_ratio,
        device=args.DEV,
        params=params,
        trainer=trainer,
        rollout_fn=rollout_fn,
        dataset_name=args.dataset if args.model_type == "poseidon" else None,
        use_fisher=args.use_fisher,
        rank_selector=args.rank_selector,
        alpha=args.alpha,
        compress_mode=args.compress_mode,
        gradient_loss=args.gradient_loss,
        balance_ratio=args.balance_ratio,
        decay_alpha=args.decay_alpha,
        order_control=args.order_control,
        calibration_seed=args.calibration_seed,
        batches_per_dset=args.batches_per_dset,
        max_calib_batches=args.max_calib_batches,
        train_seed=args.train_seed,
        fisher_batch_size=args.fisher_batch_size,
        capture_batch_size=args.capture_batch_size,
        use_trace_balance=args.use_trace_balance,
        whitening_mode=args.whitening_mode,
        eps_whiten=args.eps_whiten,
        eps_whiten_mode=args.eps_whiten_mode,
        compute_dtype=torch.float32 if args.compression_backend == "low-gpu" else torch.float64,
        offload_cache=True,
        profile_cache=args.profile_cache,
        fisher_cache=args.fisher_cache,
        rank_cache=args.rank_cache,
        save_path=args.save_path,
        overwrite_cache=args.overwrite_cache,
        checkpoint_percent=args.checkpoint_percent,
        pipeline_stage=args.pipeline_stage,
        perf_csv_path=args.save_path / "gpu_profile.csv" if args.gpu_profile else None,
    )
    if args.rank_cache.is_file():
        run_config["rank_cache_sha256"] = hashlib.sha256(args.rank_cache.read_bytes()).hexdigest()
    run_config["completed_stage"] = args.pipeline_stage
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
