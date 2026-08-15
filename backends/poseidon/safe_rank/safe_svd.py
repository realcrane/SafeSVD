#!/usr/bin/env python3
"""Load SAFE-SVD once, then run rank selection and compression in-process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SAFE_SVD_DIR = Path(__file__).resolve().parent
POSEIDON_ROOT = SAFE_SVD_DIR.parent
if str(POSEIDON_ROOT) not in sys.path:
    sys.path.insert(0, str(POSEIDON_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from poseidon_main.scOT.inference import get_trainer, rollout
from poseidon_main.scOT.problems.base import get_dataset

from SAFE_SVD.compress import compress
from SAFE_SVD.helper.calibration_dataset import (
    build_calibration_dataset,
    calibration_cache_tag,
    describe_calibration_selection,
)
from SAFE_SVD.helper.fisher_collector import collect_fisher_dict_linear_only
from SAFE_SVD.helper.input_collector import profile_inputs
from SAFE_SVD.rank_selector import (
    get_global_greedy_rank_dict_covariance_invariant,
    get_global_greedy_rank_dict_v2,
)


def calibration_seed_arg(value: str) -> int | None:
    """Parse ``default`` or a non-negative random-calibration seed."""
    if value == "default":
        return None
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be 'default' or a non-negative integer seed"
        ) from exc
    if seed < 0:
        raise argparse.ArgumentTypeError(
            "must be 'default' or a non-negative integer seed"
        )
    return seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rank selection and compression with one loaded model."
    )
    parser.add_argument("--model_name", default="Poseidon-L")
    parser.add_argument(
        "--dataset", default="fluids.incompressible.PiecewiseConstants"
    )
    parser.add_argument("--traj_num", type=int, default=64)
    parser.add_argument(
        "--calibration_seed",
        type=calibration_seed_arg,
        default=None,
        metavar="default|SEED",
        help=(
            "Use 'default' (the default) for the first traj_num training "
            "trajectories in order, or a non-negative integer to randomly "
            "sample complete trajectories with that seed."
        ),
    )
    parser.add_argument("--use_phyloss", default="no_phyloss")
    parser.add_argument("--final_time", type=int, default=20)
    parser.add_argument("--rank_ratio", type=float, default=0.2)
    parser.add_argument("--whitening_eps", type=float, default=1e-6)
    parser.add_argument(
        "--whitening_eps_mode",
        choices=["absolute", "relative", "sum_equivalent"],
        default="absolute",
        help=(
            "Interpret whitening_eps as an absolute covariance value, a relative "
            "coefficient rho, or a raw-Gram value converted per layer by 1/N."
        ),
    )
    parser.add_argument(
        "--whitening_mode", choices=["origin", "new"], default="origin",
        help="Input/covariance whitening: Fisher-style origin or spectral new.",
    )
    parser.add_argument("--DEV", default="cuda")
    parser.add_argument("--balance_ratio", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--order_control", type=int, default=2)
    parser.add_argument(
        "--gradient_loss", choices=("mse", "l1", "smooth_l1"), default="l1"
    )
    parser.add_argument(
        "--compression_backend",
        choices=("GPU", "low-gpu"),
        default="GPU",
        help="low-gpu uses float32 covariance/SVD workspaces.",
    )
    parser.add_argument(
        "--gpu_profile",
        action="store_true",
        help="Write compression-stage timing and GPU-memory rows.",
    )
    parser.add_argument("--model_path")
    parser.add_argument("--rank_path")
    parser.add_argument("--save_path")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument(
        "--dry_run", action="store_true", help="Print resolved paths without loading data."
    )
    return parser.parse_args()


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else POSEIDON_ROOT / path


def whitening_cache_tag(args: argparse.Namespace) -> str:
    """Return a filesystem-safe tag that distinguishes whitening settings."""
    # This marks empirical-mean covariance statistics and its corresponding
    # covariance-invariant global rank score.  It prevents reuse of legacy
    # raw-Gram caches and of ranks generated before the score correction.
    tag = f"whitening_{args.whitening_mode}_covmean"
    if args.whitening_mode == "new":
        tag = (
            f"{tag}_rankscale_eps{args.whitening_eps_mode}"
            f"{str(args.whitening_eps).replace('+', '')}"
        )
    return tag


def resolve_pipeline_paths(args: argparse.Namespace, calibration_tag: str):
    # resolve_path(...) -> Path | None，例如 "checkpoints/custom" -> POSEIDON_ROOT / "checkpoints/custom"。
    model_path = resolve_path(args.model_path) or (
        POSEIDON_ROOT
        / "checkpoints"
        / args.model_name
        / args.use_phyloss
        / args.dataset
        / "origin"
        / "none"
    )
    whitening_tag = whitening_cache_tag(args)
    # resolve_path(...) -> Path | None，例如未传 --rank_path 时返回 None 并使用下面的默认路径。
    rank_path = resolve_path(args.rank_path) or (
        POSEIDON_ROOT
        / "rank_results"
        / args.model_name
        / args.dataset
        / "SAFE_SVD"
        / f"{calibration_tag}_order{args.order_control}_{args.gradient_loss}_{whitening_tag}"
        / f"rank_{args.rank_ratio}.json"
    )
    # resolve_path(...) -> Path | None，例如 "/tmp/safe" -> Path("/tmp/safe")。
    save_path = resolve_path(args.save_path) or (
        POSEIDON_ROOT
        / "compression/compressed_ft_models"
        / args.dataset
        / str(args.rank_ratio)
        / "SAFE_SVD"
        / f"{calibration_tag}_{args.gradient_loss}_order{args.order_control}_alpha{args.alpha}_{whitening_tag}"
    )
    cache_root = POSEIDON_ROOT / "SAFE_SVD" / "cache" / args.dataset
    fisher_cache = (
        cache_root
        / "fisher"
        / f"{calibration_tag}_order{args.order_control}_{args.gradient_loss}_{whitening_tag}"
        / "Ldict.pt"
    )
    profile_cache = cache_root / "input" / calibration_tag / f"profiling_mat_{whitening_tag}.pt"
    return model_path, rank_path, save_path, fisher_cache, profile_cache


def dataset_constants(dataset, dataset_name: str):
    """Return normalization values such as {"mean": ..., "std": ...}."""
    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    if hasattr(base, "constants"):
        constants = base.constants
        if "mean_solution" in constants:
            return {
                "mean": constants["mean_solution"],
                "std": constants["std_solution"],
            }
        return constants
    if hasattr(base, "mean") and hasattr(base, "std"):
        return {"mean": base.mean, "std": base.std}
    raise AttributeError(f"No normalization constants found for {dataset_name}")

def main() -> int:
    # parse_args() -> argparse.Namespace，such as Namespace(traj_num=64, rank_ratio=0.2, ...).
    args = parse_args()
    if args.traj_num <= 0:
        raise SystemExit("--traj_num must be positive")
    if not 0 < args.rank_ratio <= 1:
        raise SystemExit("--rank_ratio must be in (0, 1]")
    if not 0 <= args.alpha <= 1:
        raise SystemExit("--alpha must be in [0, 1]")

    random_calibration = args.calibration_seed is not None
    # calibration_cache_tag(...) -> 名称字符串，例如 "traj64_ftime20"；随机模式示例为 "traj64_random_seed42_ftime20"。
    calibration_tag = calibration_cache_tag(
        args.traj_num,
        random_calibration,
        args.calibration_seed or 0,
        args.final_time,
    )
    # resolve_pipeline_paths(...) -> 五个 Path：model、rank JSON、输出目录、Fisher cache、input-profile cache。
    (
        model_path,
        rank_path,
        save_path,
        fisher_cache,
        profile_cache,
    ) = resolve_pipeline_paths(args, calibration_tag)
    final_path = save_path / "compressed_model.pt"

    print("==== SAFE-SVD single-process pipeline ====")
    print(
        f"dataset={args.dataset}, traj_num={args.traj_num}, "
        f"calibration_seed={'default' if args.calibration_seed is None else args.calibration_seed}, "
        f"final_time={args.final_time}, rank_ratio={args.rank_ratio}"
    )
    print(f"Model: {model_path}")
    print(f"Rank file: {rank_path}")
    print(f"Output: {final_path}")
    print(
        "Acceleration: MLP pair-cache + attention QKV/output cache + "
        "PatchUnmerging cache"
    )
    if args.dry_run:
        return 0
    if final_path.is_file() and not args.overwrite_cache:
        print(f"==== Compressed model already exists: {final_path} ====")
        return 0

    dataset_kwargs = {
        "data_path": str(POSEIDON_ROOT / "poseidon_main/camlab-ethz/down_streams")
    }
    if "elliptic" not in args.dataset:
        # dict.update(...) -> None；dataset_kwargs 会新增完整 final_time 校准范围。
        dataset_kwargs.update(
            fix_input_to_time_step=None,
            time_step_size=1,
            max_num_time_steps=args.final_time,
        )

    print(f"==== Loading calibration dataset: {args.dataset} ====")
    # build_calibration_dataset(...) -> Dataset；例如 traj_num=64 时包含 64 条完整校准轨迹对应的所有样本。
    calibration_data = build_calibration_dataset(
        get_dataset,
        dataset_name=args.dataset,
        num_trajectories=args.traj_num,
        random_calibration=random_calibration,
        calibration_seed=args.calibration_seed,
        **dataset_kwargs,
    )
    # describe_calibration_selection(...) -> 可读字符串，例如 "fixed training trajectory prefix: [0, 64)"。
    calibration_description = describe_calibration_selection(
        calibration_data, random_calibration
    )
    print(
        "==== Calibration selection: "
        f"{calibration_description} ===="
    )

    print(f"==== Loading model once: {model_path} ====")
    # get_trainer(...) -> Trainer；实际模型位于 trainer.model，后续 rank 与 compress 共用同一对象。
    trainer = get_trainer(
        model_path=str(model_path), batch_size=256, dataset=calibration_data
    )

    print("==== Rank selection ====")
    left_metrics = None
    if fisher_cache.is_file() and not args.overwrite_cache:
        # torch.load(...) -> {layer_name: (Ll, Ll_invT)}，每个矩阵为该层输出维度的 Fisher whitener。
        left_metrics = torch.load(fisher_cache, map_location="cpu")
        print(f"==== Loading Fisher cache: {fisher_cache} ====")

    if rank_path.is_file() and not args.overwrite_cache:
        with rank_path.open(encoding="utf-8") as handle:
            # json.load(...) -> {layer_name: rank | None}，例如 {"encoder...query": 64}。
            ranks = json.load(handle) or {}
        print(f"==== Loading existing rank file: {rank_path} ====")
    else:
        input_sample_counts = None
        if profile_cache.is_file() and not args.overwrite_cache:
            cached_profile = torch.load(profile_cache, map_location="cpu")
            if (
                isinstance(cached_profile, dict)
                and "profile_mat" in cached_profile
                and "sample_counts" in cached_profile
            ):
                input_metrics = cached_profile["profile_mat"]
                input_sample_counts = cached_profile["sample_counts"]
                print(f"==== Loading input profile cache: {profile_cache} ====")
            else:
                # Earlier caches contained factors only.  The covariance-invariant
                # selector needs the per-layer counts, so rebuild this cache once.
                print(
                    "==== Legacy input profile cache has no sample counts; "
                    "rebuilding it for covariance-invariant rank selection. ===="
                )

        if input_sample_counts is None and (
            not profile_cache.is_file() or args.overwrite_cache or args.whitening_mode == "new"
        ):
            # profile_inputs(...) -> (whiteners, {layer_name: flattened sample count}).
            input_metrics, input_sample_counts = profile_inputs(
                trainer,
                calibration_data,
                args.DEV,
                whitening_mode=args.whitening_mode,
                whitening_eps=args.whitening_eps,
                whitening_eps_mode=args.whitening_eps_mode,
                return_sample_counts=True,
            )
            profile_cache.parent.mkdir(parents=True, exist_ok=True)
            # Keep counts with the factors: normalized-covariance rank selection
            # needs them to compare layer energies on the original global scale.
            torch.save(
                {
                    "profile_mat": input_metrics,
                    "sample_counts": input_sample_counts,
                },
                profile_cache,
            )
            print(f"==== Input profile cache saved: {profile_cache} ====")

        if args.whitening_mode == "new":
            ranks = get_global_greedy_rank_dict_covariance_invariant(
                trainer,
                left_metrics,
                input_metrics,
                args.rank_ratio,
                input_sample_counts,
                dev=args.DEV,
            )
        else:
            ranks = get_global_greedy_rank_dict_v2(
                trainer,
                left_metrics,
                input_metrics,
                args.rank_ratio,
                dev=args.DEV,
            )
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        with rank_path.open("w", encoding="utf-8") as handle:
            # json.dump(...) -> None；rank JSON 示例为 {"encoder.layers.0...query": 64}。
            json.dump(ranks, handle, indent=2)
        print(f"==== Rank file saved: {rank_path} ====")

    print("==== Sequential compression ====")
    save_path.mkdir(parents=True, exist_ok=True)
    # str(...) -> GPU profile CSV 路径，例如 ".../gpu_profile.csv"；未启用时为 None。
    gpu_profile_csv = str(save_path / "gpu_profile.csv") if args.gpu_profile else None
    # compress(...) -> 已移到 CPU 的压缩 nn.Module；内部启用 MLP、attention、PatchUnmerging 三类缓存。
    compressed_model = compress(
        trainer,
        calibration_data,
        ranks,
        device_name=args.DEV,
        left_metrics=left_metrics,
        alpha=args.alpha,
        save_path=save_path,
        eps_whiten=args.whitening_eps,
        eps_whiten_mode=args.whitening_eps_mode,
        whitening_mode=args.whitening_mode,
        low_gpu=args.compression_backend == "low-gpu",
        overwrite_checkpoint=args.overwrite_cache,
        perf_csv_path=gpu_profile_csv,
    )
    # torch.save(...) -> None；最终文件包含可直接加载的压缩模型对象。
    torch.save(compressed_model, final_path)
    print(f"==== SAFE-SVD model saved: {final_path} ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
