"""CLI bridge for the SAFE-SVD Poseidon adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from safesvd.adapters.poseidon_runtime import PoseidonOptions, run_poseidon


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Compress a Poseidon model with SAFE-SVD core.")
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--poseidon-root", type=Path, required=True)
    result.add_argument("--model_path", type=Path, required=True)
    result.add_argument("--data_path", type=Path, required=True)
    result.add_argument("--save_path", type=Path, required=True)
    result.add_argument("--rank-root", type=Path, required=True)
    result.add_argument("--cache-root", type=Path, required=True)
    result.add_argument("--model_name", default="Poseidon-L")
    result.add_argument("--dataset", default="fluids.incompressible.PiecewiseConstants")
    result.add_argument("--traj_num", type=int, required=True)
    result.add_argument("--random_calibration", choices=("y", "n"), default="n")
    result.add_argument("--calibration_seed", type=int, default=2026)
    result.add_argument("--final_time", type=int, default=15)
    result.add_argument("--time_back", type=int, default=0)
    result.add_argument("--rank_ratio", type=float, required=True)
    result.add_argument("--DEV", default="cuda")
    result.add_argument("--use_fisher", choices=("y", "n"), default="y")
    result.add_argument("--use_prev_time", choices=("True", "False"), default="False")
    result.add_argument("--balance_ratio", type=float, default=10.0)
    result.add_argument("--order_control", type=int, default=2)
    result.add_argument("--gradient_loss", choices=("mse", "l1", "smooth_l1"), default="l1")
    result.add_argument("--alpha", type=float, default=0.7)
    result.add_argument(
        "--fast",
        action="store_true",
        help="Deprecated compatibility flag; Poseidon compression always uses MySVD_fast's 512-sample capture chunk.",
    )
    result.add_argument("--overwrite_cache", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = run_poseidon(PoseidonOptions(
        repo_root=args.repo_root.resolve(), poseidon_root=args.poseidon_root.resolve(),
        model_path=args.model_path.resolve(), data_path=args.data_path.resolve(), save_path=args.save_path.resolve(),
        rank_root=args.rank_root.resolve(), cache_root=args.cache_root.resolve(),
        model_name=args.model_name, dataset=args.dataset, traj_num=args.traj_num,
        random_calibration=args.random_calibration == "y", calibration_seed=args.calibration_seed,
        final_time=args.final_time, time_back=args.time_back, rank_ratio=args.rank_ratio, device=args.DEV,
        use_fisher=args.use_fisher == "y", use_prev_time=args.use_prev_time == "True",
        balance_ratio=args.balance_ratio, order_control=args.order_control,
        gradient_loss=args.gradient_loss, alpha=args.alpha, fast=args.fast, overwrite=args.overwrite_cache,
    ))
    print(f"Compressed Poseidon model saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
