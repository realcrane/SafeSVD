#!/usr/bin/env python3
"""Run MySVD_origin rank selection and compression as one safe pipeline."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


MYSVD_ORIGIN_DIR = Path(__file__).resolve().parent
POSEIDON_ROOT = MYSVD_ORIGIN_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MySVD_origin/mysvd_rank.py, verify the rank file, then run "
            "MySVD_origin/mysvd.py using the identical calibration selection."
        )
    )
    parser.add_argument("--model_name", default="Poseidon-L")
    parser.add_argument("--dataset", default="fluids.incompressible.PiecewiseConstants")
    parser.add_argument("--traj_num", type=int, default=64)
    parser.add_argument("--random_calibration", choices=("y", "n"), default="n")
    parser.add_argument("--calibration_seed", type=int, default=2026)
    parser.add_argument("--use_phyloss", default="no_phyloss")
    parser.add_argument("--final_time", type=int, default=20)
    parser.add_argument("--time_back", type=int, default=0)
    parser.add_argument("--time_mode", default="alltime")

    parser.add_argument("--rank_ratio", type=float, default=0.2)
    parser.add_argument("--DEV", default="cuda")
    parser.add_argument("--use_fisher", choices=("y", "n"), default="y")
    parser.add_argument("--use_prev_time", default="False")
    parser.add_argument("--balance_ratio", type=float, default=10.0)
    parser.add_argument("--order_control", type=int, default=2)
    parser.add_argument("--gradient_loss", choices=("mse", "l1"), default="l1")

    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--uniform", choices=("y", "n"), default="n")
    parser.add_argument("--energy_convert", default="")
    parser.add_argument("--rank_type", default="MySVD_origin_Rank")
    parser.add_argument("--skip_layer", default="n")

    parser.add_argument("--model_path")
    parser.add_argument(
        "--data_path",
        help="Prepared Poseidon downstream-data root. Defaults to the legacy local path.",
    )
    parser.add_argument("--rank_path")
    parser.add_argument("--save_path")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Rebuild rank-stage Fisher/profile/rank artifacts before compression.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def append_option(command: list[str], name: str, value: object) -> None:
    command.extend((f"--{name}", str(value)))


def calibration_tag(args: argparse.Namespace) -> str:
    tag = f"traj{args.traj_num}"
    if args.random_calibration == "y":
        tag += f"_random_seed{args.calibration_seed}"
    return f"{tag}_ftime{args.final_time}"


def expected_rank_path(args: argparse.Namespace) -> Path:
    if args.rank_path is not None:
        candidate = Path(args.rank_path)
        return candidate if candidate.is_absolute() else POSEIDON_ROOT / candidate
    return (
        POSEIDON_ROOT
        / "rank_results"
        / args.model_name
        / args.dataset
        / "MySVD_origin_Rank"
        / f"{calibration_tag(args)}_order{args.order_control}_{args.gradient_loss}"
        / f"rank_{args.rank_ratio}_origin.json"
    )


def common_command(script: Path, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-u", str(script)]
    for name in (
        "model_name",
        "dataset",
        "traj_num",
        "random_calibration",
        "calibration_seed",
        "use_phyloss",
        "final_time",
        "time_back",
        "time_mode",
        "rank_ratio",
        "DEV",
        "use_fisher",
        "use_prev_time",
        "balance_ratio",
        "order_control",
        "gradient_loss",
    ):
        append_option(command, name, getattr(args, name))
    for name in ("model_path", "data_path", "rank_path"):
        value = getattr(args, name)
        if value is not None:
            append_option(command, name, value)
    return command


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    rank_command = common_command(MYSVD_ORIGIN_DIR / "mysvd_rank.py", args)
    if args.overwrite_cache:
        rank_command.append("--overwrite_cache")

    compression_command = common_command(MYSVD_ORIGIN_DIR / "mysvd.py", args)
    for name in ("alpha", "uniform", "energy_convert", "rank_type", "skip_layer"):
        append_option(compression_command, name, getattr(args, name))
    if args.save_path is not None:
        append_option(compression_command, "save_path", args.save_path)
    return rank_command, compression_command


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(POSEIDON_ROOT)
        if not pythonpath
        else os.pathsep.join((str(POSEIDON_ROOT), pythonpath))
    )
    return environment


def run_stage(stage_name: str, command: list[str], environment: dict[str, str]) -> int:
    print(f"\n==== Starting {stage_name} ====", flush=True)
    print(f"$ {shlex.join(command)}", flush=True)
    started_at = time.monotonic()
    result = subprocess.run(command, cwd=POSEIDON_ROOT, env=environment, check=False)
    elapsed = time.monotonic() - started_at
    if result.returncode:
        print(
            f"==== {stage_name} failed with exit code {result.returncode} "
            f"after {elapsed:.1f}s ====",
            file=sys.stderr,
        )
    else:
        print(f"==== Finished {stage_name} in {elapsed:.1f}s ====", flush=True)
    return result.returncode


def main() -> int:
    args = parse_args()
    if args.traj_num <= 0:
        raise SystemExit("--traj_num must be positive")
    if args.final_time <= args.time_back:
        raise SystemExit("--final_time must be greater than --time_back")

    rank_command, compression_command = build_commands(args)
    rank_path = expected_rank_path(args)
    print("==== Safe MySVD_origin pipeline configuration ====")
    print(
        f"dataset={args.dataset}, traj_num={args.traj_num}, "
        f"random_calibration={args.random_calibration}, "
        f"calibration_seed={args.calibration_seed}, final_time={args.final_time}, "
        f"rank_ratio={args.rank_ratio}, alpha={args.alpha}"
    )
    print(f"Expected rank file: {rank_path}")
    if args.dry_run:
        print(f"\n[rank selection] {shlex.join(rank_command)}")
        print(f"\n[compression] {shlex.join(compression_command)}")
        return 0

    environment = child_environment()
    if run_stage("rank selection", rank_command, environment) != 0:
        print("==== Compression skipped because rank selection failed. ====")
        return 1
    if not rank_path.is_file():
        print(
            f"==== Compression skipped: rank selection did not produce {rank_path}. ====",
            file=sys.stderr,
        )
        return 1
    if run_stage("compression", compression_command, environment) != 0:
        return 1
    print("\n==== Safe MySVD_origin pipeline completed successfully. ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
