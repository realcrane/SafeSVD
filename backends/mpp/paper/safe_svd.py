#!/usr/bin/env python3
"""Run MPP MySVD rank selection and compression sequentially."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


MYSVD_DIR = Path(__file__).resolve().parent
MPP_ROOT = MYSVD_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the rank-selection stage of mysvd.py first, verify its rank "
            "cache, and then run the unchanged MySVD compression stage."
        )
    )
    parser.add_argument(
        "--yaml_config",
        default="multiple_physics_pretraining/config/mpp_avit_L_config.yaml",
    )
    parser.add_argument("--config", default="basic_config")
    parser.add_argument("--ckpt_path", default="models/MPP_AViT_L")
    parser.add_argument(
        "--data_root",
        help="Prefix relative MPP dataset paths in the selected YAML configuration.",
    )
    parser.add_argument("--rank_ratio", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--DEV", default="cuda")
    parser.add_argument("--calibration_num", type=int, default=16)
    parser.add_argument("--calibration_seed", type=int, default=1204)
    parser.add_argument(
        "--use_trace_balance",
        choices=("y", "n"),
        default="y",
    )
    parser.add_argument("--use_fisher", choices=("y", "n"), default="y")
    parser.add_argument(
        "--fisher_loss",
        choices=("l1", "mse"),
        default="l1",
    )
    parser.add_argument("--balance_ratio", type=float, default=10.0)
    parser.add_argument("--decay_alpha", type=float, default=1.0)
    parser.add_argument("--save_path")
    parser.add_argument("--profile_cache")
    parser.add_argument("--fisher_cache")
    parser.add_argument("--rank_cache")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print both commands without running either stage.",
    )
    return parser.parse_args()


def append_option(command: list[str], name: str, value: object) -> None:
    command.extend((f"--{name}", str(value)))


def shared_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-u", str(MYSVD_DIR / "mysvd.py")]
    for name in (
        "yaml_config",
        "config",
        "ckpt_path",
        "data_root",
        "rank_ratio",
        "alpha",
        "DEV",
        "calibration_num",
        "calibration_seed",
        "use_trace_balance",
        "use_fisher",
        "fisher_loss",
        "balance_ratio",
        "decay_alpha",
    ):
        append_option(command, name, getattr(args, name))

    for name in (
        "save_path",
        "profile_cache",
        "fisher_cache",
        "rank_cache",
    ):
        value = getattr(args, name)
        if value is not None:
            append_option(command, name, value)
    return command


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    common = shared_command(args)
    rank_command = [*common, "--pipeline_stage", "rank_selection"]
    compression_command = [*common, "--pipeline_stage", "compression"]
    return rank_command, compression_command


def expected_rank_path(args: argparse.Namespace) -> Path:
    if args.rank_cache is not None:
        path = Path(args.rank_cache)
        return path if path.is_absolute() else MPP_ROOT / path

    trace_tag = "_trace" if args.use_trace_balance == "y" else "_notrace"
    calibration_tag = f"cali{args.calibration_num}_seed{args.calibration_seed}"
    return (
        MPP_ROOT
        / "MySVD"
        / "cache"
        / calibration_tag
        / f"{args.rank_ratio}{trace_tag}.json"
    )


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    entries = [str(MPP_ROOT)]
    if existing_pythonpath:
        entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def run_stage(stage_name: str, command: list[str], env: dict[str, str]) -> int:
    print(f"\n==== Starting {stage_name} ====", flush=True)
    print(f"$ {shlex.join(command)}", flush=True)
    started_at = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=MPP_ROOT,
            env=env,
            check=False,
        )
    except OSError as exc:
        print(f"==== Failed to start {stage_name}: {exc} ====", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started_at
    if result.returncode != 0:
        print(
            f"==== {stage_name} failed with exit code {result.returncode} "
            f"after {elapsed:.1f}s ====",
            file=sys.stderr,
        )
        return result.returncode
    print(f"==== Finished {stage_name} in {elapsed:.1f}s ====", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.calibration_num <= 0:
        raise SystemExit("--calibration_num must be positive")
    if not 0 < args.rank_ratio <= 1:
        raise SystemExit("--rank_ratio must be in (0, 1]")

    rank_command, compression_command = build_commands(args)
    rank_path = expected_rank_path(args)
    print("==== Safe MPP MySVD pipeline configuration ====")
    print(
        f"calibration_num={args.calibration_num}, "
        f"calibration_seed={args.calibration_seed}, "
        f"rank_ratio={args.rank_ratio}, "
        f"use_trace_balance={args.use_trace_balance}"
    )
    print(f"Expected rank file: {rank_path}")

    if args.dry_run:
        print(f"\n[rank selection] {shlex.join(rank_command)}")
        print(f"\n[compression] {shlex.join(compression_command)}")
        return 0

    env = child_environment()
    rank_returncode = run_stage("rank selection", rank_command, env)
    if rank_returncode != 0:
        print("==== Compression skipped because rank selection failed. ====")
        return rank_returncode
    if not rank_path.is_file():
        print(
            "==== Compression skipped because rank selection returned success "
            f"but did not produce {rank_path}. ====",
            file=sys.stderr,
        )
        return 1

    compression_returncode = run_stage(
        "compression",
        compression_command,
        env,
    )
    if compression_returncode != 0:
        return compression_returncode

    print("\n==== Safe MPP MySVD pipeline completed successfully. ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
