#!/usr/bin/env python3
"""Run VICON MySVD rank selection and compression sequentially."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


MYSVD_DIR = Path(__file__).resolve().parent
VICON_ROOT = MYSVD_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run rank.py first, verify its rank file, and then run mysvd.py "
            "with the identical calibration configuration."
        )
    )
    parser.add_argument("--calibration_num", type=int, default=2048)
    parser.add_argument(
        "--random_calibration",
        choices=("y", "n"),
        default="y",
    )
    parser.add_argument("--calibration_seed", type=int, default=42)
    parser.add_argument("--rank_ratio", type=float, default=0.6)
    parser.add_argument(
        "--gradient_loss",
        choices=("l1", "mse", "smooth_l1"),
        default="l1",
    )
    parser.add_argument("--balance_ratio", type=float, default=10.0)
    parser.add_argument(
        "--use_trace_align",
        choices=("y", "n"),
        default="y",
    )

    parser.add_argument("--DEV", default="cuda")
    parser.add_argument("--use_fisher", choices=("y", "n"), default="y")
    parser.add_argument("--rank_type", default="MySVD_Rank")
    parser.add_argument(
        "--compress_mode",
        choices=("clean_input", "comb"),
        default="comb",
    )
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--ckpt_path")
    parser.add_argument("--save_path")
    parser.add_argument(
        "--cache_dir",
        help="Directory for Fisher/input/rank caches; defaults to legacy MySVD paths.",
    )
    parser.add_argument(
        "--data_root",
        help="Prepared VICON data root. Forwarded for the VICON dataset loader configuration.",
    )
    parser.add_argument(
        "--rank_root",
        help="Directory for the generated rank JSON; defaults to legacy rank_results/MySVD_Rank.",
    )
    parser.add_argument(
        "--rank_path",
        help="Exact rank JSON path shared by rank selection and compression.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print both commands without running either stage.",
    )
    return parser.parse_args()


def append_override(command: list[str], name: str, value: object) -> None:
    command.append(f"+{name}={value}")


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    rank_command = [sys.executable, "-u", str(MYSVD_DIR / "rank.py")]
    compression_command = [sys.executable, "-u", str(MYSVD_DIR / "mysvd.py")]

    for name in (
        "calibration_num",
        "random_calibration",
        "calibration_seed",
        "rank_ratio",
        "gradient_loss",
        "balance_ratio",
        "use_trace_align",
    ):
        value = getattr(args, name)
        append_override(rank_command, name, value)
        append_override(compression_command, name, value)

    for name in (
        "DEV",
        "use_fisher",
        "rank_type",
        "compress_mode",
        "alpha",
        "save_path",
    ):
        value = getattr(args, name)
        if value is not None:
            append_override(compression_command, name, value)

    # Rank selection and compression must see exactly the same checkpoint,
    # external dataset root, and cache/output roots.  In particular, do not
    # let rank.py silently fall back to a private training checkpoint.
    for name in ("ckpt_path", "cache_dir", "data_root", "rank_root", "rank_path"):
        value = getattr(args, name)
        if value is not None:
            append_override(rank_command, name, value)
            append_override(compression_command, name, value)
    return rank_command, compression_command


def calibration_tag(args: argparse.Namespace) -> str:
    tag = f"cali{args.calibration_num}"
    if args.random_calibration == "y":
        tag += f"_random_seed{args.calibration_seed}"
    return tag


def expected_rank_path(args: argparse.Namespace) -> Path:
    if args.rank_path:
        path = Path(args.rank_path)
        return path if path.is_absolute() else VICON_ROOT / path
    trace_tag = "trace" if args.use_trace_align == "y" else "notrace"
    rank_root = Path(args.rank_root) if args.rank_root else VICON_ROOT / "rank_results" / "MySVD_Rank"
    if not rank_root.is_absolute():
        rank_root = VICON_ROOT / rank_root
    return (
        rank_root
        / f"{calibration_tag(args)}_{args.gradient_loss}_{trace_tag}"
        / f"rank_{args.rank_ratio}.json"
    )


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    entries = [str(VICON_ROOT)]
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
            cwd=VICON_ROOT,
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
    print("==== Safe VICON MySVD pipeline configuration ====")
    print(
        f"calibration_num={args.calibration_num}, "
        f"random_calibration={args.random_calibration}, "
        f"calibration_seed={args.calibration_seed}, "
        f"rank_ratio={args.rank_ratio}, "
        f"use_trace_align={args.use_trace_align}"
    )
    print(f"Expected rank file: {rank_path}")

    if args.dry_run:
        print(f"\n[rank selection] {shlex.join(rank_command)}")
        print(f"\n[compression] {shlex.join(compression_command)}")
        return 0

    env = child_environment()
    if args.data_root:
        # The VICON loader belongs to the model source tree.  Its public
        # version should consume this variable (or the Hydra data_root field)
        # instead of a personal machine path.
        env["SAFE_SVD_DATA_ROOT"] = str(Path(args.data_root).resolve())
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

    print("\n==== Safe VICON MySVD pipeline completed successfully. ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
