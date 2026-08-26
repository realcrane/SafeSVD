"""Command line for the public SAFE-SVD interface."""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from safesvd import SafeSVDConfig, compress


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="SAFE-SVD compression for scientific foundation models.")
    commands = result.add_subparsers(dest="command", required=True)
    compress_parser = commands.add_parser("compress", help="select ranks and compress a model")
    compress_parser.add_argument("--model", choices=("poseidon", "vicon", "mpp"), required=True)
    compress_parser.add_argument("--checkpoint", type=Path)
    compress_parser.add_argument("--data-root", type=Path)
    compress_parser.add_argument("--output-dir", type=Path)
    compress_parser.add_argument("--rank-ratio", type=float, required=True)
    compress_parser.add_argument("--calibration-size", type=int, required=True)
    compress_parser.add_argument(
        "--calibration-seed",
        type=int,
        help="For Poseidon, -1 uses the fixed trajectory prefix; non-negative values select a seeded random subset.",
    )
    compress_parser.add_argument(
        "--dataset",
        default="fluids.incompressible.PiecewiseConstants",
        help="Poseidon downstream dataset used for calibration.",
    )
    compress_parser.add_argument(
        "--final-time",
        type=int,
        default=20,
        help="Poseidon calibration rollout horizon.",
    )
    compress_parser.add_argument("--alpha", type=float, default=0.7)
    compress_parser.add_argument("--device", default="cuda")
    compress_parser.add_argument("--preset", choices=("paper",), default="paper")
    compress_parser.add_argument("--overwrite", action="store_true")
    compress_parser.add_argument("--dry-run", action="store_true")
    compress_parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Temporary backend-specific options after '--'; avoid in published recipes.",
    )
    return result


def poseidon_calibration_tag(
    calibration_size: int,
    final_time: int,
    *,
    calibration_seed: int,
) -> str:
    """Build the tag shared by Poseidon output, rank, and cache artifacts."""
    selection = "default" if calibration_seed == -1 else f"seed{calibration_seed}"
    return f"traj{calibration_size}_{selection}_ftime{final_time}"


def default_output_dir(args: argparse.Namespace) -> Path:
    """Return a reproducible, hyperparameter-addressed run directory."""
    if args.model == "poseidon":
        return (
            PROJECT_DIR
            / "artifacts"
            / "poseidon"
            / args.preset
            / str(args.rank_ratio)
            / args.dataset
            / poseidon_calibration_tag(
                args.calibration_size,
                args.final_time,
                calibration_seed=args.calibration_seed,
            )
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_DIR / "artifacts" / args.model / args.preset / timestamp


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command != "compress":
        raise AssertionError(f"Unhandled command: {args.command}")
    extra = tuple(args.extra[1:] if args.extra[:1] == ["--"] else args.extra)
    calibration_seed = args.calibration_seed
    if calibration_seed is None:
        calibration_seed = -1 if args.model == "poseidon" else 42
    args.calibration_seed = calibration_seed
    if args.model == "poseidon" and calibration_seed < -1:
        raise ValueError("Poseidon --calibration-seed must be -1 or a non-negative integer.")
    if args.model != "poseidon" and (args.dataset != "fluids.incompressible.PiecewiseConstants" or args.final_time != 20):
        raise ValueError("--dataset and --final-time are currently Poseidon-only options.")
    if args.model == "poseidon":
        reserved = {
            "--dataset", "--final_time", "--random_calibration", "--calibration_seed",
            "--traj_num", "--rank_ratio",
        }
        conflicting = sorted(set(extra) & reserved)
        if conflicting:
            raise ValueError(
                f"Pass {', '.join(conflicting)} through the SAFE-SVD CLI options, not after '--'."
            )
    poseidon_extra = (
        "--dataset", args.dataset,
        "--final_time", str(args.final_time),
        "--calibration_seed", str(calibration_seed),
    ) if args.model == "poseidon" else ()
    checkpoint = args.checkpoint or PROJECT_DIR / "models" / args.model
    default_data_root = (
        PROJECT_DIR / "poseidon" / "camlab-ethz" / "down_streams"
        if args.model == "poseidon"
        else PROJECT_DIR / "data" / args.model
    )
    config = SafeSVDConfig(
        model=args.model,
        checkpoint=checkpoint,
        data_root=args.data_root or default_data_root,
        output_dir=args.output_dir or default_output_dir(args),
        rank_ratio=args.rank_ratio,
        calibration_size=args.calibration_size,
        calibration_seed=calibration_seed,
        alpha=args.alpha,
        device=args.device,
        preset=args.preset,
        overwrite=args.overwrite,
        extra=(*extra, *poseidon_extra),
    )
    result = compress(config, dry_run=args.dry_run)
    print(f"SAFE-SVD {config.model}/{config.preset}")
    print(f"Output: {result.output_dir}")
    print(f"Backend command: {shlex.join(result.backend_command)}")
    if result.returncode not in (None, 0):
        print(f"ERROR: backend exited before completion (exit code {result.returncode}).", file=sys.stderr)
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
