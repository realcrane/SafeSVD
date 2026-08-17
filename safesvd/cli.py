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
    compress_parser.add_argument("--calibration-seed", type=int, default=42)
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


def default_output_dir(model: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_DIR / "artifacts" / model / "paper" / timestamp


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command != "compress":
        raise AssertionError(f"Unhandled command: {args.command}")
    extra = tuple(args.extra[1:] if args.extra[:1] == ["--"] else args.extra)
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
        output_dir=args.output_dir or default_output_dir(args.model),
        rank_ratio=args.rank_ratio,
        calibration_size=args.calibration_size,
        calibration_seed=args.calibration_seed,
        alpha=args.alpha,
        device=args.device,
        preset=args.preset,
        overwrite=args.overwrite,
        extra=extra,
    )
    result = compress(config, dry_run=args.dry_run)
    print(f"SAFE-SVD {config.model}/{config.preset}")
    print(f"Output: {result.output_dir}")
    print(f"Backend command: {shlex.join(result.backend_command)}")
    return 0 if result.returncode is None else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
