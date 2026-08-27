"""Command-line bridge for SAFE-SVD's MPP runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from safesvd.adapters.mpp_runtime import MPPEvaluationOptions, MPPOptions, evaluate_mpp, run_mpp


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run SAFE-SVD compression or evaluation for MPP-AViT.")
    commands = result.add_subparsers(dest="command", required=True)
    compress = commands.add_parser("compress", help="profile, rank, and compress MPP")
    _add_common(compress)
    compress.add_argument("--save-path", type=Path, required=True)
    compress.add_argument("--profile-path", type=Path, required=True)
    compress.add_argument("--fisher-path", type=Path, required=True)
    compress.add_argument("--rank-path", type=Path, required=True)
    compress.add_argument("--rank-ratio", type=float, required=True)
    compress.add_argument("--calibration-size", type=int, required=True)
    compress.add_argument("--calibration-seed", type=int, required=True)
    compress.add_argument("--alpha", type=float, required=True)
    compress.add_argument("--use-fisher", choices=("y", "n"), default="y")
    compress.add_argument("--fisher-loss", choices=("l1", "mse"), default="l1")
    compress.add_argument("--balance-ratio", type=float, default=10.0)
    compress.add_argument("--decay-alpha", type=float, default=1.0)
    compress.add_argument("--trace-balance", choices=("y", "n"), default="y")
    compress.add_argument("--overwrite", action="store_true")

    evaluate = commands.add_parser("evaluate", help="evaluate original and compressed MPP models")
    _add_common(evaluate)
    evaluate.add_argument("--compressed-path", type=Path)
    evaluate.add_argument("--output-path", type=Path, required=True)
    evaluate.add_argument("--max-batches", type=int)
    return result


def _add_common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--mpp-root", type=Path, required=True)
    command.add_argument("--model-path", type=Path, required=True)
    command.add_argument("--data-path", type=Path, required=True)
    command.add_argument("--device", default="cuda")
    command.add_argument("--yaml-config", type=Path)
    command.add_argument("--config-name", default="basic_config")
    command.add_argument("--batch-size", type=int)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    mpp_root = args.mpp_root.expanduser().resolve()
    yaml_config = args.yaml_config.expanduser() if args.yaml_config else None
    if yaml_config is not None and not yaml_config.is_absolute():
        yaml_config = mpp_root / yaml_config
    common = {
        "mpp_root": mpp_root,
        "model_path": args.model_path.expanduser().resolve(),
        "data_path": args.data_path.expanduser().resolve(),
        "device": args.device,
        "yaml_config": yaml_config.resolve() if yaml_config else None,
        "config_name": args.config_name,
        "batch_size": args.batch_size,
    }
    if args.command == "compress":
        output = run_mpp(MPPOptions(
            **common,
            save_path=args.save_path.expanduser().resolve(),
            profile_path=args.profile_path.expanduser().resolve(),
            fisher_path=args.fisher_path.expanduser().resolve(),
            rank_path=args.rank_path.expanduser().resolve(),
            rank_ratio=args.rank_ratio,
            calibration_size=args.calibration_size,
            calibration_seed=args.calibration_seed,
            alpha=args.alpha,
            use_fisher=args.use_fisher == "y",
            fisher_loss=args.fisher_loss,
            balance_ratio=args.balance_ratio,
            decay_alpha=args.decay_alpha,
            trace_balance=args.trace_balance == "y",
            overwrite=args.overwrite,
        ))
        print(output)
        return 0
    output = evaluate_mpp(MPPEvaluationOptions(
        **common,
        compressed_path=args.compressed_path.expanduser().resolve() if args.compressed_path else None,
        output_path=args.output_path.expanduser().resolve(),
        max_batches=args.max_batches,
    ))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
