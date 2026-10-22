"""Command line for the public SAFE-SVD interface."""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from safesvd import SafeSVDConfig, SafeSVDEvaluationConfig, compress, evaluate


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="SAFE-SVD compression for scientific foundation models.")
    commands = result.add_subparsers(dest="command", required=True)
    compress_parser = commands.add_parser("compress", help="select ranks and compress a model")
    compress_parser.add_argument("--model", choices=("poseidon", "vicon", "mpp"), required=True)
    compress_parser.add_argument("--checkpoint", type=Path)
    compress_parser.add_argument("--data-root", type=Path)
    compress_parser.add_argument("--output-dir", type=Path)
    compress_parser.add_argument(
        "--source-root",
        type=Path,
        help="Source checkout for the backend (defaults to ./poseidon, bundled ./VICON, or ./mpp).",
    )
    compress_parser.add_argument("--rank-ratio", type=float, required=True)
    compress_parser.add_argument(
        "--rank-selector",
        choices=("safe", "legacy"),
        default=None,
        help="Rank allocator (default: legacy for MPP/VICON, safe for Poseidon).",
    )
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
    evaluate_parser = commands.add_parser("evaluate", help="evaluate an original and optional compressed model")
    evaluate_parser.add_argument("--model", choices=("poseidon", "vicon", "mpp"), required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path)
    evaluate_parser.add_argument("--data-root", type=Path)
    evaluate_parser.add_argument("--compressed-model", type=Path)
    evaluate_parser.add_argument("--output", type=Path)
    evaluate_parser.add_argument("--source-root", type=Path)
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--batch-size", type=int, help="Evaluation batch size (MPP: 1; VICON: 128).")
    evaluate_parser.add_argument(
        "--max-batches",
        type=int,
        help="Cap evaluation batches per dataset/file (MPP: 64; VICON: all).",
    )
    evaluate_parser.add_argument("--dry-run", action="store_true")
    evaluate_parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Model-specific evaluator options after '--'.",
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
            / args.rank_selector
            / str(args.rank_ratio)
            / args.dataset
            / poseidon_calibration_tag(
                args.calibration_size,
                args.final_time,
                calibration_seed=args.calibration_seed,
            )
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_DIR / "artifacts" / args.model / args.preset / args.rank_selector / timestamp


def default_checkpoint(model: str) -> Path:
    """Match the case-sensitive paths produced by ``downloads/model_download.py``."""
    defaults = {
        "poseidon": PROJECT_DIR / "models" / "poseidon",
        "vicon": PROJECT_DIR / "models" / "VICON" / "model.pth",
        "mpp": PROJECT_DIR / "models" / "MPP" / "MPP_AViT_L",
    }
    return defaults[model]


def default_data_root(model: str) -> Path:
    defaults = {
        "poseidon": PROJECT_DIR / "poseidon" / "camlab-ethz" / "down_streams",
        "vicon": PROJECT_DIR / "data" / "VICON",
        "mpp": PROJECT_DIR / "data" / "MPP",
    }
    return defaults[model]


def default_evaluation_output(args: argparse.Namespace) -> Path:
    if args.model == "poseidon":
        # Preserve the established direct evaluator convention: a compressed
        # run writes metrics at its run root; an original checkpoint uses its
        # adjacent evaluator directory.
        source = (args.compressed_model or args.checkpoint or default_checkpoint("poseidon")).expanduser()
        if source.parent.name == "compressed":
            return source.parent.parent / "metrics.json"
        return source.parent / "evaluation" / "poseidon" / "metrics.json"
    if args.compressed_model is not None:
        compressed = args.compressed_model.expanduser()
        if compressed.parent.name == "compressed":
            return compressed.parent.parent / "evaluation" / "metrics.json"
        return compressed.parent / "evaluation" / "metrics.json"
    return PROJECT_DIR / "artifacts" / args.model / "evaluation" / "metrics.json"


def _validate_backend_extra(model: str, extra: tuple[str, ...], *, evaluation: bool) -> None:
    """Prevent post-``--`` arguments from overriding the public contract."""
    reserved = {
        "poseidon": {
            "--repo-root", "--poseidon-root", "--model_path", "--data_path", "--save_path",
            "--rank-root", "--cache-root", "--DEV", "--rank_ratio", "--rank-selector", "--traj_num", "--alpha", "--overwrite_cache",
        },
        "vicon": {
            "--vicon-root", "--model-path", "--data-path", "--save-path", "--profile-path",
            "--fisher-path", "--rank-path", "--device", "--rank-ratio", "--calibration-size",
            "--calibration-seed", "--rank-selector", "--alpha", "--overwrite",
        },
        "mpp": {
            "--mpp-root", "--model-path", "--data-path", "--save-path", "--profile-path",
            "--fisher-path", "--rank-path", "--device", "--rank-ratio", "--calibration-size",
            "--calibration-seed", "--rank-selector", "--alpha", "--overwrite",
        },
    }
    if evaluation:
        reserved = {
            "poseidon": {"--poseidon-root", "--data-root", "--checkpoint", "--model", "--output", "--device", "--batch-size"},
            "vicon": {"--vicon-root", "--model-path", "--data-path", "--compressed-path", "--output-path", "--device", "--batch-size"},
            "mpp": {"--mpp-root", "--model-path", "--data-path", "--compressed-path", "--output-path", "--device", "--batch-size"},
        }
    conflicts = sorted({token.split("=", 1)[0] for token in extra if token.split("=", 1)[0] in reserved[model]})
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(f"Pass {joined} through SAFE-SVD's public CLI options, not after '--'.")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "evaluate":
        extra = tuple(args.extra[1:] if args.extra[:1] == ["--"] else args.extra)
        _validate_backend_extra(args.model, extra, evaluation=True)
        config = SafeSVDEvaluationConfig(
            model=args.model,
            checkpoint=args.checkpoint or default_checkpoint(args.model),
            data_root=args.data_root or default_data_root(args.model),
            compressed_model=args.compressed_model,
            output_path=args.output or default_evaluation_output(args),
            source_root=args.source_root,
            device=args.device,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            extra=extra,
        )
        result = evaluate(config, dry_run=args.dry_run)
        print(f"SAFE-SVD evaluation {config.model}")
        print(f"Output: {result.output_path}")
        print(f"Backend command: {shlex.join(result.backend_command)}")
        if result.returncode not in (None, 0):
            print(f"ERROR: evaluator exited before completion (exit code {result.returncode}).", file=sys.stderr)
            return result.returncode
        return 0
    if args.command != "compress":
        raise AssertionError(f"Unhandled command: {args.command}")
    if args.rank_selector is None:
        args.rank_selector = "safe" if args.model == "poseidon" else "legacy"
    extra = tuple(args.extra[1:] if args.extra[:1] == ["--"] else args.extra)
    _validate_backend_extra(args.model, extra, evaluation=False)
    calibration_seed = args.calibration_seed
    if calibration_seed is None:
        calibration_seed = {"poseidon": -1, "vicon": 2026, "mpp": 42}[args.model]
    args.calibration_seed = calibration_seed
    if args.model == "poseidon" and calibration_seed < -1:
        raise ValueError("Poseidon --calibration-seed must be -1 or a non-negative integer.")
    if args.model != "poseidon" and (args.dataset != "fluids.incompressible.PiecewiseConstants" or args.final_time != 20):
        raise ValueError("--dataset and --final-time are currently Poseidon-only options.")
    if args.model == "poseidon":
        reserved = {"--dataset", "--final_time", "--random_calibration", "--calibration_seed", "--traj_num", "--rank_ratio"}
        conflicting = sorted({token.split("=", 1)[0] for token in extra if token.split("=", 1)[0] in reserved})
        if conflicting:
            raise ValueError(f"Pass {', '.join(conflicting)} through the SAFE-SVD CLI options, not after '--'.")
    poseidon_extra = (
        "--dataset", args.dataset,
        "--final_time", str(args.final_time),
        "--calibration_seed", str(calibration_seed),
    ) if args.model == "poseidon" else ()
    config = SafeSVDConfig(
        model=args.model,
        checkpoint=args.checkpoint or default_checkpoint(args.model),
        data_root=args.data_root or default_data_root(args.model),
        output_dir=args.output_dir or default_output_dir(args),
        rank_ratio=args.rank_ratio,
        calibration_size=args.calibration_size,
        calibration_seed=calibration_seed,
        rank_selector=args.rank_selector,
        source_root=args.source_root,
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
