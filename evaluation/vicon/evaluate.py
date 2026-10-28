#!/usr/bin/env python3
"""Evaluate original and compressed VICON checkpoints with the research protocol.

Defaults match physic_model_comp/vicon/evaluation/eval_comp.py: batch size
64, all test batches, and advancing loss/physics passes over shared datasets.
Model loading and metric computation live in evaluation/vicon/.
The compression package and research compression repository are not required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support `python evaluation/vicon/evaluate.py` without installing the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vicon-root", type=Path, default=PROJECT_ROOT / "VICON",
                        help="VICON checkout containing src/ and configs/ (default: bundled VICON).")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Original VICON checkpoint.")
    parser.add_argument("--model", type=Path, help="Compressed checkpoint; omit for origin only.")
    parser.add_argument("--output", type=Path, help="Metrics JSON path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, help="Optional cap per dataset, applied to each pass.")
    parser.add_argument("--config-dir", type=Path, help="Config directory, relative to --vicon-root if not absolute.")
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--config-override", action="append", default=[], help="Hydra override; may be repeated.")
    parser.add_argument("--protocol", choices=("research", "paired"), default="research",
                        help="research matches the original advancing RNG stream; paired replays samples.")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or (args.max_batches is not None and args.max_batches <= 0):
        parser.error("--batch-size and --max-batches must be positive")
    return args


def default_output_path(checkpoint: Path, compressed_model: Path | None) -> Path:
    source = compressed_model or checkpoint
    if source.parent.name == "compressed":
        return source.parent.parent / "metrics.json"
    return source.parent / "evaluation" / "vicon" / "metrics.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from evaluation.vicon.runtime import ViconEvaluationOptions, evaluate_vicon

    root = args.vicon_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    model = args.model.expanduser().resolve() if args.model else None
    config_dir = args.config_dir.expanduser() if args.config_dir else None
    if config_dir is not None and not config_dir.is_absolute():
        config_dir = root / config_dir
    output = args.output.expanduser().resolve() if args.output else default_output_path(checkpoint, model)
    evaluate_vicon(ViconEvaluationOptions(
        vicon_root=root, model_path=checkpoint, data_path=args.data_root.expanduser().resolve(),
        compressed_path=model, output_path=output, device=args.device,
        config_dir=config_dir.resolve() if config_dir else None, config_name=args.config_name,
        config_overrides=tuple(args.config_override), batch_size=args.batch_size,
        max_batches=args.max_batches, protocol=args.protocol,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
