#!/usr/bin/env python3
"""Evaluate original and compressed MPP checkpoints with the research protocol.

Defaults match physic_model_comp/mpp/eval/eval_all.py: batch size 1,
64 batches per validation file, shuffle seed 0, and drop_last=True.
Model loading and metric computation live in evaluation/mpp/.
The compression package and research compression repository are not required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support `python evaluation/mpp/evaluate.py` without installing the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.mpp.paths import default_mpp_package


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpp-root", type=Path, default=default_mpp_package(PROJECT_ROOT),
                        help="MPP checkout containing config/, data_utils/, and models/.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Original MPP checkpoint.")
    parser.add_argument("--model", type=Path, help="Compressed checkpoint; omit for origin only.")
    parser.add_argument("--output", type=Path, help="Metrics JSON path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=64, help="Maximum batches per validation file (default: 64).")
    parser.add_argument("--yaml-config", type=Path, help="YAML config, relative to --mpp-root if not absolute.")
    parser.add_argument("--config-name", default="basic_config")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.max_batches <= 0:
        parser.error("--batch-size and --max-batches must be positive")
    return args


def default_output_path(checkpoint: Path, compressed_model: Path | None) -> Path:
    source = compressed_model or checkpoint
    if source.parent.name == "compressed":
        return source.parent.parent / "metrics.json"
    return source.parent / "evaluation" / "mpp" / "metrics.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from evaluation.mpp.runtime import MPPEvaluationOptions, evaluate_mpp

    root = args.mpp_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    model = args.model.expanduser().resolve() if args.model else None
    yaml_config = args.yaml_config.expanduser() if args.yaml_config else None
    if yaml_config is not None and not yaml_config.is_absolute():
        yaml_config = root / yaml_config
    output = args.output.expanduser().resolve() if args.output else default_output_path(checkpoint, model)
    evaluate_mpp(MPPEvaluationOptions(
        mpp_root=root, model_path=checkpoint, data_path=args.data_root.expanduser().resolve(),
        compressed_path=model, output_path=output, device=args.device,
        yaml_config=yaml_config.resolve() if yaml_config else None, config_name=args.config_name,
        batch_size=args.batch_size, max_batches=args.max_batches,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
