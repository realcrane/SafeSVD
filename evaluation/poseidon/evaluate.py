#!/usr/bin/env python3
"""Poseidon physical evaluation for an original checkpoint and SAFE-SVD model.

The control flow mirrors ``poseidon_analysis/physical_loss.py``:

1. build Poseidon's test dataset;
2. evaluate the original model first;
3. optionally evaluate the compressed model on the same data; and
4. write the physical metrics and relative changes as JSON.

Unlike the research script, this entry point has no compression sweep,
performance profiling, plotting, or CSV cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from physics import (
    evaluate_and_result,
    get_dataset_metrics,
    get_metric_increases,
)


class FidelityDataset(Dataset):
    """Replace ground-truth labels with original predictions (``--eval-obj oncomp``)."""

    def __init__(self, original_ds: Dataset, origin_preds: Any) -> None:
        self.ds = original_ds
        self.origin_preds = torch.as_tensor(origin_preds)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.ds[index]
        item["labels"] = self.origin_preds[index]
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.ds, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poseidon-root",
        type=Path,
        required=True,
        help="Poseidon checkout containing scOT/.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Poseidon downstream-data directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Original fine-tuned checkpoint directory.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="SAFE-SVD compressed_model_origin.pt; omit for origin only.",
    )
    parser.add_argument("--dataset", default="fluids.incompressible.PiecewiseConstants")
    parser.add_argument("--num-trajectories", type=int, default=64)
    parser.add_argument("--final-time", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Poseidon physical_loss.py default.",
    )
    parser.add_argument(
        "--physics-batch-size",
        type=int,
        default=10,
        help="Poseidon physical_loss1.py default.",
    )
    parser.add_argument("--eval-obj", choices=("ongt", "oncomp"), default="ongt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, help="Metrics JSON path.")
    return parser.parse_args()


def load_saved_model(path: Path) -> torch.nn.Module:
    """Load a SAFE-SVD serialized module without changing the checkpoint config."""
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(path, map_location="cpu")
    if isinstance(saved, torch.nn.Module):
        return saved
    if isinstance(saved, dict) and isinstance(saved.get("model"), torch.nn.Module):
        return saved["model"]
    raise TypeError(f"{path} does not contain a saved torch.nn.Module.")


def build_trainer(
    get_trainer: Any,
    checkpoint: Path,
    test_ds: Dataset,
    batch_size: int,
    device: str,
    compressed_model: Path | None = None,
) -> Any:
    """Create a standard Poseidon inference trainer and optionally replace its model."""
    trainer = get_trainer(
        model_path=str(checkpoint),
        batch_size=batch_size,
        dataset=test_ds,
        workers=1,
    )
    trainer.args.dataloader_num_workers = 0
    trainer.args.remove_unused_columns = False

    if compressed_model is not None:
        model = load_saved_model(compressed_model)
        if not hasattr(model, "config"):
            model.config = trainer.model.config
        trainer.model = model

    trainer.model.to(device).eval()
    return trainer


def build_test_dataset(get_dataset: Any, args: argparse.Namespace) -> Dataset:
    """Use the exact dataset defaults of ``poseidon_analysis/physical_loss.py``."""
    time_kwargs = {}
    if ".time" not in args.dataset:
        time_kwargs = {
            "fix_input_to_time_step": None,
            "time_step_size": 1,
            "max_num_time_steps": args.final_time,
        }
    return get_dataset(
        dataset=args.dataset,
        which="test",
        num_trajectories=args.num_trajectories,
        data_path=str(args.data_root),
        **time_kwargs,
    )


def default_output_path(checkpoint: Path, compressed_model: Path | None) -> Path:
    source = compressed_model or checkpoint
    if source.is_file() and source.parent.name == "compressed":
        return source.parent.parent / "evaluation" / "poseidon" / "metrics.json"
    return source.parent / "evaluation" / "poseidon" / "metrics.json"


def print_summary(title: str, metrics: dict[str, Any], dataset_name: str) -> None:
    """Print only the standard physical metrics relevant to this PDE."""
    print(f"\n{title}")
    for key in get_dataset_metrics(dataset_name):
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    args.poseidon_root = args.poseidon_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    model_path = args.model.expanduser().resolve() if args.model else None

    required_paths = {
        "Poseidon source": args.poseidon_root / "scOT",
        "data root": args.data_root,
        "checkpoint config": args.checkpoint / "config.json",
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if model_path is not None and not model_path.is_file():
        raise FileNotFoundError(f"compressed model not found: {model_path}")
    return args.poseidon_root, args.checkpoint, model_path


def main() -> int:
    args = parse_args()
    if min(args.num_trajectories, args.final_time, args.batch_size, args.physics_batch_size) <= 0:
        raise ValueError("All trajectory, time, and batch-size limits must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu.")

    poseidon_root, checkpoint, compressed_model = validate_paths(args)
    sys.path.insert(0, str(poseidon_root))
    from scOT.inference import get_trainer, rollout
    from scOT.problems.base import get_dataset

    # 1. Setup data (and optional fidelity labels), as in physical_loss.py.
    test_ds = build_test_dataset(get_dataset, args)
    origin_trainer = build_trainer(
        get_trainer,
        checkpoint,
        test_ds,
        args.batch_size,
        args.device,
    )
    if args.eval_obj == "oncomp":
        origin_preds, _, _ = rollout(origin_trainer, test_ds)
        test_ds = FidelityDataset(test_ds, origin_preds)
        origin_trainer = build_trainer(
            get_trainer,
            checkpoint,
            test_ds,
            args.batch_size,
            args.device,
        )

    # 2. Origin model is always the comparison baseline.
    print("Evaluating origin model...")
    origin_metrics = evaluate_and_result(
        origin_trainer,
        test_ds,
        args.dataset,
        rollout,
        args.physics_batch_size,
    )
    print_summary("Origin physical metrics", origin_metrics, args.dataset)

    payload: dict[str, Any] = {
        "dataset": args.dataset,
        "eval_obj": args.eval_obj,
        "num_trajectories": args.num_trajectories,
        "final_time": args.final_time,
        "batch_size": args.batch_size,
        "physics_batch_size": args.physics_batch_size,
        "origin": {"model": str(checkpoint), "metrics": origin_metrics},
    }

    # 3. SAFE-SVD model uses the same trainer configuration and test dataset.
    if compressed_model is not None:
        print("Evaluating SAFE-SVD compressed model...")
        compressed_trainer = build_trainer(
            get_trainer,
            checkpoint,
            test_ds,
            args.batch_size,
            args.device,
            compressed_model,
        )
        compressed_metrics = evaluate_and_result(
            compressed_trainer,
            test_ds,
            args.dataset,
            rollout,
            args.physics_batch_size,
        )
        print_summary("Compressed physical metrics", compressed_metrics, args.dataset)
        payload["compressed"] = {
            "model": str(compressed_model),
            "metrics": compressed_metrics,
            "increase": get_metric_increases(compressed_metrics, origin_metrics),
        }

    output = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(checkpoint, compressed_model)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved physical metrics to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
