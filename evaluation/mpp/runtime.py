"""MPP loading, fixed per-file validation sampling and physical evaluation."""
from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.reporting import build_dataset_results

from .checkpoints import TaggedBatch, load_compressed_state, model_parameter_report, resolve_device, write_json
from .physics import average_metrics, physics_metrics


@dataclass(frozen=True)
class MPPEvaluationOptions:
    mpp_root: Path
    model_path: Path
    data_path: Path
    compressed_path: Path | None
    output_path: Path
    device: str
    yaml_config: Path | None = None
    config_name: str = "basic_config"
    batch_size: int | None = 1
    max_batches: int | None = 64


@dataclass(frozen=True)
class MPPPayload:
    history: torch.Tensor
    boundary_conditions: torch.Tensor
    target: torch.Tensor
    label_indices: tuple[int, ...]
    title: str


@dataclass
class MPPBundle:
    package_root: Path
    params: Any
    model: nn.Module
    train_dataset: Any
    valid_dataset: Any
    device: torch.device

    def forward(self, model: nn.Module, payload: MPPPayload) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history = payload.history.to(self.device)
        boundary_conditions = payload.boundary_conditions.to(self.device)
        target = payload.target.to(self.device)
        labels = torch.tensor(payload.label_indices, device=self.device).unsqueeze(0).expand(target.shape[0], -1)
        # Input follows the upstream AViT convention: T,B,C,H,W.
        inputs = history.permute(1, 0, 2, 3, 4)
        return model(inputs, labels, boundary_conditions), target, inputs


def evaluate_mpp(options: MPPEvaluationOptions) -> Path:
    """Evaluate original and optional SAFE-SVD MPP model on the same per-file subsets."""
    bundle = load_mpp_bundle(
        options.mpp_root, options.model_path, options.data_path, options.device,
        yaml_config=options.yaml_config, config_name=options.config_name, batch_size=options.batch_size,
    )
    batches = _mpp_batch_factory(
        bundle,
        bundle.valid_dataset,
        labels_from=bundle.train_dataset,
        calibration_size=None,
        seed=0,
        max_batches=options.max_batches,
    )
    origin = _evaluate_mpp_model(bundle, bundle.model, batches)
    print_summary("Origin", origin)
    metadata: dict[str, Any] = {
        "model": "mpp", "protocol": "research_per_file", "split": "val", "sampling_seed": 0,
        "eval_obj": "ongt",
        "batch_size": int(bundle.params.batch_size), "max_batches_per_file": options.max_batches,
    }
    compressed_metrics = None
    if options.compressed_path is not None:
        # AViT-L is large enough that keeping its baseline weights on GPU while
        # constructing the compressed topology can exceed ordinary evaluator
        # hardware.  The batch factory stores CPU payloads only.
        bundle.model.cpu()
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()
        compressed_bundle = load_mpp_bundle(
            options.mpp_root, options.model_path, options.data_path, options.device,
            yaml_config=options.yaml_config, config_name=options.config_name, batch_size=options.batch_size,
        )
        compressed = load_compressed_state(compressed_bundle.model, options.compressed_path)
        compressed_metrics = _evaluate_mpp_model(compressed_bundle, compressed, batches)
        print_summary("Compressed", compressed_metrics)
    dataset_metadata = {
        name: {"category": values["category"], "batches": {"origin": values["batches"]}}
        for name, values in origin["datasets"].items()
    }
    if compressed_metrics is not None:
        for name, values in compressed_metrics["datasets"].items():
            dataset_metadata[name]["batches"]["compressed"] = values["batches"]
    payload = build_dataset_results(
        options.model_path, _dataset_metrics(origin),
        compressed_model=options.compressed_path,
        compressed_metrics=_dataset_metrics(compressed_metrics) if compressed_metrics is not None else None,
        metadata=metadata, dataset_metadata=dataset_metadata,
    )
    write_json(options.output_path, payload)
    print(f"Saved MPP metrics to {options.output_path}")
    return options.output_path


def find_mpp_package(root: Path) -> Path:
    """Find the MPP package and fail explicitly when its gitignored model code is absent."""
    root = root.expanduser().resolve()
    for candidate in (root / "multiple_physics_pretraining", root):
        if (candidate / "data_utils").is_dir() and (candidate / "config").is_dir():
            missing = [
                candidate / "models",
                candidate / "models" / "avit.py",
            ]
            if any(not path.exists() for path in missing):
                raise FileNotFoundError(
                    "MPP source is incomplete: expected multiple_physics_pretraining/models/avit.py and its "
                    f"model dependencies below {candidate}. The research snapshot excludes this directory; "
                    "install the complete upstream MPP checkout or pass --source-root."
                )
            return candidate
    raise FileNotFoundError(
        f"MPP source was not found below {root}; expected multiple_physics_pretraining/config and data_utils."
    )


def load_mpp_bundle(
    root: Path,
    checkpoint: Path,
    data_root: Path,
    device: str | torch.device,
    *,
    yaml_config: Path | None = None,
    config_name: str = "basic_config",
    batch_size: int | None = None,
    include_validation: bool = True,
) -> MPPBundle:
    """Build AViT and its train/validation datasets using the original YAML contract."""
    package_root = find_mpp_package(root)
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    datasets_module = importlib.import_module("multiple_physics_pretraining.data_utils.datasets")
    avit_module = importlib.import_module("multiple_physics_pretraining.models.avit")
    yparams_module = importlib.import_module("multiple_physics_pretraining.utils.YParams")
    config = yaml_config or package_root / "config" / "mpp_avit_L_config.yaml"
    params = yparams_module.YParams(str(config.expanduser().resolve()), config_name)
    _rebase_data_paths(params, data_root)
    params.use_ddp = False
    params.log_to_wandb = False
    if batch_size is not None:
        params["batch_size"] = batch_size
    target = resolve_device(device)
    _unused_loader, train_dataset, _sampler = datasets_module.get_data_loader(
        params,
        params.train_data_paths,
        distributed=False,
        split="train",
        rank=0,
        train_offset=getattr(params, "embedding_offset", 0),
    )
    valid_dataset = None
    if include_validation:
        _unused_loader, valid_dataset, _sampler = datasets_module.get_data_loader(
            params, params.valid_data_paths, distributed=False, split="val", rank=0,
        )
    model = avit_module.build_avit(params).to(target)
    model.load_state_dict(_checkpoint_state_dict(checkpoint))
    model.eval()
    return MPPBundle(package_root, params, model, train_dataset, valid_dataset, target)


def _mpp_batch_factory(
    bundle: MPPBundle,
    dataset: Any,
    *,
    labels_from: Any,
    calibration_size: int | None,
    seed: int,
    max_batches: int | None = None,
) -> Callable[[], Iterable[TaggedBatch]]:
    batch_size = int(bundle.params.batch_size)
    workers = int(bundle.params.num_data_workers)
    per_subset_limit = max_batches
    if per_subset_limit is None and calibration_size is not None:
        per_subset_limit = math.ceil(calibration_size / batch_size)

    def batches() -> Iterable[TaggedBatch]:
        for subset_group in dataset.sub_dsets:
            for subset in subset_group.get_per_file_dsets():
                category = str(subset.get_name())
                validation_fields = getattr(dataset, "subset_dict", {}).get(category, ())
                labels = tuple(labels_from.subset_dict.get(category, [-1] * len(validation_fields)))
                loader = DataLoader(
                    subset,
                    batch_size=batch_size,
                    num_workers=workers,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(seed),
                    drop_last=True,
                )
                for index, data in enumerate(loader):
                    if per_subset_limit is not None and index >= per_subset_limit:
                        break
                    history, boundary_conditions, target = data
                    yield TaggedBatch(
                        category,
                        MPPPayload(history, boundary_conditions, target, labels, str(getattr(subset, "title", category))),
                    )

    return batches


def _evaluate_mpp_model(
    bundle: MPPBundle,
    model: nn.Module,
    batches: Callable[[], Iterable[TaggedBatch]],
) -> dict[str, Any]:
    model = model.to(bundle.device).eval()
    values: dict[str, list[dict[str, float]]] = {}
    categories: dict[str, str] = {}
    with torch.inference_mode():
        for tagged in batches():
            output, target, inputs = bundle.forward(model, tagged.payload)
            spatial_dims = tuple(range(2, output.ndim))
            residual = output - target
            target_norm = 1e-7 + target.square().mean(spatial_dims, keepdim=True)
            relative_mse = (residual.square().mean(spatial_dims, keepdim=True) / target_norm).mean()
            entry = {"mse": float(residual.square().mean()), "rel_mse": float(relative_mse)}
            entry.update(physics_metrics(tagged.category, output, target, inputs[-1]))
            values.setdefault(tagged.payload.title, []).append(entry)
            categories[tagged.payload.title] = tagged.category
    return {
        **model_parameter_report(model),
        "datasets": {
            title: {"category": categories[title], "batches": len(entries), **average_metrics(entries)}
            for title, entries in values.items()
        },
    }


def _rebase_data_paths(params: Any, data_root: Path) -> None:
    def rebase(paths: Iterable[Iterable[Any]]) -> list[list[Any]]:
        result: list[list[Any]] = []
        for path_spec in paths:
            values = list(path_spec)
            if values and isinstance(values[0], str):
                path = Path(values[0]).expanduser()
                values[0] = str(path if path.is_absolute() else data_root / path)
            result.append(values)
        return result

    params.train_data_paths = rebase(params.train_data_paths)
    params.valid_data_paths = rebase(params.valid_data_paths)


def _checkpoint_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"MPP checkpoint {path} is not a mapping")
    state = payload.get("model_state", payload.get("model_state_dict", payload.get("state_dict", payload)))
    if not isinstance(state, Mapping):
        raise TypeError(f"MPP checkpoint {path} does not contain a state dict")
    return {key.removeprefix("module."): value for key, value in state.items() if torch.is_tensor(value)}


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(path, map_location="cpu")


def _dataset_metrics(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten each dataset report using Poseidon's loss and size field names."""
    return {
        name: {
            **{key: value for key, value in values.items() if key not in {"category", "batches", "mse", "rel_mse"}},
            "base_loss": values["mse"],
            "rel_base_loss": values["rel_mse"],
            "model_size": report["parameters"],
            "linear_size": report["linear_parameters"],
        }
        for name, values in report["datasets"].items()
    }


def print_summary(title: str, report: Mapping[str, Any]) -> None:
    """Print the per-dataset physical metrics using the Poseidon entry layout."""
    print(f"\n{title}")
    print(f"  parameters: {report['parameters']}")
    for name, metrics in report['datasets'].items():
        print(f"  {name}:")
        for key, value in metrics.items():
            if key not in {'category', 'batches'}:
                print(f"    {key}: {value}")
