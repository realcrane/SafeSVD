"""SAFE-SVD runtime adapter for Multiple Physics Pretraining (MPP-AViT)."""

from __future__ import annotations

import importlib
import json
import math
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from safesvd.adapters.common_runtime import (
    CategoryAwareStatistics,
    TaggedBatch,
    cache_tagged_batches,
    collect_fisher_whiteners,
    collect_input_whiteners,
    get_module_by_name,
    load_compressed_state,
    model_parameter_report,
    resolve_device,
    save_portable_checkpoint,
    set_module_by_name,
    write_json,
)
from safesvd.adapters.mpp_physics import SOBOLEV_ORDERS, average_metrics, physics_metrics
from safesvd.core import CompressionCallbacks, allocate_greedy_ranks, compress_clean_input, compress_sequentially


@dataclass(frozen=True)
class MPPOptions:
    mpp_root: Path
    model_path: Path
    data_path: Path
    save_path: Path
    profile_path: Path
    fisher_path: Path
    rank_path: Path
    rank_ratio: float
    calibration_size: int
    calibration_seed: int
    alpha: float
    device: str
    yaml_config: Path | None = None
    config_name: str = "basic_config"
    batch_size: int | None = None
    use_fisher: bool = True
    fisher_loss: str = "l1"
    balance_ratio: float = 10.0
    decay_alpha: float = 1.0
    trace_balance: bool = True
    overwrite: bool = False


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
    batch_size: int | None = None
    max_batches: int | None = None


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


def run_mpp(options: MPPOptions) -> Path:
    """Run the MPP data/model route through the shared SAFE-SVD core."""
    _validate_options(options)
    bundle = load_mpp_bundle(
        options.mpp_root, options.model_path, options.data_path, options.device,
        yaml_config=options.yaml_config, config_name=options.config_name, batch_size=options.batch_size,
    )
    batches = _mpp_batch_factory(
        bundle, bundle.train_dataset, labels_from=bundle.train_dataset,
        calibration_size=options.calibration_size, seed=options.calibration_seed,
    )
    profile = _load_or_collect(
        options.profile_path,
        options.overwrite,
        lambda: collect_input_whiteners(
            bundle.model, batches, bundle_forward(bundle), device=bundle.device,
            trace_balance=options.trace_balance, category_average_without_trace=False,
        ),
    )
    fisher = None
    if options.use_fisher:
        fisher = _load_or_collect(
            options.fisher_path,
            options.overwrite,
            lambda: collect_fisher_whiteners(
                bundle.model,
                batches,
                _mpp_fisher_loss(bundle, options.fisher_loss, options.balance_ratio, options.decay_alpha),
                device=bundle.device,
                trace_balance=options.trace_balance,
                category_average_without_trace=False,
            ),
        )
    ranks = _load_or_select_ranks(options, bundle.model, fisher, profile)

    options.save_path.mkdir(parents=True, exist_ok=True)
    in_progress = options.save_path / "compressing_model.pt"
    initial_student: nn.Module | None = None
    compressed_layers: set[str] | None = None
    if options.alpha != 0 and in_progress.is_file() and not options.overwrite:
        partial = _torch_load(in_progress)
        if isinstance(partial, Mapping) and isinstance(partial.get("model"), nn.Module):
            initial_student = partial["model"]
            compressed_layers = set(partial.get("compressed_layers", ()))
            print(f"Resuming MPP compression after {len(compressed_layers)} layers", flush=True)

    total_layers = sum(isinstance(module, nn.Linear) for module in bundle.model.modules())
    checkpoint_every = max(1, total_layers // 10)

    def checkpoint(student: nn.Module, done: set[str]) -> None:
        torch.save({"model": student.cpu(), "compressed_layers": sorted(done)}, in_progress)
        student.to(bundle.device)

    compression_batches = cache_tagged_batches(batches) if options.alpha != 0 else batches
    callbacks = CompressionCallbacks(
        capture_inputs=lambda _model, _name: (),
        get_module=get_module_by_name,
        set_module=set_module_by_name,
        checkpoint=checkpoint,
        progress=_progress("MPP"),
        capture_statistics=CategoryAwareStatistics(
            compression_batches, bundle_forward(bundle), device=bundle.device,
            trace_balance=options.trace_balance, category_average_without_trace=False,
        ),
    )
    if options.alpha == 0:
        compressed = compress_clean_input(
            bundle.model, ranks, profile, callbacks, device=bundle.device, left_whiteners=fisher,
        )
    else:
        compressed = compress_sequentially(
            bundle.model,
            ranks,
            callbacks,
            device=bundle.device,
            alpha=options.alpha,
            left_whiteners=fisher,
            checkpoint_every=checkpoint_every,
            initial_student=initial_student,
            compressed_layers=compressed_layers,
        )
    output = save_portable_checkpoint(
        options.save_path / "compressed_model.pt",
        compressed,
        model_name="mpp",
        ranks=ranks,
        metadata={
            "source_root": str(options.mpp_root),
            "yaml_config": str(_yaml_config(options)),
            "config_name": options.config_name,
            "checkpoint": str(options.model_path),
            "data_root": str(options.data_path),
            "alpha": options.alpha,
        },
    )
    if in_progress.exists():
        in_progress.unlink()
    print(f"Compressed MPP model saved to {output}")
    return output


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
    payload: dict[str, Any] = {"model": "mpp", "origin": {"model": str(options.model_path), "metrics": origin}}
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
        payload["compressed"] = {
            "model": str(options.compressed_path),
            "metrics": compressed_metrics,
            "increase": _nested_increases(compressed_metrics, origin),
        }
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
    model = avit_module.build_avit(params).to(target)
    model.load_state_dict(_checkpoint_state_dict(checkpoint))
    model.eval()
    _unused_loader, train_dataset, _sampler = datasets_module.get_data_loader(
        params,
        params.train_data_paths,
        distributed=False,
        split="train",
        rank=0,
        train_offset=getattr(params, "embedding_offset", 0),
    )
    _unused_loader, valid_dataset, _sampler = datasets_module.get_data_loader(
        params, params.valid_data_paths, distributed=False, split="val", rank=0,
    )
    return MPPBundle(package_root, params, model, train_dataset, valid_dataset, target)


def bundle_forward(bundle: MPPBundle) -> Callable[[nn.Module, MPPPayload], torch.Tensor]:
    def forward(model: nn.Module, payload: MPPPayload) -> torch.Tensor:
        output, _target, _inputs = bundle.forward(model, payload)
        return output

    return forward


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


def _mpp_fisher_loss(
    bundle: MPPBundle,
    loss_name: str,
    balance_ratio: float,
    decay_alpha: float,
) -> Callable[[nn.Module, TaggedBatch], torch.Tensor]:
    loss_fn = {"l1": functional.l1_loss, "mse": functional.mse_loss}.get(loss_name)
    if loss_fn is None:
        raise ValueError(f"Unsupported MPP Fisher loss {loss_name!r}")

    def loss(model: nn.Module, tagged: TaggedBatch) -> torch.Tensor:
        output, target, inputs = bundle.forward(model, tagged.payload)
        spatial_dims = tuple(range(2, output.ndim))
        residual = output - target
        target_norm = 1e-7 + target.square().mean(spatial_dims, keepdim=True)
        base = (residual.square().mean(spatial_dims, keepdim=True) / target_norm).mean()
        order = SOBOLEV_ORDERS.get(tagged.category, 1)
        return base + _mpp_sobolev_loss(
            tagged.category,
            output,
            target,
            inputs[-1],
            loss_fn,
            order=order,
            balance_ratio=balance_ratio,
            decay_alpha=decay_alpha,
            base_loss=float(base.detach()),
        )

    return loss


def _mpp_sobolev_loss(
    category: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    previous: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor],
    *,
    order: int,
    balance_ratio: float,
    decay_alpha: float,
    base_loss: float,
) -> torch.Tensor:
    """The derivative regularisation used solely to collect MPP Fisher data."""
    total = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    predictions = [prediction]
    targets = [target]
    for current_order in range(1, order + 1):
        next_predictions: list[torch.Tensor] = []
        next_targets: list[torch.Tensor] = []
        derivative = torch.zeros_like(total)
        for current_prediction, current_target in zip(predictions, targets):
            pred_x = current_prediction[..., 1:] - current_prediction[..., :-1]
            target_x = current_target[..., 1:] - current_target[..., :-1]
            pred_y = current_prediction[..., 1:, :] - current_prediction[..., :-1, :]
            target_y = current_target[..., 1:, :] - current_target[..., :-1, :]
            derivative = derivative + loss_fn(pred_x, target_x) + loss_fn(pred_y, target_y)
            if current_order < order:
                next_predictions.extend((pred_x, pred_y))
                next_targets.extend((target_x, target_y))
        magnitude = float(derivative.detach())
        weight = min((base_loss / magnitude) * balance_ratio, 1000.0) if magnitude > 1e-20 else 0.0
        total = total + weight * decay_alpha ** (current_order - 1) * derivative
        predictions, targets = next_predictions, next_targets
    if category == "incompNS":
        pred_x = prediction[:, 0, :, 1:] - prediction[:, 0, :, :-1]
        pred_y = prediction[:, 1, 1:, :] - prediction[:, 1, :-1, :]
        target_x = target[:, 0, :, 1:] - target[:, 0, :, :-1]
        target_y = target[:, 1, 1:, :] - target[:, 1, :-1, :]
        height = min(pred_x.shape[-2], pred_y.shape[-2], target_x.shape[-2], target_y.shape[-2])
        width = min(pred_x.shape[-1], pred_y.shape[-1], target_x.shape[-1], target_y.shape[-1])
        pred_divergence = pred_x[..., :height, :width] + pred_y[..., :height, :width]
        target_divergence = target_x[..., :height, :width] + target_y[..., :height, :width]
        divergence = loss_fn(pred_divergence, target_divergence)
        magnitude = float(divergence.detach())
        weight = min((base_loss / magnitude) * balance_ratio, 1000.0) if magnitude > 1e-20 else 0.0
        total = total + weight * divergence
    return total


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
            title: {"category": categories[title], **average_metrics(entries)}
            for title, entries in values.items()
        },
    }


def _rebase_data_paths(params: Any, data_root: Path) -> None:
    def rebase(paths: Iterable[Iterable[Any]]) -> list[list[Any]]:
        result: list[list[Any]] = []
        for path_spec in paths:
            values = list(path_spec)
            if values and isinstance(values[0], str) and not Path(values[0]).is_absolute():
                values[0] = str(data_root / values[0])
            result.append(values)
        return result

    params.train_data_paths = rebase(params.train_data_paths)
    params.valid_data_paths = rebase(params.valid_data_paths)


def _checkpoint_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"MPP checkpoint {path} is not a mapping")
    state = payload.get("model_state", payload.get("state_dict", payload))
    if not isinstance(state, Mapping):
        raise TypeError(f"MPP checkpoint {path} does not contain a state dict")
    return {key.removeprefix("module."): value for key, value in state.items() if torch.is_tensor(value)}


def _load_or_collect(path: Path, overwrite: bool, collector: Callable[[], Any]) -> Any:
    if path.is_file() and not overwrite:
        print(f"Loading cache: {path}", flush=True)
        return _torch_load(path)
    result = collector()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


def _load_or_select_ranks(
    options: MPPOptions,
    model: nn.Module,
    fisher: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None,
    profile: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, int | None]:
    if options.rank_path.is_file() and not options.overwrite:
        return json.loads(options.rank_path.read_text())
    ranks = allocate_greedy_ranks(model, fisher, profile, options.rank_ratio, options.device)
    options.rank_path.parent.mkdir(parents=True, exist_ok=True)
    options.rank_path.write_text(json.dumps(ranks, indent=2) + "\n")
    return ranks


def _validate_options(options: MPPOptions) -> None:
    if options.calibration_size <= 0:
        raise ValueError("MPP calibration size must be positive")
    if not 0 < options.rank_ratio <= 1:
        raise ValueError("rank_ratio must be in (0, 1]")
    if not 0 <= options.alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if not options.model_path.is_file():
        raise FileNotFoundError(f"MPP checkpoint not found: {options.model_path}")
    if not options.data_path.exists():
        raise FileNotFoundError(f"MPP data root not found: {options.data_path}")
    find_mpp_package(options.mpp_root)


def _yaml_config(options: MPPOptions) -> Path:
    return options.yaml_config or find_mpp_package(options.mpp_root) / "config" / "mpp_avit_L_config.yaml"


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(path, map_location="cpu")


def _progress(prefix: str) -> Callable[[int, int, str], None]:
    def progress(index: int, total: int, name: str) -> None:
        print(f"[{prefix}] {index}/{total}: {name}", flush=True)

    return progress


def _nested_increases(value: Any, baseline: Any) -> Any:
    if isinstance(value, Mapping) and isinstance(baseline, Mapping):
        return {key: _nested_increases(item, baseline[key]) for key, item in value.items() if key in baseline}
    if isinstance(value, (int, float)) and isinstance(baseline, (int, float)) and baseline != 0:
        return round(100.0 * (float(value) - float(baseline)) / abs(float(baseline)), 6)
    return None
