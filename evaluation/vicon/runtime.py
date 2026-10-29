"""VICON loading, research/paired traversal and evaluation; no compression dependency."""
from __future__ import annotations

import copy
import importlib
import inspect
import os
import random
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from evaluation.reporting import build_dataset_results

from .checkpoints import TaggedBatch, load_compressed_state, model_parameter_report, resolve_device, write_json
from .attention import convert_attn_to_linear, disable_transformer_fast_path
from .physics import average_metrics, physics_metrics


@dataclass(frozen=True)
class ViconEvaluationOptions:
    vicon_root: Path
    model_path: Path
    data_path: Path
    compressed_path: Path | None
    output_path: Path
    device: str
    config_dir: Path | None = None
    config_name: str = "default"
    config_overrides: tuple[str, ...] = ()
    batch_size: int = 64
    max_batches: int | None = None
    protocol: str = "research"


@dataclass
class ViconBundle:
    root: Path
    config: Any
    trainer: Any
    all_datasets: Callable[..., Mapping[str, Any]]
    data_root: Path
    device: torch.device

    def prepare(self, payload: Any) -> tuple[Any, Any, Any]:
        _ignored, pairs, *_rest = payload
        pairs = self.trainer._move_to_device(pairs)
        data, mean, std = self.trainer._data_preprocess(pairs)
        return data, mean, std

    def forward(self, model: nn.Module, payload: Any) -> tuple[torch.Tensor, Any, Any, Any]:
        self.trainer.model = model
        data, mean, std = self.prepare(payload)
        output = self.trainer._model_forward(data)
        return output, data, mean, std

    def loss(self, model: nn.Module, payload: Any) -> Any:
        """Call the upstream trainer's evaluation-loss contract exactly."""
        self.trainer.model = model
        _ignored, pairs, *_rest = payload
        return self.trainer.get_loss(self.trainer._move_to_device(pairs))


def evaluate_vicon(options: ViconEvaluationOptions) -> Path:
    """Evaluate VICON with the research script's shared test-dataset RNG stream."""
    if options.protocol not in {"research", "paired"}:
        raise ValueError("VICON evaluation protocol must be research or paired")
    bundle = load_vicon_bundle(
        options.vicon_root, options.model_path, options.data_path, options.device,
        config_dir=options.config_dir, config_name=options.config_name, config_overrides=options.config_overrides,
        explicit_attention=False,
    )
    batch_factory = _vicon_batch_factory(
        bundle,
        split="test",
        batch_size=options.batch_size,
        calibration_size=None,
        seed=int(getattr(bundle.config, "test_seed", 0)),
        random_subset=False,
        max_batches=options.max_batches,
        replay=options.protocol == "paired",
    )
    origin = _evaluate_vicon_model(bundle, bundle.trainer.model, batch_factory)
    print_summary("Origin", origin)
    metadata: dict[str, Any] = {
        "model": "vicon",
        "eval_obj": "ongt",
        "split": "test",
        "protocol": options.protocol,
        "test_seed": int(getattr(bundle.config, "test_seed", 0)),
        "batch_size": options.batch_size,
        "max_batches": options.max_batches,
        "config_overrides": list(options.config_overrides),
    }
    compressed_metrics = None
    if options.compressed_path is not None:
        # Do not retain a second full ViCON model on GPU while rebuilding the
        # portable low-rank topology for the compressed evaluator.
        bundle.trainer.model.cpu()
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()
        compressed_bundle = load_vicon_bundle(
            options.vicon_root, options.model_path, options.data_path, options.device,
            config_dir=options.config_dir, config_name=options.config_name, config_overrides=options.config_overrides,
        )
        compressed = load_compressed_state(compressed_bundle.trainer.model, options.compressed_path)
        disable_transformer_fast_path(compressed)
        compressed_metrics = _evaluate_vicon_model(compressed_bundle, compressed, batch_factory)
        print_summary("Compressed", compressed_metrics)
    dataset_metadata = {
        name: {"sampling": {"origin": _dataset_sampling(origin, name)}}
        for name in origin["losses"]
    }
    if compressed_metrics is not None:
        for name in compressed_metrics["losses"]:
            dataset_metadata[name]["sampling"]["compressed"] = _dataset_sampling(compressed_metrics, name)
    payload = build_dataset_results(
        options.model_path, _dataset_metrics(origin),
        compressed_model=options.compressed_path,
        compressed_metrics=_dataset_metrics(compressed_metrics) if compressed_metrics is not None else None,
        metadata=metadata, dataset_metadata=dataset_metadata,
    )
    write_json(options.output_path, payload)
    print(f"Saved VICON metrics to {options.output_path}")
    return options.output_path


def load_vicon_bundle(
    root: Path,
    checkpoint: Path,
    data_root: Path,
    device: str | torch.device,
    *,
    config_dir: Path | None = None,
    config_name: str = "default",
    config_overrides: tuple[str, ...] = (),
    explicit_attention: bool = True,
) -> ViconBundle:
    """Load a VICON model/trainer without importing paper compression code."""
    package_root = find_vicon_package(root)
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    models = importlib.import_module("VICON.src.models")
    trainer_module = importlib.import_module("VICON.src.trainer")
    target = resolve_device(device)
    trainer_module.device = target
    dataset_module = importlib.import_module("VICON.src.dataset")
    config = _load_config(config_dir or package_root / "configs", config_name, config_overrides)
    _apply_data_root(config, data_root, config_overrides=config_overrides)
    model_type = str(getattr(config.model, "type", ""))
    constructor = getattr(models, "ICON_CROPPED" if model_type == "crop" else "ICON_UNCROPPED")
    model = constructor(config.model)
    trainer = trainer_module.Trainer(
        model, config.model, config.opt, config.loss,
        trainable_mode=getattr(config, "trainable_mode", None),
    )
    trainer.device = target
    state_dict = _checkpoint_state_dict(checkpoint)
    trainer.model.load_state_dict(state_dict)
    if explicit_attention:
        convert_attn_to_linear(trainer.model)
        disable_transformer_fast_path(trainer.model)
    trainer.model.to(target).eval()
    return ViconBundle(package_root, config, trainer, dataset_module.all_datasets, data_root, target)


def find_vicon_package(root: Path) -> Path:
    """Locate ``VICON/src`` for either a wrapper checkout or direct clone."""
    root = root.expanduser().resolve()
    for candidate in (root / "VICON", root):
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError(
        f"VICON source was not found below {root}; expected VICON/src and VICON/configs. "
        "Clone the complete upstream VICON checkout or pass --source-root."
    )


def _vicon_batch_factory(
    bundle: ViconBundle,
    *,
    split: str,
    batch_size: int,
    calibration_size: int | None,
    seed: int,
    random_subset: bool,
    max_batches: int | None = None,
    source_dataset_workers: int | None = None,
    replay: bool = False,
) -> Callable[[], Iterable[TaggedBatch]]:
    loader_workers = int(getattr(bundle.config, "dataset_workers", 0))
    dataset_workers = loader_workers if source_dataset_workers is None else source_dataset_workers
    dataset_seed = _vicon_dataset_seed(bundle, split, seed, random_subset)
    datasets = _all_datasets(bundle, dataset_workers, dataset_seed, split)
    randomizer = random.Random(seed)
    selected: dict[str, Any] = {}
    for category, dataset in datasets.items():
        current = dataset
        if calibration_size is not None and random_subset:
            try:
                count = min(calibration_size, len(dataset))
            except (TypeError, AttributeError, NotImplementedError):
                count = None
            if count is not None:
                current = Subset(dataset, randomizer.sample(range(len(dataset)), count))
        selected[category] = current

    def batches() -> Iterable[TaggedBatch]:
        for category, dataset in (copy.deepcopy(selected) if replay else selected).items():
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=loader_workers, shuffle=False)
            limit = max_batches
            if limit is None and calibration_size is not None:
                # ViCON's MySVD entrypoint computes ``int(cali / batch_size)``
                # and consumes that many complete loader iterations.
                limit = calibration_size // batch_size
            for index, batch in enumerate(loader):
                if limit is not None and index >= limit:
                    break
                yield TaggedBatch(str(category), batch)

    return batches


def _all_datasets(bundle: ViconBundle, workers: int, seed: int, split: str) -> Mapping[str, Any]:
    """Support both the research and vendored rollout-aware dataset API."""
    kwargs = {}
    if "rollout_cfg" in inspect.signature(bundle.all_datasets).parameters:
        kwargs["rollout_cfg"] = bundle.config.rollout
    return bundle.all_datasets(bundle.config.datasets, workers, seed, split, **kwargs)


def _vicon_dataset_seed(bundle: ViconBundle, split: str, seed: int, random_subset: bool) -> int:
    """Mirror ViCON's random-subset versus fixed-prefix data seed contract."""
    if split == "train" and not random_subset:
        return int(getattr(bundle.config, "train_seed", seed))
    return seed


def _evaluate_vicon_model(
    bundle: ViconBundle,
    model: nn.Module,
    batches: Callable[[], Iterable[TaggedBatch]],
) -> dict[str, Any]:
    model = model.to(bundle.device).eval()
    losses: dict[str, list[float]] = {}
    label_energies: dict[str, list[float]] = {}
    physical: dict[str, list[dict[str, float]]] = {}
    with torch.inference_mode():
        # The research evaluator traverses every dataset for loss first, then
        # traverses them again for physics. Keep the same dataset instances:
        # their time-pair RNG advances across both passes and across models.
        for tagged in batches():
            # ViCON's Trainer owns details such as channel/type masking and
            # normalisation.  Do not approximate those with a hand-written MSE.
            loss = float(bundle.loss(model, tagged.payload))
            data, _normalization_mean, _normalization_std = bundle.prepare(tagged.payload)
            min_examples = int(bundle.trainer.loss_cfg.min_ex)
            _initial, _labels, channel_mask = data
            # This matches ViCON's evaluation script: use the trainer label
            # helper and account for the spatial dimensions masked by c_mask.
            label = bundle.trainer._get_label(data)[:, min_examples:, :-1]
            channel_validity = channel_mask[:, :, :-1]
            label_energy = float(
                ((label.square() * channel_validity.float()).sum())
                / (channel_validity.sum() * label.shape[-1] * label.shape[-2] + 1e-12)
            )
            losses.setdefault(tagged.category, []).append(loss)
            label_energies.setdefault(tagged.category, []).append(label_energy)
        for tagged in batches():
            output, data, mean, std = bundle.forward(model, tagged.payload)
            min_examples = int(bundle.trainer.loss_cfg.min_ex)
            _initial, labels, _channel_mask = data
            prediction = output[:, min_examples:, :-1]
            target = labels[:, min_examples:, :-1]
            prediction_denormalized = prediction * std[:, :, :-1] + mean[:, :, :-1]
            target_denormalized = target * std[:, :, :-1] + mean[:, :, :-1]
            physical.setdefault(tagged.category, []).append(
                physics_metrics(tagged.category, prediction_denormalized, target_denormalized)
            )
    report = model_parameter_report(model)
    # Count fused Q/K/V projections without changing the native origin forward.
    report["linear_parameters"] += sum(
        parameter.numel() for layer in model.modules() if isinstance(layer, nn.MultiheadAttention)
        for name, parameter in layer.named_parameters(recurse=False)
        if name in {"in_proj_weight", "in_proj_bias", "q_proj_weight", "k_proj_weight", "v_proj_weight"}
    )
    return {
        **report,
        "losses": {key: _mean(values) for key, values in losses.items()},
        "label_energies": {key: _mean(values) for key, values in label_energies.items()},
        "loss_batches": {key: len(values) for key, values in losses.items()},
        "physics_batches": {key: len(values) for key, values in physical.items()},
        "relative_losses": {
            key: _mean(values) / (_mean(label_energies[key]) + 1e-12)
            for key, values in losses.items()
        },
        "physics": {key: average_metrics(values) for key, values in physical.items()},
    }


def _load_config(directory: Path, name: str, overrides: tuple[str, ...] = ()) -> Any:
    try:
        from hydra import compose, initialize_config_dir
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("VICON requires hydra-core and omegaconf; install the VICON route dependencies.") from exc
    config_name = name.removesuffix(".yaml")
    with initialize_config_dir(version_base=None, config_dir=str(directory.resolve())):
        return compose(config_name=config_name, overrides=list(overrides))


def _apply_data_root(
    config: Any, data_root: Path, *, config_overrides: tuple[str, ...] = (),
) -> None:
    """Resolve dataset folders using the same data-root rules as safe_svd.main.

    Keep ordinary relative subdirectories; rebase legacy absolute/../ paths
    using their basename. Explicit Hydra path overrides take precedence.
    """
    try:
        from omegaconf import DictConfig, ListConfig, open_dict
    except ModuleNotFoundError as exc:  # pragma: no cover - surfaced by _load_config first
        raise ModuleNotFoundError("VICON requires omegaconf") from exc
    root = data_root.expanduser().resolve()
    os.environ["SAFE_SVD_DATA_ROOT"] = str(root)
    explicit_fields = {override.lstrip("+").split("=", 1)[0] for override in config_overrides}

    def visit(value: Any, in_datasets: bool = False, location: str = "") -> None:
        if isinstance(value, DictConfig):
            with open_dict(value):
                for key in list(value.keys()):
                    child = value[key]
                    lowered = str(key).lower()
                    dataset_scope = in_datasets or lowered in {"datasets", "dataset"}
                    child_location = f"{location}.{key}" if location else str(key)
                    if lowered in {"data_root", "dataset_root", "data_dir", "dataset_dir"}:
                        value[key] = str(root)
                    elif dataset_scope and lowered in {"folder", "path", "data_path", "root", "file"} and isinstance(child, str):
                        source = Path(child).expanduser()
                        if child_location in explicit_fields:
                            destination = source
                        else:
                            relative = Path(source.name) if source.is_absolute() or ".." in source.parts else source
                            destination = root / relative
                        value[key] = str(destination.resolve())
                    else:
                        visit(child, dataset_scope, child_location)
        elif isinstance(value, ListConfig):
            for index, child in enumerate(value):
                visit(child, in_datasets, f"{location}[{index}]")

    visit(config)


def _checkpoint_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"VICON checkpoint {path} is not a mapping")
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, Mapping):
        raise TypeError(f"VICON checkpoint {path} does not contain a state dict")
    return {key.removeprefix("module."): value for key, value in state.items() if torch.is_tensor(value)}


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(path, map_location="cpu")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dataset_metrics(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Gather each category's losses and physics in a flat Poseidon record."""
    return {
        name: {
            **report["physics"].get(name, {}),
            "base_loss": loss,
            "rel_base_loss": report["relative_losses"][name],
            "model_size": report["parameters"],
            "linear_size": report["linear_parameters"],
        }
        for name, loss in report["losses"].items()
    }


def _dataset_sampling(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    return {
        "loss_batches": report["loss_batches"][name],
        "physics_batches": report["physics_batches"].get(name, 0),
        "label_energy": report["label_energies"][name],
    }


def print_summary(title: str, report: Mapping[str, Any]) -> None:
    """Print masked losses and physical metrics using the Poseidon entry layout."""
    print(f"\n{title}")
    print(f"  parameters: {report['parameters']}")
    for name, metrics in report['physics'].items():
        print(f"  {name}:")
        print(f"    loss: {report['losses'][name]}")
        print(f"    relative_loss: {report['relative_losses'][name]}")
        for key, value in metrics.items():
            print(f"    {key}: {value}")
