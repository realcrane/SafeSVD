"""SAFE-SVD runtime adapter for VICON.

Only VICON-specific concerns are implemented here: loading its Hydra model,
turning VICON batches into model calls, and computing its task/Fisher metrics.
Whitening, rank selection, factorisation, and sequential replacement are all
delegated to :mod:`safesvd.core` and ``common_runtime``.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Subset

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
from safesvd.adapters.vicon_attention import (
    convert_attn_to_linear,
    disable_transformer_fast_path,
    patched_vicon_fisher_forward,
)
from safesvd.adapters.vicon_physics import SOBOLEV_ORDERS, average_metrics, physics_metrics
from safesvd.core import CompressionCallbacks, allocate_ranks, compress_clean_input, compress_sequentially


@dataclass(frozen=True)
class ViconOptions:
    vicon_root: Path
    model_path: Path
    data_path: Path
    save_path: Path
    profile_path: Path
    fisher_path: Path
    rank_path: Path
    rank_ratio: float
    rank_selector: str
    calibration_size: int
    calibration_seed: int
    alpha: float
    device: str
    config_dir: Path | None = None
    config_name: str = "default"
    config_overrides: tuple[str, ...] = ()
    batch_size: int = 128
    use_fisher: bool = True
    fisher_loss: str = "l1"
    balance_ratio: float = 10.0
    trace_balance: bool = True
    random_calibration: bool = True
    overwrite: bool = False


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
    batch_size: int = 128
    max_batches: int | None = None


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


def run_vicon(options: ViconOptions) -> Path:
    """Run VICON profile, Fisher/rank selection, and shared SAFE-SVD compression."""
    _validate_options(options)
    bundle = load_vicon_bundle(
        options.vicon_root, options.model_path, options.data_path, options.device,
        config_dir=options.config_dir, config_name=options.config_name, config_overrides=options.config_overrides,
    )
    batch_factory = _vicon_batch_factory(
        bundle,
        split="train",
        batch_size=options.batch_size,
        calibration_size=options.calibration_size,
        seed=options.calibration_seed,
        random_subset=options.random_calibration,
        source_dataset_workers=0,
    )
    trace_groups = {
        name: SOBOLEV_ORDERS.get(name, 1)
        for name in _dataset_names(
            bundle,
            "train",
            options.calibration_seed,
            random_subset=options.random_calibration,
            source_dataset_workers=0,
        )
    }

    profile = _load_or_collect(
        options.profile_path,
        options.overwrite,
        lambda: collect_input_whiteners(
            bundle.trainer.model, batch_factory, bundle_forward(bundle), device=bundle.device,
            # VICON aligns profile covariances across every physical dataset.  Its
            # Sobolev-order grouping is specific to the Fisher objective below.
            trace_balance=options.trace_balance,
        ),
    )
    fisher = None
    if options.use_fisher:
        fisher = _load_or_collect(
            options.fisher_path,
            options.overwrite,
            lambda: _collect_vicon_fisher(bundle, batch_factory, options, trace_groups),
        )
    ranks = _load_or_select_ranks(options, bundle.trainer.model, fisher, profile)

    options.save_path.mkdir(parents=True, exist_ok=True)
    in_progress = options.save_path / "compressing_model.pt"
    initial_student: nn.Module | None = None
    compressed_layers: set[str] | None = None
    if options.alpha != 0 and in_progress.is_file() and not options.overwrite:
        partial = _torch_load(in_progress)
        if isinstance(partial, Mapping) and isinstance(partial.get("model"), nn.Module):
            initial_student = partial["model"]
            compressed_layers = set(partial.get("compressed_layers", ()))
            print(f"Resuming VICON compression after {len(compressed_layers)} layers", flush=True)

    total_layers = sum(isinstance(module, nn.Linear) for module in bundle.trainer.model.modules())
    checkpoint_every = max(1, total_layers // 10)

    def checkpoint(student: nn.Module, done: set[str]) -> None:
        torch.save({"model": student.cpu(), "compressed_layers": sorted(done)}, in_progress)
        student.to(bundle.device)

    compression_batches = cache_tagged_batches(batch_factory) if options.alpha != 0 else batch_factory
    callbacks = CompressionCallbacks(
        capture_inputs=lambda _model, _name: (),
        get_module=get_module_by_name,
        set_module=set_module_by_name,
        checkpoint=checkpoint,
        progress=_progress("VICON"),
        capture_statistics=CategoryAwareStatistics(
            compression_batches,
            bundle_forward(bundle),
            device=bundle.device,
            trace_balance=options.trace_balance,
        ),
    )
    if options.alpha == 0:
        compressed = compress_clean_input(
            bundle.trainer.model, ranks, profile, callbacks, device=bundle.device, left_whiteners=fisher,
        )
    else:
        compressed = compress_sequentially(
            bundle.trainer.model,
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
        options.save_path / "compressed_model.pth",
        compressed,
        model_name="vicon",
        ranks=ranks,
        metadata={
            "source_root": str(options.vicon_root),
            "config_dir": str(_config_dir(options)),
            "config_name": options.config_name,
            "checkpoint": str(options.model_path),
            "data_root": str(options.data_path),
            "alpha": options.alpha,
            "rank_ratio": options.rank_ratio,
            "rank_selector": options.rank_selector,
        },
    )
    if in_progress.exists():
        in_progress.unlink()
    print(f"Compressed VICON model saved to {output}")
    return output


def evaluate_vicon(options: ViconEvaluationOptions) -> Path:
    """Evaluate baseline and optional compressed VICON model on identical test data."""
    bundle = load_vicon_bundle(
        options.vicon_root, options.model_path, options.data_path, options.device,
        config_dir=options.config_dir, config_name=options.config_name, config_overrides=options.config_overrides,
    )
    batch_factory = _vicon_batch_factory(
        bundle,
        split="test",
        batch_size=options.batch_size,
        calibration_size=None,
        seed=int(getattr(bundle.config, "test_seed", 0)),
        random_subset=False,
        max_batches=options.max_batches,
    )
    origin = _evaluate_vicon_model(bundle, bundle.trainer.model, batch_factory)
    payload: dict[str, Any] = {
        "model": "vicon",
        "origin": {"model": str(options.model_path), "metrics": origin},
    }
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
        payload["compressed"] = {
            "model": str(options.compressed_path),
            "metrics": compressed_metrics,
            "increase": _nested_increases(compressed_metrics, origin),
        }
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
) -> ViconBundle:
    """Load a VICON model/trainer without importing paper compression code."""
    package_root = find_vicon_package(root)
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    models = importlib.import_module("VICON.src.models")
    trainer_module = importlib.import_module("VICON.src.trainer")
    dataset_module = importlib.import_module("VICON.src.dataset")
    config = _load_config(config_dir or package_root / "configs", config_name, config_overrides)
    _apply_data_root(config, data_root)
    model_type = str(getattr(config.model, "type", ""))
    constructor = getattr(models, "ICON_CROPPED" if model_type == "crop" else "ICON_UNCROPPED")
    model = constructor(config.model)
    trainer = trainer_module.Trainer(
        model, config.model, config.opt, config.loss,
        trainable_mode=getattr(config, "trainable_mode", None),
    )
    state_dict = _checkpoint_state_dict(checkpoint)
    trainer.model.load_state_dict(state_dict)
    convert_attn_to_linear(trainer.model)
    disable_transformer_fast_path(trainer.model)
    target = resolve_device(device)
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


def bundle_forward(bundle: ViconBundle) -> Callable[[nn.Module, Any], torch.Tensor]:
    """Adapt VICON's Trainer API to the common runtime forward callback."""
    def forward(model: nn.Module, payload: Any) -> torch.Tensor:
        output, _data, _mean, _std = bundle.forward(model, payload)
        return output

    return forward


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
) -> Callable[[], Iterable[TaggedBatch]]:
    loader_workers = int(getattr(bundle.config, "dataset_workers", 0))
    dataset_workers = loader_workers if source_dataset_workers is None else source_dataset_workers
    dataset_seed = _vicon_dataset_seed(bundle, split, seed, random_subset)
    datasets = bundle.all_datasets(bundle.config.datasets, dataset_workers, dataset_seed, split)
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
        for category, dataset in selected.items():
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


def _dataset_names(
    bundle: ViconBundle,
    split: str,
    seed: int,
    *,
    random_subset: bool,
    source_dataset_workers: int | None = None,
) -> tuple[str, ...]:
    loader_workers = int(getattr(bundle.config, "dataset_workers", 0))
    dataset_workers = loader_workers if source_dataset_workers is None else source_dataset_workers
    datasets = bundle.all_datasets(
        bundle.config.datasets,
        dataset_workers,
        _vicon_dataset_seed(bundle, split, seed, random_subset),
        split,
    )
    return tuple(str(name) for name in datasets)


def _vicon_dataset_seed(bundle: ViconBundle, split: str, seed: int, random_subset: bool) -> int:
    """Mirror ViCON's random-subset versus fixed-prefix data seed contract."""
    if split == "train" and not random_subset:
        return int(getattr(bundle.config, "train_seed", seed))
    return seed


def _vicon_fisher_loss(
    bundle: ViconBundle,
    loss_name: str,
    balance_ratio: float,
) -> Callable[[nn.Module, TaggedBatch], torch.Tensor]:
    functions = {"l1": functional.l1_loss, "mse": functional.mse_loss, "smooth_l1": functional.smooth_l1_loss}
    try:
        derivative_loss = functions[loss_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported VICON Fisher loss {loss_name!r}") from exc

    def loss(model: nn.Module, tagged: TaggedBatch) -> torch.Tensor:
        output, data, _mean, _std = bundle.forward(model, tagged.payload)
        category = tagged.category
        min_examples = int(bundle.trainer.loss_cfg.min_ex)
        _initial, labels, channel_mask = data
        prediction = output[:, min_examples:, :-1]
        target = labels[:, min_examples:, :-1]
        type_mask = labels[:, min_examples:, -1:]
        mask = (type_mask == 0).float() * channel_mask[:, :, :-1].float()
        base = _masked_loss(prediction, target, mask, functional.mse_loss)
        order = SOBOLEV_ORDERS.get(category, 1)
        if order <= 0:
            return base
        sobolev = _vicon_sobolev_loss(prediction, target, channel_mask[:, :, :-1], type_mask, order, derivative_loss)
        value_component = _masked_loss(prediction, target, mask, derivative_loss)
        return base + balance_ratio * (sobolev - value_component)

    return loss


def _collect_vicon_fisher(
    bundle: ViconBundle,
    batches: Callable[[], Iterable[TaggedBatch]],
    options: ViconOptions,
    trace_groups: Mapping[str, object],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Collect Fisher data with ViCON's crop-model-safe forward patch."""
    with patched_vicon_fisher_forward(bundle.trainer.model, checkpoint_layers=True):
        return collect_fisher_whiteners(
            bundle.trainer.model,
            batches,
            _vicon_fisher_loss(bundle, options.fisher_loss, options.balance_ratio),
            device=bundle.device,
            trace_balance=options.trace_balance,
            trace_groups=trace_groups,
        )


def _masked_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, fn: Callable[..., torch.Tensor]) -> torch.Tensor:
    values = fn(prediction, target, reduction="none") * mask
    return values.sum() / mask.sum().clamp(min=1.0)


def _vicon_sobolev_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_mask: torch.Tensor,
    type_mask: torch.Tensor,
    order: int,
    fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Exact first/second-order masked finite-difference objective from MySVD."""
    interior = (type_mask == 0).float()
    mask = interior * channel_mask.float()
    total = _masked_loss(prediction, target, mask, fn)
    if order < 1:
        return total
    prediction_x = prediction[..., 1:] - prediction[..., :-1]
    target_x = target[..., 1:] - target[..., :-1]
    prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    interior_x = torch.minimum(interior[..., 1:], interior[..., :-1])
    interior_y = torch.minimum(interior[..., 1:, :], interior[..., :-1, :])
    total = total + _masked_loss(prediction_x, target_x, interior_x * channel_mask.float(), fn)
    total = total + _masked_loss(prediction_y, target_y, interior_y * channel_mask.float(), fn)
    if order < 2:
        return total
    prediction_xx = prediction_x[..., 1:] - prediction_x[..., :-1]
    target_xx = target_x[..., 1:] - target_x[..., :-1]
    prediction_yy = prediction_y[..., 1:, :] - prediction_y[..., :-1, :]
    target_yy = target_y[..., 1:, :] - target_y[..., :-1, :]
    interior_xx = torch.minimum(interior_x[..., 1:], interior_x[..., :-1])
    interior_yy = torch.minimum(interior_y[..., 1:, :], interior_y[..., :-1, :])
    total = total + _masked_loss(prediction_xx, target_xx, interior_xx * channel_mask.float(), fn)
    total = total + _masked_loss(prediction_yy, target_yy, interior_yy * channel_mask.float(), fn)
    return total


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
        for tagged in batches():
            # ViCON's Trainer owns details such as channel/type masking and
            # normalisation.  Do not approximate those with a hand-written MSE.
            loss = float(bundle.loss(model, tagged.payload))
            output, data, mean, std = bundle.forward(model, tagged.payload)
            min_examples = int(bundle.trainer.loss_cfg.min_ex)
            _initial, labels, channel_mask = data
            prediction = output[:, min_examples:, :-1]
            target = labels[:, min_examples:, :-1]
            type_mask = labels[:, min_examples:, -1:]
            del prediction, target, type_mask
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
            # Reconstruct the physical fields after preserving the trainer's
            # loss semantics above.
            prediction = output[:, min_examples:, :-1]
            target = labels[:, min_examples:, :-1]
            prediction_denormalized = prediction * std[:, :, :-1] + mean[:, :, :-1]
            target_denormalized = target * std[:, :, :-1] + mean[:, :, :-1]
            physical.setdefault(tagged.category, []).append(
                physics_metrics(tagged.category, prediction_denormalized, target_denormalized)
            )
    return {
        **model_parameter_report(model),
        "losses": {key: _mean(values) for key, values in losses.items()},
        "relative_losses": {
            key: _mean(values) / (_mean(label_energies[key]) + 1e-12)
            for key, values in losses.items()
        },
        "physics": {key: average_metrics(values) for key, values in physical.items()},
    }


def _load_or_collect(path: Path, overwrite: bool, collector: Callable[[], Any]) -> Any:
    if path.is_file() and not overwrite:
        print(f"Loading cache: {path}", flush=True)
        return _torch_load(path)
    result = collector()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


def _load_or_select_ranks(
    options: ViconOptions,
    model: nn.Module,
    fisher: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None,
    profile: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, int | None]:
    if options.rank_path.is_file() and not options.overwrite:
        return json.loads(options.rank_path.read_text())
    ranks = allocate_ranks(
        model,
        fisher,
        profile,
        options.rank_ratio,
        options.device,
        selector=options.rank_selector,
    )
    options.rank_path.parent.mkdir(parents=True, exist_ok=True)
    options.rank_path.write_text(json.dumps(ranks, indent=2) + "\n")
    return ranks


def _load_config(directory: Path, name: str, overrides: tuple[str, ...] = ()) -> Any:
    try:
        from hydra import compose, initialize_config_dir
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("VICON requires hydra-core and omegaconf; install the VICON route dependencies.") from exc
    config_name = name.removesuffix(".yaml")
    with initialize_config_dir(version_base=None, config_dir=str(directory.resolve())):
        return compose(config_name=config_name, overrides=list(overrides))


def _apply_data_root(config: Any, data_root: Path) -> None:
    """Rebase common VICON dataset config fields and expose a stable env var.

    Direct absolute dataset paths are deliberately rejected when they point
    outside ``data_root``.  Silently retaining a path from the checkout
    author's machine would make the public ``--data-root`` contract unsafe.
    """
    try:
        from omegaconf import DictConfig, ListConfig, open_dict
    except ModuleNotFoundError as exc:  # pragma: no cover - surfaced by _load_config first
        raise ModuleNotFoundError("VICON requires omegaconf") from exc
    root = data_root.expanduser().resolve()
    os.environ["SAFE_SVD_DATA_ROOT"] = str(root)
    unresolved_absolute_paths: list[str] = []

    def is_within_root(path: Path) -> bool:
        try:
            path.expanduser().resolve().relative_to(root)
            return True
        except ValueError:
            return False

    def visit(value: Any, in_datasets: bool = False, location: str = "config") -> None:
        if isinstance(value, DictConfig):
            with open_dict(value):
                for key in list(value.keys()):
                    child = value[key]
                    lowered = str(key).lower()
                    dataset_scope = in_datasets or lowered in {"datasets", "dataset"}
                    child_location = f"{location}.{key}"
                    if lowered in {"data_root", "dataset_root", "data_dir", "dataset_dir"}:
                        value[key] = str(root)
                    elif dataset_scope and lowered in {"path", "data_path", "root", "file"} and isinstance(child, str):
                        source = Path(child).expanduser()
                        if source.is_absolute():
                            if not is_within_root(source):
                                unresolved_absolute_paths.append(f"{child_location}={child!r}")
                        else:
                            value[key] = str(root / source)
                    else:
                        visit(child, dataset_scope, child_location)
        elif isinstance(value, ListConfig):
            for index, child in enumerate(value):
                visit(child, in_datasets, f"{location}[{index}]")

    visit(config)
    if unresolved_absolute_paths:
        details = "; ".join(unresolved_absolute_paths[:3])
        suffix = "" if len(unresolved_absolute_paths) <= 3 else f"; and {len(unresolved_absolute_paths) - 3} more"
        raise ValueError(
            "VICON config contains absolute dataset paths outside --data-root: "
            f"{details}{suffix}. Pass matching --config-override values or choose a data root that contains them."
        )


def _checkpoint_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"VICON checkpoint {path} is not a mapping")
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, Mapping):
        raise TypeError(f"VICON checkpoint {path} does not contain a state dict")
    return {key.removeprefix("module."): value for key, value in state.items() if torch.is_tensor(value)}


def _validate_options(options: ViconOptions) -> None:
    if options.calibration_size <= 0 or options.batch_size <= 0:
        raise ValueError("VICON calibration size and batch size must be positive")
    if options.calibration_size < options.batch_size:
        raise ValueError(
            "VICON calibration size must be at least one batch; the paper route uses "
            "floor(calibration_size / batch_size) calibration batches."
        )
    if not 0 < options.rank_ratio <= 1:
        raise ValueError("rank_ratio must be in (0, 1]")
    if options.rank_selector not in {"legacy", "safe"}:
        raise ValueError("rank_selector must be one of: legacy, safe")
    if not 0 <= options.alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if not options.model_path.is_file():
        raise FileNotFoundError(f"VICON checkpoint not found: {options.model_path}")
    if not options.data_path.exists():
        raise FileNotFoundError(f"VICON data root not found: {options.data_path}")
    find_vicon_package(options.vicon_root)


def _config_dir(options: ViconOptions) -> Path:
    return options.config_dir or find_vicon_package(options.vicon_root) / "configs"


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(path, map_location="cpu")


def _progress(prefix: str) -> Callable[[int, int, str], None]:
    def progress(index: int, total: int, name: str) -> None:
        print(f"[{prefix}] {index}/{total}: {name}", flush=True)

    return progress


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _nested_increases(value: Any, baseline: Any) -> Any:
    if isinstance(value, Mapping) and isinstance(baseline, Mapping):
        return {key: _nested_increases(item, baseline[key]) for key, item in value.items() if key in baseline}
    if isinstance(value, (int, float)) and isinstance(baseline, (int, float)) and baseline != 0:
        return round(100.0 * (float(value) - float(baseline)) / abs(float(baseline)), 6)
    return None
