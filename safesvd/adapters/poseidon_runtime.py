"""Poseidon implementation of the SAFE-SVD adapter protocol.

This module owns only Poseidon-specific concerns: ScOT loading, downstream
trajectory sampling, rollout-based activation capture, and the physics-aware
Fisher loss.  All SVD mathematics is imported from :mod:`safesvd.core`.
"""

from __future__ import annotations

import json
import random
import sys
import gc
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from safesvd.core import (
    CompressionCallbacks,
    allocate_greedy_ranks,
    compress_clean_input,
    compress_sequentially,
    svd_whitener,
)


# These values reproduce MySVD_origin.  Keep them in the adapter rather than
# changing Poseidon's own inference code.
FISHER_BATCH_SIZE = 8
ORIGIN_CAPTURE_BATCH_SIZE = 256


@dataclass(frozen=True)
class PoseidonOptions:
    repo_root: Path
    poseidon_root: Path
    model_path: Path
    data_path: Path
    save_path: Path
    rank_root: Path
    cache_root: Path
    model_name: str
    dataset: str
    traj_num: int
    calibration_seed: int
    final_time: int
    time_back: int
    rank_ratio: float
    device: str
    use_fisher: bool
    use_prev_time: bool
    balance_ratio: float
    order_control: int
    gradient_loss: str
    alpha: float
    fast: bool
    overwrite: bool

    @property
    def calibration_tag(self) -> str:
        selection = "default" if self.calibration_seed == -1 else f"seed{self.calibration_seed}"
        return f"traj{self.traj_num}_{selection}_ftime{self.final_time}"

    @property
    def random_calibration(self) -> bool:
        """Whether this run uses seeded random trajectory selection."""
        return self.calibration_seed >= 0

    @property
    def rank_path(self) -> Path:
        return (
            self.rank_root / self.model_name / self.dataset / "MySVD_origin_Rank"
            / f"{self.calibration_tag}_order{self.order_control}_{self.gradient_loss}"
            / f"rank_{self.rank_ratio}_origin.json"
        )

    @property
    def profile_path(self) -> Path:
        return self.cache_root / self.model_name / self.dataset / "input" / self.calibration_tag / "profiling_mat.pt"

    @property
    def fisher_path(self) -> Path:
        return (
            self.cache_root / self.model_name / self.dataset / "fisher"
            / f"{self.calibration_tag}_order{self.order_control}_{self.gradient_loss}" / "Ldict.pt"
        )


class TrajectorySubset(Dataset):
    """A complete-trajectory calibration subset preserving source metadata."""

    def __init__(self, dataset, trajectory_indices: list[int]) -> None:
        self.source_dataset = dataset
        self.selected_trajectory_indices = tuple(trajectory_indices)
        self.num_trajectories = len(self.selected_trajectory_indices)
        total = int(dataset.num_trajectories)
        if total <= 0:
            raise ValueError("The source dataset must contain at least one trajectory.")
        if len(set(self.selected_trajectory_indices)) != self.num_trajectories:
            raise ValueError("trajectory_indices must not contain duplicates.")
        if any(index < 0 or index >= total for index in self.selected_trajectory_indices):
            raise IndexError(f"trajectory_indices must be in [0, {total}).")
        if len(dataset) % total:
            raise ValueError("Dataset length must be divisible by num_trajectories.")
        samples_per_trajectory = len(dataset) // total
        self.sample_indices = tuple(
            trajectory * samples_per_trajectory + offset
            for trajectory in self.selected_trajectory_indices
            for offset in range(samples_per_trajectory)
        )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int):
        return self.source_dataset[self.sample_indices[index]]

    def __getattr__(self, name: str):
        if name == "source_dataset":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "source_dataset"), name)


class PrevTimeDataset(Dataset):
    """Poseidon's previous-time counterpart used by temporal Fisher loss."""

    def __init__(self, base_dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        _, _, time_start, time_end = self.base_dataset._idx_map(index)
        return self.base_dataset[index - 1] if time_end > time_start else self.base_dataset[index]

    def __getattr__(self, name: str):
        return getattr(self.base_dataset, name)


def build_calibration_dataset(
    dataset_factory,
    *,
    dataset_name: str,
    num_trajectories: int,
    random_calibration: bool,
    calibration_seed: int,
    **dataset_kwargs,
):
    """Build the exact fixed-prefix or seeded trajectory selection of MySVD_origin."""
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive.")
    dataset = dataset_factory(
        dataset=dataset_name,
        which="train",
        num_trajectories=-1 if random_calibration else num_trajectories,
        **dataset_kwargs,
    )
    if not random_calibration:
        return dataset
    total = int(dataset.num_trajectories)
    if num_trajectories > total:
        raise ValueError(f"Cannot sample {num_trajectories} trajectories from {total}.")
    return TrajectorySubset(dataset, random.Random(calibration_seed).sample(range(total), num_trajectories))


def describe_calibration_selection(dataset, random_calibration: bool) -> str:
    if random_calibration:
        return f"random training trajectory indices: {list(dataset.selected_trajectory_indices)}"
    return f"fixed training trajectory prefix: [0, {dataset.num_trajectories})"


def _prepend_import_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _imports(options: PoseidonOptions):
    """Import the unmodified upstream ScOT package and SAFE compatibility code.

    ``poseidon_root`` is deliberately a checkout of camlab-ethz/poseidon, whose
    public Python package is named ``scOT``.  SAFE-SVD must never import the
    research-only ``poseidon_main`` fork: doing so would make compression depend
    on local edits to the upstream repository.
    """
    if not (options.poseidon_root / "scOT").is_dir():
        raise FileNotFoundError(
            "Poseidon source checkout must contain scOT/. Clone the unmodified "
            "https://github.com/camlab-ethz/poseidon repository into "
            f"{options.poseidon_root}."
        )
    _prepend_import_path(options.poseidon_root)
    try:
        from scOT.inference import get_trainer, rollout
        from scOT.problems.base import get_dataset
        from backends.poseidon.paper.helper.fisher_collector import (
            collect_fisher_dict_linear_only,
            collect_fisher_dict_time_sob,
        )
        from backends.poseidon.paper.helper.input_collector import (
            as_2d_x,
            get_module_by_name,
            input_catch,
            set_module_by_name,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Poseidon runtime dependencies are unavailable. Activate the Poseidon "
            "environment or set SAFE_SVD_RUNTIME_PYTHON to its Python executable."
        ) from exc
    return {
        "as_2d_x": as_2d_x,
        "collect_fisher_dict_linear_only": collect_fisher_dict_linear_only,
        "collect_fisher_dict_time_sob": collect_fisher_dict_time_sob,
        "get_dataset": get_dataset,
        "get_module_by_name": get_module_by_name,
        "get_trainer": get_trainer,
        "input_catch": input_catch,
        "rollout": rollout,
        "set_module_by_name": set_module_by_name,
    }


def _build_memory_safe_trainer(get_trainer, *, model_path: str, batch_size: int, dataset):
    """Create the upstream trainer with MySVD_origin's data-loader settings.

    Official Poseidon interprets ``workers=-1`` as every CPU assigned to the
    job.  That is unsafe for downstream HDF5 data: the profiling rollout starts
    one process per core immediately after Fisher collection and can exhaust
    node memory.  The local ``poseidon_main`` fork used ``0`` workers.  The
    official public factory requires a positive argument, so build with one
    worker and set the returned Trainer arguments before any dataloader exists.
    """
    trainer = get_trainer(model_path=model_path, batch_size=batch_size, dataset=dataset, workers=1)
    trainer.args.dataloader_num_workers = 0
    trainer.args.remove_unused_columns = False
    return trainer


@torch.no_grad()
def _profile_inputs(trainer, calibration_data, device: str, rollout) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The MySVD_origin profiling pass, including its float32 Gram sums."""
    target = torch.device(device)
    model = trainer.model.to(target).eval()
    layers: list[tuple[str, nn.Linear]] = []
    handles = []

    def hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], _output) -> None:
        x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
        module.raw_scaling_diag_matrix += x.T @ x

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = torch.zeros(
                module.in_features, module.in_features, device=target, dtype=torch.float32
            )
            handles.append(module.register_forward_hook(hook))
            layers.append((name, module))
    try:
        rollout(trainer, calibration_data)
    finally:
        for handle in handles:
            handle.remove()
    profile: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, module in layers:
        profile[name] = tuple(tensor.cpu().float() for tensor in svd_whitener(module.raw_scaling_diag_matrix.double()))
        del module.raw_scaling_diag_matrix
    model.cpu()
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return profile


def _capture_inputs(trainer, calibration_data, helpers: dict[str, object], device: str, fast: bool):
    """Return a compatibility callback using MySVD_fast's 512-sample chunk.

    The paired callback below is used by Poseidon compression.  This callback
    remains for the backend-neutral fallback path.
    """
    input_catch = helpers["input_catch"]
    as_2d_x = helpers["as_2d_x"]
    batch_size = ORIGIN_CAPTURE_BATCH_SIZE

    def capture(model: nn.Module, layer_name: str) -> torch.Tensor:
        module = helpers["get_module_by_name"](model, layer_name)
        assert isinstance(module, nn.Linear)
        inputs: list[torch.Tensor] = []
        for start in range(0, len(calibration_data), batch_size):
            subset = Subset(calibration_data, list(range(start, min(start + batch_size, len(calibration_data)))))
            captured = input_catch(trainer, model, layer_name, subset, torch.device(device))
            inputs.append(as_2d_x(captured, module.in_features).cpu())
        return tuple(inputs)

    return capture


def _capture_input_pairs(trainer, calibration_data, helpers: dict[str, object], device: str):
    """Yield one MySVD_fast-style teacher/student capture pair at a time.

    This avoids retaining all teacher chunks and then all student chunks before
    covariance accumulation.  At most one 512-sample pair is live at once.
    """
    input_catch = helpers["input_catch"]
    as_2d_x = helpers["as_2d_x"]
    target = torch.device(device)

    def capture_pairs(teacher: nn.Module, student: nn.Module, layer_name: str):
        teacher_module = helpers["get_module_by_name"](teacher, layer_name)
        student_module = helpers["get_module_by_name"](student, layer_name)
        if not isinstance(teacher_module, nn.Linear) or not isinstance(student_module, nn.Linear):
            raise TypeError(f"{layer_name} is expected to be nn.Linear in both teacher and student")
        for start in range(0, len(calibration_data), ORIGIN_CAPTURE_BATCH_SIZE):
            stop = min(start + ORIGIN_CAPTURE_BATCH_SIZE, len(calibration_data))
            subset = Subset(calibration_data, list(range(start, stop)))
            teacher_inputs = input_catch(trainer, teacher, layer_name, subset, target)
            student_inputs = input_catch(trainer, student, layer_name, subset, target)
            yield (
                as_2d_x(teacher_inputs, teacher_module.in_features).cpu(),
                as_2d_x(student_inputs, student_module.in_features).cpu(),
            )
            del teacher_inputs, student_inputs, subset
            if target.type == "cuda":
                torch.cuda.empty_cache()

    return capture_pairs


def _release_cuda_stage(trainer, device: str) -> None:
    """Approximate MySVD_fast's fresh-process boundary between stages."""
    trainer.model.cpu()
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _compression_progress():
    """Build a lazy terminal progress reporter for the sequential layer loop."""
    progress_bar = None

    def update(completed: int, total: int, layer_name: str) -> None:
        nonlocal progress_bar
        if progress_bar is None:
            from tqdm.auto import tqdm

            progress_bar = tqdm(total=total, desc="Compressing", unit="layer", dynamic_ncols=True)
        progress_bar.set_postfix_str(layer_name[-56:], refresh=False)
        progress_bar.update(completed - progress_bar.n)
        if completed == total:
            progress_bar.close()

    return update


def _load_or_collect_fisher(options: PoseidonOptions, trainer, calibration_data, helpers: dict[str, object]):
    if not options.use_fisher:
        return None
    if options.fisher_path.is_file() and not options.overwrite:
        return torch.load(options.fisher_path, map_location="cpu")
    print(f"Collecting Fisher: batch_size={FISHER_BATCH_SIZE}", flush=True)
    loader = DataLoader(calibration_data, batch_size=FISHER_BATCH_SIZE, shuffle=False)
    constants = calibration_data.constants
    kwargs = {
        "gradient_loss": options.gradient_loss,
        "order_control": options.order_control,
        "constants": constants,
        "transpose": getattr(calibration_data, "transpose", False),
        "balance_ratio": int(options.balance_ratio),
    }
    if options.use_prev_time:
        fisher = helpers["collect_fisher_dict_time_sob"](
            trainer,
            options.dataset,
            loader,
            DataLoader(PrevTimeDataset(calibration_data), batch_size=FISHER_BATCH_SIZE, shuffle=False),
            options.device, **kwargs,
        )
    else:
        fisher = helpers["collect_fisher_dict_linear_only"](trainer, options.dataset, loader, options.device, **kwargs)
    options.fisher_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fisher, options.fisher_path)
    return fisher


def run_poseidon(options: PoseidonOptions) -> Path:
    """Select ranks and compress a Poseidon model using the origin objective."""
    if options.traj_num <= 0:
        raise ValueError("traj_num must be positive")
    if options.final_time <= options.time_back:
        raise ValueError("final_time must be greater than time_back")
    helpers = _imports(options)
    target_time = options.final_time - options.time_back
    calibration_data = build_calibration_dataset(
        helpers["get_dataset"],
        dataset_name=options.dataset,
        num_trajectories=options.traj_num,
        random_calibration=options.random_calibration,
        calibration_seed=options.calibration_seed,
        data_path=str(options.data_path),
        fix_input_to_time_step=None,
        time_step_size=1,
        max_num_time_steps=target_time,
    )
    print("Calibration selection:", describe_calibration_selection(calibration_data, options.random_calibration))
    print("Building Poseidon trainer: eval_batch_size=256, dataloader_workers=0", flush=True)
    trainer = _build_memory_safe_trainer(
        helpers["get_trainer"], model_path=str(options.model_path), batch_size=256, dataset=calibration_data
    )

    fisher = _load_or_collect_fisher(options, trainer, calibration_data, helpers)
    _release_cuda_stage(trainer, options.device)
    if options.profile_path.is_file() and not options.overwrite:
        profile = torch.load(options.profile_path, map_location="cpu")
    else:
        print("Profiling activations: eval_batch_size=256, dataloader_workers=0", flush=True)
        profile = _profile_inputs(trainer, calibration_data, options.device, helpers["rollout"])
        options.profile_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(profile, options.profile_path)
    _release_cuda_stage(trainer, options.device)
    if options.rank_path.is_file() and not options.overwrite:
        ranks = json.loads(options.rank_path.read_text())
    else:
        print("Selecting ranks from Fisher and activation profiles", flush=True)
        ranks = allocate_greedy_ranks(trainer.model, fisher, profile, options.rank_ratio, options.device)
        options.rank_path.parent.mkdir(parents=True, exist_ok=True)
        options.rank_path.write_text(json.dumps(ranks, indent=2) + "\n")
    _release_cuda_stage(trainer, options.device)

    options.save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = options.save_path / "compressing_model.pt"
    total_linear_layers = sum(1 for module in trainer.model.modules() if isinstance(module, nn.Linear))
    checkpoint_every = max(int(total_linear_layers * 0.1), 1)
    initial_student = None
    compressed_layers = None
    if options.alpha != 0 and checkpoint_path.is_file() and not options.overwrite:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        initial_student = checkpoint_payload["model"]
        compressed_layers = set(checkpoint_payload["compressed_layers"])
        print(f"Resuming compression after {len(compressed_layers)} completed layers", flush=True)

    def checkpoint(student: nn.Module, compressed_layers: set[str]) -> None:
        """Mirror MySVD_fast's periodic CPU checkpoint and CUDA defragmentation."""
        torch.save(
            {"model": student.cpu(), "compressed_layers": sorted(compressed_layers)},
            checkpoint_path,
        )
        student.to(options.device)
        if torch.device(options.device).type == "cuda":
            torch.cuda.empty_cache()

    callbacks = CompressionCallbacks(
        capture_inputs=_capture_inputs(trainer, calibration_data, helpers, options.device, options.fast),
        capture_pairs=_capture_input_pairs(trainer, calibration_data, helpers, options.device),
        get_module=helpers["get_module_by_name"],
        set_module=helpers["set_module_by_name"],
        checkpoint=checkpoint,
        progress=_compression_progress(),
    )
    trainer.set_ar_steps(None, output_all_steps=False)
    capture_batch_size = ORIGIN_CAPTURE_BATCH_SIZE
    print(f"Compressing layers: capture_batch_size={capture_batch_size}", flush=True)
    if options.alpha == 0:
        model = compress_clean_input(
            trainer.model, ranks, profile, callbacks, device=options.device, left_whiteners=fisher,
        )
    else:
        model = compress_sequentially(
            trainer.model,
            ranks,
            callbacks,
            device=options.device,
            alpha=options.alpha,
            left_whiteners=fisher,
            checkpoint_every=checkpoint_every,
            initial_student=initial_student,
            compressed_layers=compressed_layers,
        )
    final_path = options.save_path / "compressed_model_origin.pt"
    torch.save(model, final_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return final_path
