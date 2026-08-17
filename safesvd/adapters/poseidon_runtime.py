"""Poseidon implementation of the SAFE-SVD adapter protocol.

This module owns only Poseidon-specific concerns: ScOT loading, downstream
trajectory sampling, rollout-based activation capture, and the physics-aware
Fisher loss.  All SVD mathematics is imported from :mod:`safesvd.core`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from safesvd.core import (
    CompressionCallbacks,
    allocate_greedy_ranks,
    compress_clean_input,
    compress_sequentially,
    svd_whitener,
)


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
    random_calibration: bool
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
        tag = f"traj{self.traj_num}"
        if self.random_calibration:
            tag += f"_random_seed{self.calibration_seed}"
        return f"{tag}_ftime{self.final_time}"

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


def _prepend_import_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _imports(options: PoseidonOptions):
    """Import runtime dependencies only after the upstream source path is known."""
    _prepend_import_path(options.repo_root)
    _prepend_import_path(options.poseidon_root)
    try:
        from poseidon_main.scOT.inference import get_trainer, rollout
        from poseidon_main.scOT.problems.base import get_dataset
        from backends.poseidon.paper.helper.calibration_dataset import (
            build_calibration_dataset,
            describe_calibration_selection,
        )
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
        "build_calibration_dataset": build_calibration_dataset,
        "collect_fisher_dict_linear_only": collect_fisher_dict_linear_only,
        "collect_fisher_dict_time_sob": collect_fisher_dict_time_sob,
        "describe_calibration_selection": describe_calibration_selection,
        "get_dataset": get_dataset,
        "get_module_by_name": get_module_by_name,
        "get_trainer": get_trainer,
        "input_catch": input_catch,
        "rollout": rollout,
        "set_module_by_name": set_module_by_name,
    }


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
        rollout(trainer, calibration_data, model_path=trainer.model.config.name_or_path)
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
    """Return a core callback that reproduces origin's 512-sample capture loop.

    ``fast`` changes only the subset size.  The accumulated sufficient
    statistics are unchanged, so it is a performance parameter rather than a
    different compression objective.
    """
    input_catch = helpers["input_catch"]
    as_2d_x = helpers["as_2d_x"]
    batch_size = 1024 if fast else 512

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


def _load_or_collect_fisher(options: PoseidonOptions, trainer, calibration_data, helpers: dict[str, object]):
    if not options.use_fisher:
        return None
    if options.fisher_path.is_file() and not options.overwrite:
        return torch.load(options.fisher_path, map_location="cpu")
    loader = DataLoader(calibration_data, batch_size=32, shuffle=False)
    constants = calibration_data.constants
    kwargs = {
        "gradient_loss": options.gradient_loss,
        "order_control": options.order_control,
        "constants": constants,
        "transpose": getattr(calibration_data, "transpose", False),
        "balance_ratio": int(options.balance_ratio),
    }
    if options.use_prev_time:
        from backends.poseidon.paper.helper.prevds_collector import PrevTimeDataset
        fisher = helpers["collect_fisher_dict_time_sob"](
            trainer, options.dataset, loader, DataLoader(PrevTimeDataset(calibration_data), batch_size=32, shuffle=False),
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
    calibration_data = helpers["build_calibration_dataset"](
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
    print("Calibration selection:", helpers["describe_calibration_selection"](calibration_data, options.random_calibration))
    trainer = helpers["get_trainer"](model_path=str(options.model_path), batch_size=256, dataset=calibration_data)

    fisher = _load_or_collect_fisher(options, trainer, calibration_data, helpers)
    if options.profile_path.is_file() and not options.overwrite:
        profile = torch.load(options.profile_path, map_location="cpu")
    else:
        profile = _profile_inputs(trainer, calibration_data, options.device, helpers["rollout"])
        options.profile_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(profile, options.profile_path)
    if options.rank_path.is_file() and not options.overwrite:
        ranks = json.loads(options.rank_path.read_text())
    else:
        ranks = allocate_greedy_ranks(trainer.model, fisher, profile, options.rank_ratio, options.device)
        options.rank_path.parent.mkdir(parents=True, exist_ok=True)
        options.rank_path.write_text(json.dumps(ranks, indent=2) + "\n")

    options.save_path.mkdir(parents=True, exist_ok=True)
    callbacks = CompressionCallbacks(
        capture_inputs=_capture_inputs(trainer, calibration_data, helpers, options.device, options.fast),
        get_module=helpers["get_module_by_name"],
        set_module=helpers["set_module_by_name"],
    )
    trainer.set_ar_steps(None, output_all_steps=False)
    if options.alpha == 0:
        model = compress_clean_input(
            trainer.model, ranks, profile, callbacks, device=options.device, left_whiteners=fisher,
        )
    else:
        model = compress_sequentially(
            trainer.model, ranks, callbacks, device=options.device, alpha=options.alpha, left_whiteners=fisher,
        )
    final_path = options.save_path / "compressed_model_origin.pt"
    torch.save(model, final_path)
    return final_path
