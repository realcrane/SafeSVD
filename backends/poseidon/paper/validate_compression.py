"""Validate a few MySVD_origin compression steps against a saved model.

The script deliberately replays the numerical path in
``MySVD_origin.compression_method.compress_comb_input.mysvd_comb``: it uses
the original 512-sample covariance batches, float64 covariances, and the
original input collector.  After each low-rank layer is rebuilt, its effective
weight (``second.weight @ first.weight``) is compared with the corresponding
layer in a saved compressed model.

Comparing the effective weight is important: an SVD factorisation is not
unique, so two equivalent compressed layers can have different signs in their
individual factor tensors.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Subset


# Make the script runnable both from the repository root and from this folder.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
os.chdir(REPOSITORY_ROOT)

def load_project_dependencies() -> None:
    """Delay heavyweight Poseidon imports so ``--help`` works in any Python env."""
    global get_trainer, get_dataset
    global as_2d_x, get_module_by_name, input_catch, set_module_by_name, to_2d_features
    global svd_whitener

    from poseidon_main.scOT.inference import get_trainer as imported_get_trainer
    from poseidon_main.scOT.problems.base import get_dataset as imported_get_dataset
    from MySVD_origin.helper.input_collector import (
        as_2d_x as imported_as_2d_x,
        get_module_by_name as imported_get_module_by_name,
        input_catch as imported_input_catch,
        set_module_by_name as imported_set_module_by_name,
        to_2d_features as imported_to_2d_features,
    )
    from MySVD_origin.helper.whitener import svd_whitener as imported_svd_whitener

    get_trainer = imported_get_trainer
    get_dataset = imported_get_dataset
    as_2d_x = imported_as_2d_x
    get_module_by_name = imported_get_module_by_name
    input_catch = imported_input_catch
    set_module_by_name = imported_set_module_by_name
    to_2d_features = imported_to_2d_features
    svd_whitener = imported_svd_whitener


def torch_load(path: Path, *, map_location: str | torch.device) -> Any:
    """Load both old and new PyTorch checkpoints without assuming a format."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before ``weights_only`` existed.
        return torch.load(path, map_location=map_location)


def unwrap_model(loaded: Any, source: Path) -> nn.Module:
    """Accept a directly saved model or a resumable MySVD checkpoint dict."""
    if isinstance(loaded, nn.Module):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("model"), nn.Module):
        return loaded["model"]
    raise TypeError(
        f"{source} does not contain an nn.Module or a checkpoint dict with a "
        "'model' nn.Module."
    )


def _max_abs_and_relative(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a64 = a.detach().cpu().double()
    b64 = b.detach().cpu().double()
    max_abs = (a64 - b64).abs().max().item() if a64.numel() else 0.0
    scale = max(a64.abs().max().item() if a64.numel() else 0.0,
                b64.abs().max().item() if b64.numel() else 0.0,
                torch.finfo(torch.float64).eps)
    return max_abs, max_abs / scale


def _allclose(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> bool:
    return a.shape == b.shape and torch.allclose(
        a.detach().cpu(), b.detach().cpu(), atol=atol, rtol=rtol
    )


def _low_rank_parts(module: nn.Module) -> Tuple[nn.Linear, nn.Linear] | None:
    """Return (right, left) for the ``Linear -> Linear`` MySVD replacement."""
    if not isinstance(module, nn.Sequential) or len(module) != 2:
        return None
    right, left = module[0], module[1]
    if not isinstance(right, nn.Linear) or not isinstance(left, nn.Linear):
        return None
    return right, left


def compare_layer(
    layer_name: str,
    replayed: nn.Module,
    reference: nn.Module,
    *,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    """Compare modules while treating the reconstructed weight as authoritative."""
    replayed_parts = _low_rank_parts(replayed)
    reference_parts = _low_rank_parts(reference)

    if replayed_parts is not None and reference_parts is not None:
        replayed_right, replayed_left = replayed_parts
        reference_right, reference_left = reference_parts
        replayed_weight = replayed_left.weight @ replayed_right.weight
        reference_weight = reference_left.weight @ reference_right.weight
        weight_abs, weight_rel = _max_abs_and_relative(replayed_weight, reference_weight)

        bias_ok = (replayed_left.bias is None and reference_left.bias is None) or (
            replayed_left.bias is not None
            and reference_left.bias is not None
            and _allclose(replayed_left.bias, reference_left.bias, atol, rtol)
        )
        bias_abs, bias_rel = (0.0, 0.0)
        if replayed_left.bias is not None and reference_left.bias is not None:
            bias_abs, bias_rel = _max_abs_and_relative(
                replayed_left.bias, reference_left.bias
            )

        # These are diagnostic only.  Singular-vector signs can differ even
        # when the two compressed linear maps are exactly equivalent.
        right_abs, _ = _max_abs_and_relative(
            replayed_right.weight, reference_right.weight
        )
        left_abs, _ = _max_abs_and_relative(
            replayed_left.weight, reference_left.weight
        )
        return {
            "name": layer_name,
            "kind": "low_rank",
            "replayed_shape": tuple(replayed_weight.shape),
            "reference_shape": tuple(reference_weight.shape),
            "rank_replayed": replayed_right.out_features,
            "rank_reference": reference_right.out_features,
            "weight_max_abs": weight_abs,
            "weight_max_rel": weight_rel,
            "bias_max_abs": bias_abs,
            "bias_max_rel": bias_rel,
            "factor_right_max_abs": right_abs,
            "factor_left_max_abs": left_abs,
            "passed": _allclose(replayed_weight, reference_weight, atol, rtol)
            and bias_ok,
        }

    # A zero-rank layer is represented as an ordinary Linear with a zeroed
    # weight in the current MySVD implementation.  Supporting it here makes
    # the validation useful for rank JSON files containing zero ranks as well.
    if isinstance(replayed, nn.Linear) and isinstance(reference, nn.Linear):
        weight_abs, weight_rel = _max_abs_and_relative(replayed.weight, reference.weight)
        bias_ok = (replayed.bias is None and reference.bias is None) or (
            replayed.bias is not None
            and reference.bias is not None
            and _allclose(replayed.bias, reference.bias, atol, rtol)
        )
        bias_abs, bias_rel = (0.0, 0.0)
        if replayed.bias is not None and reference.bias is not None:
            bias_abs, bias_rel = _max_abs_and_relative(replayed.bias, reference.bias)
        return {
            "name": layer_name,
            "kind": "linear",
            "replayed_shape": tuple(replayed.weight.shape),
            "reference_shape": tuple(reference.weight.shape),
            "rank_replayed": None,
            "rank_reference": None,
            "weight_max_abs": weight_abs,
            "weight_max_rel": weight_rel,
            "bias_max_abs": bias_abs,
            "bias_max_rel": bias_rel,
            "factor_right_max_abs": None,
            "factor_left_max_abs": None,
            "passed": _allclose(replayed.weight, reference.weight, atol, rtol)
            and bias_ok,
        }

    return {
        "name": layer_name,
        "kind": "architecture_mismatch",
        "replayed_shape": None,
        "reference_shape": None,
        "rank_replayed": None,
        "rank_reference": None,
        "weight_max_abs": None,
        "weight_max_rel": None,
        "bias_max_abs": None,
        "bias_max_rel": None,
        "factor_right_max_abs": None,
        "factor_left_max_abs": None,
        "passed": False,
        "detail": f"replayed={type(replayed).__name__}, reference={type(reference).__name__}",
    }


def load_ranks(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        ranks = json.load(handle)
    if not isinstance(ranks, dict):
        raise TypeError(f"Rank file must be a JSON object: {path}")
    return ranks


def compress_one_layer_like_origin(
    trainer: Any,
    teacher: nn.Module,
    student: nn.Module,
    layer_name: str,
    original_layer: nn.Linear,
    rank: int,
    calibration_dataset: Any,
    ldict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] | None,
    alpha: float,
    device: torch.device,
    batch_size: int,
) -> None:
    """Replay one non-zero-rank iteration of origin ``mysvd_comb`` exactly."""
    in_features, out_features = original_layer.in_features, original_layer.out_features
    sigma_hhp = torch.zeros((in_features, in_features), device=device, dtype=torch.float64)
    sigma_hh = torch.zeros((in_features, in_features), device=device, dtype=torch.float64)
    sigma_hphp = torch.zeros((in_features, in_features), device=device, dtype=torch.float64)

    for start in range(0, len(calibration_dataset), batch_size):
        subset = Subset(calibration_dataset, list(range(start, min(start + batch_size, len(calibration_dataset)))))

        h = input_catch(trainer, teacher, layer_name, subset, device)
        h = as_2d_x(h, in_features).to(device, non_blocking=True).double()
        h = to_2d_features(h, in_features)
        hp = input_catch(trainer, student, layer_name, subset, device)
        hp = as_2d_x(hp, in_features).to(device, non_blocking=True).double()
        hp = to_2d_features(hp, in_features)

        sigma_hphp.addmm_(hp.T, hp)
        sigma_hh.addmm_(h.T, h)
        sigma_hhp.addmm_(h.T, hp)
        del h, hp, subset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    weight = original_layer.weight.detach().double().to(device)
    sigma_comb = alpha * sigma_hphp + (1.0 - alpha) * sigma_hh
    right_whitener, right_whitener_inv_t = svd_whitener(sigma_comb, eps=1e-6)
    right_whitener = right_whitener.to(weight.device, weight.dtype)
    right_whitener_inv_t = right_whitener_inv_t.to(weight.device, weight.dtype)

    if ldict is not None and layer_name in ldict:
        left_whitener, left_whitener_inv_t = ldict[layer_name]
        left_whitener = left_whitener.to(weight.device, weight.dtype)
        left_whitener_inv_t = left_whitener_inv_t.to(weight.device, weight.dtype)
    else:
        left_whitener = torch.eye(out_features, device=weight.device, dtype=weight.dtype)
        left_whitener_inv_t = torch.eye(out_features, device=weight.device, dtype=weight.dtype)

    combined_right = ((1.0 - alpha) * sigma_hh + alpha * sigma_hhp).to(weight.dtype)
    combined_right = combined_right @ right_whitener_inv_t
    matrix_star = (left_whitener @ weight) @ combined_right
    u, singular_values, vt = torch.linalg.svd(matrix_star, full_matrices=False)
    u_k, singular_values_k, vt_k = u[:, :rank], singular_values[:rank], vt[:rank, :]
    sqrt_s = torch.sqrt(singular_values_k)
    left_factor = (u_k * sqrt_s.unsqueeze(0)).float()
    right_factor = (sqrt_s.unsqueeze(1) * vt_k).float()
    right_factor = (right_factor @ right_whitener_inv_t.T.to(right_factor.dtype)).to(weight.dtype)
    left_factor = (left_whitener_inv_t.to(left_factor.dtype) @ left_factor).to(weight.dtype)

    replacement = nn.Sequential(
        nn.Linear(in_features, rank, bias=False),
        nn.Linear(rank, out_features, bias=original_layer.bias is not None),
    ).to(device)
    replacement[0].weight.data.copy_(right_factor)
    replacement[1].weight.data.copy_(left_factor)
    if original_layer.bias is not None:
        replacement[1].bias.data.copy_(original_layer.bias.data)
    set_module_by_name(student, layer_name, replacement)


def iter_compressible_layers(
    linear_layers: Iterable[Tuple[str, nn.Linear]], ranks: Dict[str, Any]
) -> Iterable[Tuple[str, nn.Linear, int]]:
    """Yield layers for which the rank file selects an actual low-rank or zero-rank replacement."""
    for layer_name, linear in linear_layers:
        rank = ranks.get(layer_name)
        if rank is None:
            continue
        rank = int(rank)
        if rank < 0:
            raise ValueError(f"{layer_name}: rank must be non-negative, got {rank}")
        if rank == 0 or rank * (linear.in_features + linear.out_features) < linear.in_features * linear.out_features:
            yield layer_name, linear, rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the first MySVD_origin layers and compare them to a saved compressed model."
    )
    parser.add_argument(
        "--reference-model",
        type=Path,
        default=Path(
            "compression/compressed_ft_models/fluids.incompressible.PiecewiseConstants/"
            "0.2/MySVD/l1_order2_alpha0.7/compressed_model.pt"
        ),
        help="The compressed_model.pt to validate against.",
    )
    parser.add_argument(
        "--rank-path",
        type=Path,
        default=Path(
            "rank_results/Poseidon-L/fluids.incompressible.PiecewiseConstants/"
            "MySVD_Rank/traj64_order2_l1/rank_0.2_zero.json"
        ),
        help="The rank JSON used to create the reference model.",
    )
    parser.add_argument(
        "--fisher-path",
        type=Path,
        default=Path(
            "MySVD/fisher_cache/fluids.incompressible.PiecewiseConstants/"
            "traj64_order2_l1/Ldict.pt"
        ),
        help="Fisher Ldict used to create the reference model. Ignored with --no-fisher.",
    )
    parser.add_argument("--no-fisher", action="store_true", help="Replay without Fisher left whitening.")
    parser.add_argument("--model-name", default="Poseidon-L")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dataset", default="fluids.incompressible.PiecewiseConstants")
    parser.add_argument("--traj-num", type=int, default=64)
    parser.add_argument("--final-time", type=int, default=20)
    parser.add_argument("--time-back", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu (CPU will be slow).")
    parser.add_argument("--batch-size", type=int, default=512, help="Keep 512 to follow MySVD_origin exactly.")
    parser.add_argument("--max-layers", type=int, default=5, help="Number of actual compressed layers to validate.")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--report-path", type=Path, default=None, help="Optional path for a JSON report."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_layers <= 0:
        raise ValueError("--max-layers must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.time_back < 0 or args.time_back >= args.final_time:
        raise ValueError("--time-back must be in [0, final-time)")

    reference_path = args.reference_model.resolve()
    rank_path = args.rank_path.resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference model not found: {reference_path}")
    if not rank_path.is_file():
        raise FileNotFoundError(f"Rank file not found: {rank_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu if necessary.")
    target_time = args.final_time - args.time_back
    model_path = args.model_path or Path(
        f"checkpoints/{args.model_name}/no_phyloss/{args.dataset}/origin/none"
    )

    print(f"Loading reference model: {reference_path}")
    reference_model = unwrap_model(torch_load(reference_path, map_location="cpu"), reference_path).cpu().eval()
    ranks = load_ranks(rank_path)
    print(f"Loaded {len(ranks)} rank entries from: {rank_path}")

    ldict = None
    if not args.no_fisher:
        fisher_path = args.fisher_path.resolve()
        if not fisher_path.is_file():
            raise FileNotFoundError(
                f"Fisher cache not found: {fisher_path}. Pass --fisher-path or --no-fisher."
            )
        ldict = torch_load(fisher_path, map_location="cpu")
        if not isinstance(ldict, dict):
            raise TypeError(f"Fisher cache is not a dictionary: {fisher_path}")
        print(f"Loaded Fisher cache: {fisher_path}")

    load_project_dependencies()
    print(f"Loading calibration dataset: {args.dataset} (traj_num={args.traj_num})")
    calibration_dataset = get_dataset(
        dataset=args.dataset,
        which="train",
        num_trajectories=args.traj_num,
        data_path="poseidon_main/camlab-ethz/down_streams",
        fix_input_to_time_step=None,
        time_step_size=1,
        max_num_time_steps=target_time,
    )
    trainer = get_trainer(model_path=str(model_path), batch_size=256, dataset=calibration_dataset)
    trainer.set_ar_steps(None, output_all_steps=False)
    teacher = trainer.model.to(device).eval()
    replayed_model = copy.deepcopy(teacher).to(device).eval()

    linear_layers = [
        (name, module) for name, module in teacher.named_modules() if isinstance(module, nn.Linear)
    ]
    selected_layers = list(iter_compressible_layers(linear_layers, ranks))[: args.max_layers]
    if not selected_layers:
        raise RuntimeError("No compressible nn.Linear layers were selected by the rank file.")
    if len(selected_layers) < args.max_layers:
        print(
            f"Warning: rank file contains only {len(selected_layers)} compressible layers; "
            f"validating all of them."
        )

    print("\nReplaying MySVD_origin and comparing effective weights")
    print("-" * 116)
    print(f"{'#':>2}  {'layer':<72} {'rank':>9} {'max|ΔW|':>12} {'relative':>10}  result")
    print("-" * 116)
    reports = []
    for index, (layer_name, original_layer, rank) in enumerate(selected_layers, start=1):
        if rank == 0:
            replayed_layer = get_module_by_name(replayed_model, layer_name)
            if not isinstance(replayed_layer, nn.Linear):
                raise TypeError(f"{layer_name} is no longer nn.Linear before zero-rank replacement")
            replayed_layer.weight.data.zero_()
            if original_layer.bias is not None and replayed_layer.bias is not None:
                replayed_layer.bias.data.copy_(original_layer.bias.data)
        else:
            compress_one_layer_like_origin(
                trainer=trainer,
                teacher=teacher,
                student=replayed_model,
                layer_name=layer_name,
                original_layer=original_layer,
                rank=rank,
                calibration_dataset=calibration_dataset,
                ldict=ldict,
                alpha=args.alpha,
                device=device,
                batch_size=args.batch_size,
            )

        replayed_layer = get_module_by_name(replayed_model, layer_name)
        try:
            reference_layer = get_module_by_name(reference_model, layer_name)
        except AttributeError as error:
            raise AttributeError(
                f"Layer {layer_name!r} does not exist in the reference model. "
                "Check --rank-path and --reference-model describe the same run."
            ) from error
        report = compare_layer(
            layer_name, replayed_layer, reference_layer, atol=args.atol, rtol=args.rtol
        )
        report["requested_rank"] = rank
        reports.append(report)
        weight_error = report["weight_max_abs"]
        relative_error = report["weight_max_rel"]
        rank_label = (
            f"{report['rank_replayed']}/{report['rank_reference']}"
            if report["rank_replayed"] is not None
            else "zero"
        )
        result = "PASS" if report["passed"] else "FAIL"
        print(
            f"{index:>2}  {layer_name:<72.72} {rank_label:>9} "
            f"{weight_error if weight_error is not None else float('nan'):>12.4e} "
            f"{relative_error if relative_error is not None else float('nan'):>10.3e}  {result}"
        )

    passed = all(report["passed"] for report in reports)
    print("-" * 116)
    print(
        f"Validation {'PASSED' if passed else 'FAILED'}: "
        f"{sum(report['passed'] for report in reports)}/{len(reports)} layers "
        f"within atol={args.atol:g}, rtol={args.rtol:g}."
    )
    print(
        "Note: factor_right_max_abs/factor_left_max_abs in the JSON report are diagnostics only; "
        "SVD factor signs may differ while the effective weight is identical."
    )

    if args.report_path is not None:
        report_path = args.report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "reference_model": str(reference_path),
                    "rank_path": str(rank_path),
                    "fisher_path": None if args.no_fisher else str(args.fisher_path.resolve()),
                    "alpha": args.alpha,
                    "atol": args.atol,
                    "rtol": args.rtol,
                    "passed": passed,
                    "layers": reports,
                },
                handle,
                indent=2,
            )
        print(f"JSON report written to: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
