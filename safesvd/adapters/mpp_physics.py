"""MPP-AViT evaluation metrics kept independent of the legacy sweep script."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


SOBOLEV_ORDERS = {"incompNS": 2, "compNS": 1, "swe": 1, "diffre2d": 2}


def sobolev_metric(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    order: int,
    dx: float = 1 / 128,
    dy: float = 1 / 128,
) -> tuple[float, float]:
    """The finite-difference L1 Sobolev metric used by MPP's evaluator."""
    pred = _numpy(prediction)
    label = _numpy(target)
    total = float(np.mean(np.abs(pred - label)))
    norm = float(np.mean(np.abs(label)))
    predictions = [pred]
    labels = [label]
    for current_order in range(1, order + 1):
        next_predictions: list[np.ndarray] = []
        next_labels: list[np.ndarray] = []
        error = reference = 0.0
        for current_prediction, current_label in zip(predictions, labels):
            pred_x = (current_prediction[..., :, 1:] - current_prediction[..., :, :-1]) / dx
            label_x = (current_label[..., :, 1:] - current_label[..., :, :-1]) / dx
            pred_y = (current_prediction[..., 1:, :] - current_prediction[..., :-1, :]) / dy
            label_y = (current_label[..., 1:, :] - current_label[..., :-1, :]) / dy
            error += float(np.mean(np.abs(pred_x - label_x)) + np.mean(np.abs(pred_y - label_y)))
            reference += float(np.mean(np.abs(label_x)) + np.mean(np.abs(label_y)))
            if current_order < order:
                next_predictions.extend((pred_x, pred_y))
                next_labels.extend((label_x, label_y))
        total += error
        norm += reference
        predictions, labels = next_predictions, next_labels
    return total, total / (norm + 1e-12)


def physics_metrics(
    category: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    previous: torch.Tensor | None = None,
) -> dict[str, float]:
    """Physical metrics matching ``mpp/eval/eval_all.py`` for one batch."""
    pred = _numpy(prediction)
    label = _numpy(target)
    order = SOBOLEV_ORDERS.get(category, 1)
    sobolev, relative_sobolev = sobolev_metric(pred, label, order=order)
    if category == "incompNS":
        return {
            "sobolev": sobolev,
            "rel_sobolev": relative_sobolev,
            **_incompressible(pred, label),
        }
    if category == "compNS":
        if previous is None:
            raise ValueError("compNS metrics require the final input frame")
        return {
            "sobolev": sobolev,
            "rel_sobolev": relative_sobolev,
            **_compressible(pred, label, _numpy(previous)),
        }
    return {"sobolev": sobolev, "rel_sobolev": relative_sobolev}


def average_metrics(values: list[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    return {key: float(np.mean([value[key] for value in values if key in value])) for key in values[0]}


def _incompressible(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    du_dx = _central_dx(prediction[:, 0])
    dv_dy = _central_dy(prediction[:, 1])
    divergence = float(np.mean(np.abs(du_dx + dv_dy)))
    target_du_dx = _central_dx(target[:, 0])
    target_dv_dy = _central_dy(target[:, 1])
    scale = float(np.mean(np.abs(target_du_dx)) + np.mean(np.abs(target_dv_dy)) + 1e-12)
    curl = _central_dx(prediction[:, 1]) - _central_dy(prediction[:, 0])
    target_curl = _central_dx(target[:, 1]) - _central_dy(target[:, 0])
    vorticity_error = float(np.mean(np.abs(curl - target_curl)))
    return {
        "divergence": divergence,
        "rel_divergence": divergence / scale,
        "vorticity_error": vorticity_error,
        "rel_vorticity_error": vorticity_error / (float(np.mean(np.abs(target_curl))) + 1e-12),
    }


def _compressible(prediction: np.ndarray, target: np.ndarray, previous: np.ndarray) -> dict[str, float]:
    rho, velocity_x, velocity_y, pressure = prediction[:, 2], prediction[:, 0], prediction[:, 1], prediction[:, 3]
    rho_previous, velocity_x_previous, velocity_y_previous, pressure_previous = (
        previous[:, 2], previous[:, 0], previous[:, 1], previous[:, 3]
    )
    gamma = 1.4
    mass_time = rho - rho_previous
    mass_x = _central_dx(rho * velocity_x)
    mass_y = _central_dy(rho * velocity_y)
    mass = float(np.mean(np.abs(mass_time + mass_x + mass_y)))
    mass_scale = float(np.mean(np.abs(mass_time)) + np.mean(np.abs(mass_x)) + np.mean(np.abs(mass_y)) + 1e-12)

    momentum_x_time = rho * velocity_x - rho_previous * velocity_x_previous
    momentum_x_x = _central_dx(rho * velocity_x**2 + pressure)
    momentum_x_y = _central_dy(rho * velocity_x * velocity_y)
    momentum_x = float(np.mean(np.abs(momentum_x_time + momentum_x_x + momentum_x_y)))
    momentum_x_scale = float(
        np.mean(np.abs(momentum_x_time)) + np.mean(np.abs(momentum_x_x)) + np.mean(np.abs(momentum_x_y)) + 1e-12
    )
    momentum_y_time = rho * velocity_y - rho_previous * velocity_y_previous
    momentum_y_x = _central_dx(rho * velocity_x * velocity_y)
    momentum_y_y = _central_dy(rho * velocity_y**2 + pressure)
    momentum_y = float(np.mean(np.abs(momentum_y_time + momentum_y_x + momentum_y_y)))
    momentum_y_scale = float(
        np.mean(np.abs(momentum_y_time)) + np.mean(np.abs(momentum_y_x)) + np.mean(np.abs(momentum_y_y)) + 1e-12
    )

    energy = pressure / (gamma - 1.0) + 0.5 * rho * (velocity_x**2 + velocity_y**2)
    energy_previous = pressure_previous / (gamma - 1.0) + 0.5 * rho_previous * (
        velocity_x_previous**2 + velocity_y_previous**2
    )
    energy_time = energy - energy_previous
    energy_x = _central_dx((energy + pressure) * velocity_x)
    energy_y = _central_dy((energy + pressure) * velocity_y)
    energy_residual = float(np.mean(np.abs(energy_time + energy_x + energy_y)))
    energy_scale = float(np.mean(np.abs(energy_time)) + np.mean(np.abs(energy_x)) + np.mean(np.abs(energy_y)) + 1e-12)
    return {
        "mass_cons": mass,
        "rel_mass_cons": mass / mass_scale,
        "momentum_cons": (momentum_x + momentum_y) / 2,
        "rel_momentum_cons": (momentum_x / momentum_x_scale + momentum_y / momentum_y_scale) / 2,
        "energy_cons": energy_residual,
        "rel_energy_cons": energy_residual / energy_scale,
    }


def _central_dx(value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(value)
    result[..., 1:-1] = (value[..., 2:] - value[..., :-2]) / 2
    result[..., 0] = value[..., 1] - value[..., 0]
    result[..., -1] = value[..., -1] - value[..., -2]
    return result


def _central_dy(value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(value)
    result[..., 1:-1, :] = (value[..., 2:, :] - value[..., :-2, :]) / 2
    result[..., 0, :] = value[..., 1, :] - value[..., 0, :]
    result[..., -1, :] = value[..., -1, :] - value[..., -2, :]
    return result


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    return value.detach().cpu().float().numpy() if isinstance(value, torch.Tensor) else value

