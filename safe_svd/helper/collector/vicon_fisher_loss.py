"""VICON masked Sobolev objective copied from MySVD/helper/fisher_collector.py."""

from __future__ import annotations

import torch
import torch.nn.functional as F

def compute_sobolev_loss_vicon(
    pred, label, c_mask, type_mask,
    order=1, loss_func=F.l1_loss,
):
    interior = (type_mask == 0).float()
    valid_mask = interior * c_mask.float()

    value_loss = _masked_loss(pred, label, valid_mask, loss_func)
    total_loss = value_loss

    if order >= 1:
        dx_pred  = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        dx_label = label[:, :, :, :, 1:] - label[:, :, :, :, :-1]
        dy_pred  = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        dy_label = label[:, :, :, 1:, :] - label[:, :, :, :-1, :]

        dx_interior = torch.minimum(
            interior[:, :, :, :, 1:], interior[:, :, :, :, :-1])
        dx_mask = dx_interior * c_mask.float()
        dy_interior = torch.minimum(
            interior[:, :, :, 1:, :], interior[:, :, :, :-1, :])
        dy_mask = dy_interior * c_mask.float()

        grad_loss_x = _masked_loss(dx_pred, dx_label, dx_mask, loss_func)
        grad_loss_y = _masked_loss(dy_pred, dy_label, dy_mask, loss_func)
        total_loss = total_loss + grad_loss_x + grad_loss_y

    if order >= 2:
        d2x_pred  = dx_pred[:, :, :, :, 1:] - dx_pred[:, :, :, :, :-1]
        d2x_label = dx_label[:, :, :, :, 1:] - dx_label[:, :, :, :, :-1]
        d2y_pred  = dy_pred[:, :, :, 1:, :] - dy_pred[:, :, :, :-1, :]
        d2y_label = dy_label[:, :, :, 1:, :] - dy_label[:, :, :, :-1, :]

        d2x_interior = torch.minimum(
            dx_interior[:, :, :, :, 1:], dx_interior[:, :, :, :, :-1])
        d2x_mask = d2x_interior * c_mask.float()
        d2y_interior = torch.minimum(
            dy_interior[:, :, :, 1:, :], dy_interior[:, :, :, :-1, :])
        d2y_mask = d2y_interior * c_mask.float()

        grad2_loss_x = _masked_loss(d2x_pred, d2x_label, d2x_mask, loss_func)
        grad2_loss_y = _masked_loss(d2y_pred, d2y_label, d2y_mask, loss_func)
        total_loss = total_loss + grad2_loss_x + grad2_loss_y

    return total_loss


def _masked_loss(pred, label, mask, loss_func):
    pixel_loss = loss_func(pred, label, reduction='none')
    masked = pixel_loss * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def sobolev_loss_for_fisher(
    trainer, data_preprocessed,
    order=1, balance_ratio=1.0, loss_func=F.l1_loss,
):
    min_ex = trainer.loss_cfg.min_ex
    init_norm, end_norm, c_mask_expanded = data_preprocessed

    label = end_norm[:, min_ex:, :-1]
    c_mask = c_mask_expanded[:, :, :-1]
    type_mask = end_norm[:, min_ex:, -1:]

    output = trainer._model_forward(data_preprocessed)
    pred = output[:, min_ex:, :-1]

    interior = (type_mask == 0).float()
    valid_mask = interior * c_mask.float()
    base_loss = _masked_loss(pred, label, valid_mask, F.mse_loss)

    if order >= 1:
        sob_loss = compute_sobolev_loss_vicon(
            pred, label, c_mask, type_mask,
            order=order, loss_func=loss_func,
        )
        grad_only = sob_loss - _masked_loss(pred, label, valid_mask, loss_func)
        total_loss = base_loss + balance_ratio * grad_only
    else:
        total_loss = base_loss

    return total_loss

