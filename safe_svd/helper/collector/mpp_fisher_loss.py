"""MPP physical loss copied from MySVD/mysvd.py."""

from __future__ import annotations

import torch
import torch.nn.functional as F

DSET_SOBOLEV_CONFIG = {
    'incompNS':  {'mode': 'spatial_only',   'has_divergence': True,
                  'vel_indices': [0, 1], 'order': 2},             # Laplacian present
    'compNS':    {'mode': 'spatial_only', 'has_divergence': False,
                  'vel_indices': None, 'order': 1},               # Mass/momentum flux
    'swe':       {'mode': 'spatial_only',   'has_divergence': False,
                  'vel_indices': None, 'order': 1},               # Hyperbolic conservation
    'diffre2d':  {'mode': 'spatial_only',   'has_divergence': False,
                  'vel_indices': None, 'order': 2},               # Diffusion driven
}


def sobolev_loss_mpp(dset_name, pred, tar, prev_frame, loss_func,
                     order_control=1, balance_ratio=0.5, decay_alpha=1.0,
                     base_loss_val=1.0):
    cfg = DSET_SOBOLEV_CONFIG.get(dset_name, {'mode': 'spatial_only',
                                               'has_divergence': False,
                                               'vel_indices': None})
    total_sob_loss = torch.tensor(0.0, device=pred.device)

    # ---- 1. Spatial Sobolev (all datasets) ----
    current_preds  = [pred]
    current_targets = [tar]

    for k in range(1, order_control + 1):
        next_preds, next_targets = [], []
        current_order_loss = torch.tensor(0.0, device=pred.device)

        for p, t in zip(current_preds, current_targets):
            dx_p = p[..., 1:] - p[..., :-1]
            dx_t = t[..., 1:] - t[..., :-1]
            dy_p = p[..., 1:, :] - p[..., :-1, :]
            dy_t = t[..., 1:, :] - t[..., :-1, :]

            current_order_loss = current_order_loss + loss_func(dx_p, dx_t) + loss_func(dy_p, dy_t)

            if k < order_control:
                next_preds.extend([dx_p, dy_p])
                next_targets.extend([dx_t, dy_t])

        order_val = current_order_loss.item() if isinstance(current_order_loss, torch.Tensor) else current_order_loss
        if order_val > 1e-20:
            norm_weight = (base_loss_val / order_val) * balance_ratio
        else:
            norm_weight = 0.0
        norm_weight = min(norm_weight, 1000.0)
        decay = decay_alpha ** (k - 1)
        total_sob_loss = total_sob_loss + norm_weight * decay * current_order_loss

        current_preds = next_preds
        current_targets = next_targets

    # ---- 2. Divergence term for incompNS ----
    if cfg['has_divergence'] and cfg['vel_indices'] is not None:
        u_idx, v_idx = cfg['vel_indices']
        du_dx_p = pred[:, u_idx, :, 1:] - pred[:, u_idx, :, :-1]
        dv_dy_p = pred[:, v_idx, 1:, :] - pred[:, v_idx, :-1, :]
        du_dx_t = tar[:, u_idx, :, 1:]  - tar[:, u_idx, :, :-1]
        dv_dy_t = tar[:, v_idx, 1:, :]  - tar[:, v_idx, :-1, :]
        min_h = min(du_dx_p.shape[-2], dv_dy_p.shape[-2])
        min_w = min(du_dx_p.shape[-1], dv_dy_p.shape[-1])
        div_pred = du_dx_p[..., :min_h, :min_w] + dv_dy_p[..., :min_h, :min_w]
        div_gt   = du_dx_t[..., :min_h, :min_w] + dv_dy_t[..., :min_h, :min_w]
        div_loss = loss_func(div_pred, div_gt)
        div_val = div_loss.item()
        if div_val > 1e-20:
            div_weight = min((base_loss_val / div_val) * balance_ratio, 1000.0)
        else:
            div_weight = 0.0
        total_sob_loss = total_sob_loss + div_weight * div_loss

    # ---- 3. Temporal Sobolev (compNS only) ----
    if cfg['mode'] == 'spatiotemporal' and prev_frame is not None:
        pred_time_diff = pred - prev_frame
        tar_time_diff  = tar  - prev_frame
        time_loss = loss_func(pred_time_diff, tar_time_diff)
        time_val = time_loss.item()
        if time_val > 1e-20:
            time_weight = min((base_loss_val / time_val) * balance_ratio, 1000.0)
        else:
            time_weight = 0.0
        total_sob_loss = total_sob_loss + time_weight * time_loss

    return total_sob_loss

