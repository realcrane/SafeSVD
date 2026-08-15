#coding:utf8
"""
Sobolev Loss for VICON FWSVD Fisher Collection.

【改动说明】
  collect_mysvd_fisher_vicon 新增 use_trace_align 超参:
    True  → 按 dataset_type 分桶收集，Cholesky 前 trace 对齐
    False → 按 dataset_type 分桶收集，Cholesky 前简单平均 (无缩放)
"""
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Tuple

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


# =====================================================================
# Core: Full-matrix Fisher collection
# =====================================================================

def collect_mysvd_fisher_vicon(
    trainer,
    calib_loaders: dict,
    dev: str = "cuda",
    max_calib_batches: int = None,
    gradient_loss: str = "l1",
    balance_ratio: float = 10.0,
    care_I: bool = False,
    use_gradient_checkpointing: bool = True,
    use_trace_align: bool = True,        # ← 新增超参
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Collect full-matrix Fisher information for MySVD on VICON.

    use_trace_align:
        True  → 按 dataset_type 分桶收集，合并前 trace 对齐
        False → 按 dataset_type 分桶收集，合并时简单平均 (无缩放)
    """
    from MySVD.helper.input_collector import _patched_forward

    loss_func = {
        "l1": F.l1_loss,
        "mse": F.mse_loss,
        "smooth_l1": F.smooth_l1_loss,
    }.get(gradient_loss)
    if loss_func is None:
        raise ValueError(f"Unsupported gradient_loss: {gradient_loss}")

    align_tag = "ON" if use_trace_align else "OFF"
    print(f"Collecting Full-Matrix Fisher for MySVD (VICON)")
    print(f"  gradient_loss={gradient_loss}, "
          f"balance_ratio={balance_ratio}, care_I={care_I}, trace_align={align_tag}")

    device = torch.device(dev)
    model = trainer.model.to(device).train()
    model.zero_grad()

    def disable_inplace(m):
        if hasattr(m, 'inplace'):
            m.inplace = False
    model.apply(disable_inplace)

    # 按 (layer_name, dataset_type) 分桶
    cov_dict:   Dict[str, Dict[str, torch.Tensor]] = {}
    count_dict: Dict[str, Dict[str, int]] = {}

    def get_fisher_hook(name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0].detach()
            if g.dim() > 2:
                g = g.reshape(-1, g.shape[-1])
            cov = g.t() @ g

            cat = getattr(model, '_current_dataset_type', 'default')

            if name not in cov_dict:
                cov_dict[name] = {}
                count_dict[name] = {}

            if cat not in cov_dict[name]:
                cov_dict[name][cat] = cov
                count_dict[name][cat] = g.shape[0]
            else:
                cov_dict[name][cat] += cov
                count_dict[name][cat] += g.shape[0]
        return hook

    handles = []
    hook_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_full_backward_hook(get_fisher_hook(name)))
            hook_count += 1
    print(f"  Backward hooks registered on {hook_count} Linear layers.")

    torch.backends.cudnn.benchmark = False

    total_batches = 0
    sobolev_order_map = {
        "NS2D": 2,
        "COMPRESSIBLE2D": 1,
        "EULER2D": 1
    }
    with _patched_forward(model, use_gradient_checkpointing=use_gradient_checkpointing):
        for dataset_type, calib_loader in calib_loaders.items():
            print(f"  -> Dataset: {dataset_type}")
            model._current_dataset_type = dataset_type
            total = max_calib_batches if max_calib_batches else len(calib_loader)
            current_order = sobolev_order_map.get(dataset_type, 1)
            print(f"     Applying Sobolev Order: {current_order}")
            for batch_cnt, batch in enumerate(tqdm(
                calib_loader, total=total,
                desc=f"MySVD Fisher [{dataset_type}]"
            )):
                if max_calib_batches is not None and batch_cnt >= max_calib_batches:
                    break

                try:
                    _, pairs, *_ = batch
                    pairs = trainer._move_to_device(pairs)
                    data, mean, std = trainer._data_preprocess(pairs)

                    loss = sobolev_loss_for_fisher(
                        trainer, data,
                        order=current_order,
                        balance_ratio=balance_ratio,
                        loss_func=loss_func,
                    )

                    model.zero_grad(set_to_none=True)
                    loss.backward()
                    total_batches += 1

                    if batch_cnt == 0:
                        print(f"    First batch loss: {loss.item():.6f}")

                    del loss, pairs, data
                    if batch_cnt % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  Batch {batch_cnt} OOM, skipping...")
                        model.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"  Batch {batch_cnt} error: {e}")
                        continue

    for h in handles:
        h.remove()
    print(f"  Total batches processed: {total_batches}")

    # =====================================================================
    # Post-process: 按类别取平均 → (可选 trace 对齐) → 合并 → Cholesky
    # =====================================================================
    print("Computing Cholesky decomposition ...")
    Ldict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for name, cat_covs in tqdm(cov_dict.items(), desc="Cholesky"):
        try:
            per_cat_avg = {}
            for cat in cat_covs:
                N = count_dict[name].get(cat, 0)
                if N > 0:
                    per_cat_avg[cat] = (cat_covs[cat] / N).double()

            if len(per_cat_avg) == 0:
                raise ValueError("No valid gradients collected.")

            cats = list(per_cat_avg.keys())
            if use_trace_align and len(cats) > 1:
                # --- Partial trace align: 只在相同 sobolev order 的 dataset 之间对齐 ---
                order_groups: Dict[int, list] = {}
                for c in cats:
                    o = sobolev_order_map.get(c, 1)
                    order_groups.setdefault(o, []).append(c)

                balanced = []
                for o, group_cats in order_groups.items():
                    if len(group_cats) > 1:
                        # 组内 trace 对齐
                        traces = {c: torch.trace(per_cat_avg[c]).item() for c in group_cats}
                        target_trace = max(traces.values())
                        for c in group_cats:
                            tr = traces[c]
                            if tr > 1e-15:
                                balanced.append(per_cat_avg[c] * (target_trace / tr))
                            else:
                                balanced.append(per_cat_avg[c])
                    else:
                        # 该 order 组只有一个 dataset，直接保留
                        balanced.append(per_cat_avg[group_cats[0]])
                Sigma = sum(balanced) / len(balanced)
            else:
                Sigma = sum(per_cat_avg[c] for c in cats) / len(cats)

            dim = Sigma.shape[0]
            diag_mean = torch.diagonal(Sigma).mean().item()

            if care_I:
                scale = (1.0 / diag_mean) if diag_mean > 0 else 1.0
                Sigma_final = (Sigma * scale) + torch.eye(dim, device=device, dtype=torch.float64)
            else:
                jitter_val = max(diag_mean * 0.01, 1e-18) if diag_mean > 0 else 1e-9
                Sigma_final = Sigma + torch.eye(dim, device=device, dtype=torch.float64) * jitter_val

            try:
                L_mat = torch.linalg.cholesky(Sigma_final)
            except RuntimeError:
                print(f"  [Warning] Cholesky failed for {name}, switching to eigh...")
                eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_final)
                eigenvalues = torch.clamp(eigenvalues, min=1e-18)
                S_sqrt = torch.sqrt(eigenvalues)
                L_mat = eigenvectors * S_sqrt.unsqueeze(0)

            L_mat = L_mat.float()
            Lt = L_mat.t()
            try:
                Lt_inv = torch.linalg.inv(Lt)
            except Exception:
                Lt_inv = torch.linalg.pinv(Lt)

            Ldict[name] = (Lt.cpu(), Lt_inv.cpu())
            del Sigma, Sigma_final, L_mat, Lt, Lt_inv
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [Error] {name}: {e}, using identity fallback")
            any_cov = next(iter(cat_covs.values()))
            dim = any_cov.shape[0]
            I_cpu = torch.eye(dim)
            Ldict[name] = (I_cpu, I_cpu)

    model.eval()
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"MySVD Full-Matrix Fisher collected: {len(Ldict)} layers.")
    return Ldict