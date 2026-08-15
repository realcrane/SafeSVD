import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from SAFE_SVD.helper.whitener import fisher_cholesky_whitener

def sobolev_loss_calc_denorm(dataset_name, pred, labels, order_control, base_loss_val,
                            loss_func, device, constants,
                            prev_pred=None,
                            prev_labels=None,
                            transpose=False,
                            decay_alpha=1, balance_ratio=0.5):
    """
    修复版: 强制将 labels, mean, std 全部移动到 device，防止 device mismatch 报错。
    """
    # [修复 1] 强制将 labels 搬运到 GPU (如果它还在 CPU)
    # pred 通常已经在 GPU 上，但为了保险也可以 .to(device)
    if labels.device != device:
        labels = labels.to(device)

    if constants is None:
        constants = {"mean": 0.0, "std": 1.0}
    elif "mean" not in constants and "mean_solution" in constants:
        constants = {
            **constants,
            "mean": constants["mean_solution"],
            "std": constants["std_solution"],
        }

    # 1. 准备统计量 (Mean/Std)
    # [修复 2] 确保 mean/std 创建时就在 GPU 上
    if not torch.is_tensor(constants['mean']):
        mean = torch.tensor(constants['mean'], device=device)
    else:
        mean = constants['mean'].to(device)

    if not torch.is_tensor(constants['std']):
        std = torch.tensor(constants['std'], device=device)
    else:
        std = constants['std'].to(device)

    # 调整维度以便广播
    if mean.ndim == 1: mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1: std = std.view(1, -1, 1, 1)

    # 2. 反归一化 (还原物理量级)
    # 此时所有变量都在 device 上，不会报错
    pred_phys = pred * std + mean
    labels_phys = labels * std + mean
    time_loss = 0.0
    if prev_pred is not None and prev_labels is not None:
        prev_pred_phys = prev_pred * std + mean
        prev_labels_phys = prev_labels * std + mean
        pred_time_diff = pred_phys - prev_pred_phys
        labels_time_diff = labels_phys - prev_labels_phys
        time_loss = loss_func(pred_time_diff, labels_time_diff)

    C = pred_phys.shape[1]
    if "incompressible" in dataset_name:
        if C == 3:
            indices = [0, 1, 2] # [u, v, p]
        else:
            indices = [1, 2, 3] # [rho, u, v, p]
    elif "compressible" in dataset_name:
        indices = [0, 1, 2, 3]
    elif "wave" in dataset_name.lower() or "Wave" in dataset_name:
        indices = list(range(C))
    elif "elliptic" in dataset_name.lower() or "helmholtz" in dataset_name.lower() or "poisson" in dataset_name.lower():
        indices = list(range(C))
    else:
        indices = list(range(C))
    if not transpose:
        pred_phys =  pred_phys.transpose(-2, -1)
        labels_phys =  labels_phys.transpose(-2, -1)
    # [修复 3] 确保 pred_uv 和 target_uv 是连续的内存 (可选，但推荐)
    pred_uv = pred_phys[:, indices, ...].contiguous()
    target_uv = labels_phys[:, indices, ...].contiguous()

    # 初始化循环变量
    current_preds = [pred_uv]
    current_targets = [target_uv]
    total_weighted_grad_loss = 0.0

    # Order 0: 函数值本身的 loss (反归一化物理空间)
    order0_loss = loss_func(pred_uv, target_uv)
    order0_loss_val = order0_loss.item() if isinstance(order0_loss, torch.Tensor) else order0_loss
    if order0_loss_val > 1e-20:
        norm_weight_0 = (base_loss_val / order0_loss_val) * balance_ratio
    else:
        norm_weight_0 = 0.0
    norm_weight_0 = min(norm_weight_0, 1000.0)
    total_weighted_grad_loss += norm_weight_0 * order0_loss

    # 4. 循环计算导数
    for k in range(1, order_control + 1):
        next_preds = []
        next_targets = []
        current_order_loss = 0.0

        for p, t in zip(current_preds, current_targets):
            # 计算梯度
            # 这里的切片操作后的 tensor 依然在 device 上
            dx_p = (p[..., 1:] - p[..., :-1])
            dx_t = (t[..., 1:] - t[..., :-1])
            dy_p = (p[..., 1:, :] - p[..., :-1, :])
            dy_t = (t[..., 1:, :] - t[..., :-1, :])

            # 累加 Loss
            current_order_loss += loss_func(dx_p, dx_t) + loss_func(dy_p, dy_t)

            if k < order_control:
                next_preds.extend([dx_p, dy_p])
                next_targets.extend([dx_t, dy_t])

        order_loss_val = current_order_loss.item() if isinstance(current_order_loss, torch.Tensor) else current_order_loss

        if order_loss_val > 1e-20:
            norm_weight = (base_loss_val / order_loss_val) * balance_ratio
        else:
            norm_weight = 0.0

        # 限制权重上限
        norm_weight = min(norm_weight, 1000.0)

        decay = decay_alpha ** (k - 1)
        final_lambda_k = norm_weight * decay

        total_weighted_grad_loss += final_lambda_k * current_order_loss

        current_preds = next_preds
        current_targets = next_targets

    if isinstance(time_loss, torch.Tensor):
        time_loss_val = time_loss.item()
        if time_loss_val > 1e-20:
            time_weight = (base_loss_val / time_loss_val) * balance_ratio
        else:
            time_weight = 0.0
        time_weight = min(time_weight, 1000.0)
        total_weighted_grad_loss += time_weight * time_loss

    return total_weighted_grad_loss


def collect_fisher_dict_linear_only(trainer, dataset_name, calib_loader,
                                    device, gradient_loss="mse",
                                    care_I=False,
                                    order_control=0, constants=None,
                                    transpose=False, balance_ratio=0.5):
    """
    Collect Fisher Information Matrix for Linear Layers.
    Includes fixes for:
    1. Automatic dimension matching (4D inputs).
    2. Disabling in-place operations to prevent BackwardHook errors.
    """
    if gradient_loss == 'l1':
        loss_func = F.l1_loss
    elif gradient_loss == 'mse':
        loss_func = F.mse_loss
    elif gradient_loss == 'smooth_l1':
        loss_func = F.smooth_l1_loss
    else:
        raise ValueError(f"Unsupported gradient_loss type: {gradient_loss}")

    print("Collecting Fisher Information Matrix (Linear Layers Only)...")

    # 1. Prepare Model
    model = trainer.model.to(device).train() # Must be in train mode for backward
    model.zero_grad()

    # =========================================================
    # [Fix] Disable In-place Operations
    # This prevents "Output 0 of BackwardHookFunctionBackward is a view..." error
    # caused by ReLU(inplace=True) modifying outputs needed by hooks.
    # =========================================================
    print("Disabling in-place operations for gradient collection...")
    def disable_inplace(m):
        if hasattr(m, 'inplace'):
            m.inplace = False

    model.apply(disable_inplace)

    cov_dict = {}
    count_dict = {}

    # --- Hook Definition ---
    def get_fisher_hook(name):
        def hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient w.r.t layer output
            g = grad_output[0].detach()

            # Flatten all dimensions (Batch, Time, H, W) -> [N, Out_Dim]
            if g.dim() > 2:
                g = g.reshape(-1, g.shape[-1])

            # Calculate g^T * g (Approximated Fisher)
            cov = g.t() @ g

            if name not in cov_dict:
                cov_dict[name] = cov
                count_dict[name] = g.shape[0]
            else:
                cov_dict[name] += cov
                count_dict[name] += g.shape[0]
        return hook

    # --- Register Hooks ---
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_full_backward_hook(get_fisher_hook(name)))

    # --- Run Data Loop ---
    for i, batch in enumerate(tqdm(calib_loader)):
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # 2. Fix Dimensions (Auto-fix for 3D/5D inputs)
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if pv.dim() == 3: # [C, H, W] -> [1, C, H, W]
                inputs["pixel_values"] = pv.unsqueeze(0)
            elif pv.dim() == 5: # [B, T, C, H, W] -> [B, C, H, W] (Take 1st frame)
                inputs["pixel_values"] = pv[:, 0, ...]
        # 3. Forward
        try:
            outputs = model(**inputs)
        except RuntimeError as e:
            print(f"[Error] Batch {i} failed. Input shape: {inputs.get('pixel_values', 'N/A').shape}")
            raise e
        # 4. Get Loss
        loss = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        elif isinstance(outputs, (list, tuple)):
            loss = outputs[0]
        if loss is None:
            print("loss is none")
            continue
        base_loss_val = loss.item()
        # ==== Calculate Gradient Loss ====
        labels = batch["labels"]
        pred = outputs.output
        # grad_loss = sobolev_loss_calc(pred, labels, order_control, base_loss_val, loss_func, device)
        grad_loss = sobolev_loss_calc_denorm(dataset_name, pred, labels, order_control, base_loss_val, loss_func, device, constants=constants, transpose=transpose, balance_ratio=balance_ratio)
        # ==================================
        loss = loss + grad_loss
        # 5. Backward
        model.zero_grad()
        loss.backward()
        if i == 0: # 只看第一个 batch
            print(f"Current Batch Loss: {loss.item()}")
            for n, p in model.named_parameters():
                if "layers.0.blocks.0.attention.self.query" in n and p.grad is not None:
                    print(f"Gradient Norm for {n}: {p.grad.norm().item()}")
        del outputs, loss

    # --- Post-process (Cholesky) ---
    Ldict = {}
    print("Computing Cholesky decomposition with Trace Normalization...")
    for name, cov in cov_dict.items():
        Lt, Lt_inv = fisher_cholesky_whitener(cov, count_dict[name])
        Ldict[name] = (Lt.cpu(), Lt_inv.cpu())
        del Lt, Lt_inv
        torch.cuda.empty_cache()

    for h in handles: h.remove()
    model.eval()

    return Ldict
