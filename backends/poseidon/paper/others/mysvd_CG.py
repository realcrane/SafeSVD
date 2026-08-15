import torch
import torch.nn as nn
from torch.func import jvp, vjp
import copy

# Import your existing tools
from poseidon_main.scOT.inference import get_trainer, rollout
from poseidon_main.scOT.problems.base import get_dataset
from MySVD.helper.input_collector import get_module_by_name

# ==========================================
# 0. Profile Inputs (Reuse your existing function)
# ==========================================
@torch.no_grad()
def profile_inputs_physics(trainer, calib_loader, dev, flatten: bool = True, to_cpu: bool = True):
    """
    基于 Trainer + rollout，收集每个 nn.Linear 层的输入激活 X。
    - 不再计算 X^T X 和 Cholesky
    - 默认把每个 batch 的输入展平成 [N, in_features] 并搬到 CPU
    返回: Dict[str, Tensor 或 List[Tensor]]，键是 layer 的名字
    """
    device = torch.device(dev)
    model = trainer.model.to(device).eval()

    handles = []
    names, modules = [], []

    # 用 pre-hook 抓输入
    def make_pre_hook(layer_name: str):
        def _pre(module, inp):
            x = inp[0].detach()
            if isinstance(x, (list, tuple)):
                x = x[0]
            if flatten:
                x = x.reshape(-1, x.shape[-1])  # [N, C]
            if to_cpu:
                x = x.float().cpu()
            module._inputs_bucket.append(x)
        return _pre

    # 注册所有 Linear
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._inputs_bucket = []
            handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
            names.append(name)
            modules.append(module)

    print("Start gathering inputs via rollout…")
    _ = rollout(trainer, calib_loader, model_path=getattr(trainer.model.config, 'name_or_path', ''))

    # 卸载 hook + 组装返回
    for h in handles:
        h.remove()

    inputs_by_layer = {}
    for name, module in zip(names, modules):
        xs = module._inputs_bucket
        if len(xs) == 0:
            continue
        try:
            inputs_by_layer[name] = torch.cat(xs, dim=0)  # [N, C]
        except Exception:
            # 维度不齐时保留为 list
            inputs_by_layer[name] = xs
        del module._inputs_bucket

    model = model.cpu()
    torch.cuda.empty_cache()
    return inputs_by_layer

# ==========================================
# 1. Core Experiment Function
# ==========================================
def experiment_cg_on_layer(trainer, layer_name, acts):
    print(f"\n{'='*60}")
    print(f"🧪 Experimenting CG on Layer: {layer_name}")
    print(f"{'='*60}")
    
    # Move model to device and use double precision for CG
    model = trainer.model.eval().to(device).double()
    
    # --- A. Retrieve H from acts ---
    if layer_name not in acts:
        print(f"❌ Layer {layer_name} not found in captured acts!")
        return

    # Move H to GPU and convert to double
    H = acts[layer_name].to(device).double()
    
    # Optional: Truncate H if it's too large for quick testing
    # H = H[:100, :] 
    print(f"✅ Captured H shape: {H.shape}")

    # --- B. Whitening Prep ---
    target_layer = get_module_by_name(model, layer_name)
    if not isinstance(target_layer, nn.Linear):
        print(f"Skipping {layer_name}: Not a Linear layer")
        return

    print("Step 2: Preparing Whitening Matrix...")
    W_original = target_layer.weight.data.double() 
    bias_original = target_layer.bias.data.double() if target_layer.bias is not None else None
    
    # Compute Lp
    cov = H.T @ H + 1e-6 * torch.eye(H.shape[1], device=device, dtype=torch.double)
    Lp = torch.eye(H.shape[1], device=device, dtype=torch.double)
    Lp_inv = Lp # Identity inverse is Identity
    # Lp = torch.linalg.cholesky(cov)
    # Lp_inv = torch.inverse(Lp)
    
    # Optimization variable X (Whitened Weights)
    X_point = (W_original @ Lp).flatten()
    dim_x = X_point.shape[0]
    out_dim, in_dim = W_original.shape
    print(f"Optimization Variable dim: {dim_x}")

    # --- C. Stateless Wrapper ---
    print("Step 3: Building Stateless Wrapper...")
    
    # We verify the "Layer + Nonlinearity" block locally
    def model_func_stateless(x_vec):
        # 1. Reconstruct W
        W_new = (x_vec.view(out_dim, in_dim) @ Lp_inv)
        
        # 2. Local Forward Pass
        # Using the REAL H we captured
        layer_out = torch.nn.functional.linear(H, W_new, bias_original)
        
        # 3. Simulated Tail (e.g., GELU)
        # This proves JVP/VJP can backprop through the layer logic
        out = torch.nn.functional.gelu(layer_out)
        
        return out.flatten()

    # --- D. Implicit CG Solver ---
    print("Step 4: Running Implicit CG Solver...")
    
    # 1. Ground Truth Output
    with torch.no_grad():
        out_gt = model_func_stateless(X_point)
    
    # 2. Construct Targets
    Y_top = X_point.clone()        # Reg Target
    Y_bottom = out_gt.clone()      # Physics/Data Target
    
    # 3. Implicit RHS (b)
    _, vjp_fn = vjp(model_func_stateless, X_point)
    term_2 = vjp_fn(Y_bottom)[0]
    b_implicit = Y_top + term_2
    
    # 4. Implicit Operator A(v)
    def A_operator(v):
        _, Jv = jvp(model_func_stateless, (X_point,), (v,))
        term_vjp = vjp_fn(Jv)[0]
        return v + term_vjp # (I + J^T J) v
        
    # 5. Run CG
    x_est = torch.zeros_like(X_point)
    r = b_implicit - A_operator(x_est)
    p = r.clone()
    rsold = r @ r
    
    print(f"Initial Residual: {torch.sqrt(rsold):.4e}")
    
    for i in range(50):
        Ap = A_operator(p)
        alpha = rsold / (p @ Ap)
        x_est += alpha * p
        r -= alpha * Ap
        rsnew = r @ r
        
        if torch.sqrt(rsnew) < 1e-6:
            print(f"✅ Converged at step {i}!")
            break
        
        p = r + (rsnew/rsold) * p
        rsold = rsnew
        if i % 10 == 0:
            print(f"Step {i}: Residual {torch.sqrt(rsnew):.4e}")

    final_err = torch.norm(x_est - X_point) / torch.norm(X_point)
    print(f"Final Relative Error: {final_err:.8e}")
    
    if final_err < 1e-5:
        print("🎉 Success! Implicit CG works with captured acts.")
    else:
        print("❌ Failed to converge.")

# ==========================================
# Main
# ==========================================
device = "cuda"

if __name__ == '__main__':
    # Config
    downsteam_task = "fluids.incompressible.BrownianBridge"
    use_phyloss = "no_phyloss"
    model_name = "Poseidon-L"

    print("==== 1. Loading Dataset ====")
    cali_data = get_dataset(
        dataset=downsteam_task,
        which="val",
        num_trajectories=1,
        data_path=f"poseidon_main/camlab-ethz/down_streams",
        fix_input_to_time_step=0,
        time_step_size=10,
        max_num_time_steps=1,
    )
    
    print("==== 2. Loading Trainer ====")
    trainer = get_trainer(
        model_path=f"checkpoints/{model_name}/{use_phyloss}/{downsteam_task}/origin/none",
        batch_size=256, 
        dataset=cali_data,
    )
    
    # 1. Capture acts ONCE for the whole model
    print("==== 3. Profiling Inputs (Acts) ====")
    acts = profile_inputs_physics(trainer, cali_data, device)
    # 2. Select Layer
    layer_name = "encoder.layers.0.blocks.0.attention.output.dense" 
    
    # 3. Run Experiment using captured acts
    experiment_cg_on_layer(trainer, layer_name, acts)