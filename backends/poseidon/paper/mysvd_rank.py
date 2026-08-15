import torch
import math
#coding:utf8
import os
import sys
import argparse
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 项目路径引用
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

from poseidon_main.scOT.inference import get_trainer, rollout
from poseidon_main.scOT.problems.base import get_dataset
 
from MySVD_origin.helper.input_collector import *
from MySVD_origin.helper.rank_selector import *
from MySVD_origin.helper.fisher_collector import *
from MySVD_origin.helper.prevds_collector import *
from MySVD_origin.helper.whitener import *
from MySVD_origin.helper.calibration_dataset import (
    build_calibration_dataset,
    calibration_cache_tag,
    describe_calibration_selection,
)
# 压缩方法导入
@torch.no_grad()
def profile_svd_physics(trainer, calib_loader, dev):
    device = torch.device(dev)
    model = trainer.model.to(device).eval()
    layers = []
    handles = []

    def hook(module, inp, out):
        x = inp[0].detach().float()
        C = x.shape[-1]
        X = x.reshape(-1, C)
        module.raw_scaling_diag_matrix += X.T @ X

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = torch.zeros(
                module.in_features, module.in_features,
                device=device, dtype=torch.float32
            )
            handles.append(module.register_forward_hook(hook))
            layers.append((name, module))  

    print("Start gathering Gram matrices via rollout...")
    _ = rollout(trainer, calib_loader, model_path=trainer.model.config.name_or_path)

    for h in handles:
        h.remove()
    
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        name, module = layers[i]
        G = module.raw_scaling_diag_matrix.double()
        Lr, Lr_invT = svd_whitener(G, eps=1e-6)
        profiling_mat[name] = (Lr.cpu().float(), Lr_invT.cpu().float())
        del module.raw_scaling_diag_matrix
    
    model.cpu()
    torch.cuda.empty_cache()
    return profiling_mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- 1. 基础配置 (统一参数名) ---
    parser.add_argument('--model_name', type=str, default='Poseidon-L')
    parser.add_argument('--dataset', type=str, default='wave.Gaussians')
    parser.add_argument('--traj_num', type=int, default=64)
    parser.add_argument('--random_calibration', type=str, default='n', choices=['y', 'n'],
                        help='Sample traj_num complete training trajectories using calibration_seed.')
    parser.add_argument('--calibration_seed', type=int, default=2026,
                        help='Seed used only when --random_calibration y.')
    parser.add_argument('--use_phyloss', type=str, default='no_phyloss')
    parser.add_argument('--final_time', type=int, default=15)
    parser.add_argument('--time_back', type=int, default=0)
    parser.add_argument('--time_mode', type=str, default="alltime")
    
    # --- 2. 压缩配置 (统一参数名) ---
    parser.add_argument('--rank_ratio', type=float, default=0.2, help='Uniform rank ratio or Energy threshold')
    parser.add_argument('--DEV', type=str, default='cuda')

    # --- 3. MySVD 特定参数 ---
    parser.add_argument('--use_fisher', type=str, default='y', choices=['y', 'n'])
    parser.add_argument('--use_prev_time', type=str, default="False")
    parser.add_argument('--balance_ratio', type=float, default=10.0)
    parser.add_argument('--order_control', type=int, default=2)
    parser.add_argument('--gradient_loss', type=str, default='l1', choices=['mse', 'l1'])
    # --- 4. 路径覆盖 ---
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='poseidon_main/camlab-ethz/down_streams')
    parser.add_argument('--rank_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Recompute Fisher/profile/rank artifacts instead of reusing this calibration tag.')

    args = parser.parse_args()
    if args.traj_num <= 0:
        raise ValueError('--traj_num must be positive')
    if args.final_time <= args.time_back:
        raise ValueError('--final_time must be greater than --time_back')
    random_calibration = args.random_calibration == 'y'
    calibration_tag = calibration_cache_tag(
        args.traj_num,
        random_calibration=random_calibration,
        calibration_seed=args.calibration_seed,
        final_time=args.final_time,
    )
    args.model_path = args.model_path or f"checkpoints/{args.model_name}/{args.use_phyloss}/{args.dataset}/origin/none"
    target_cali_time = args.final_time - args.time_back
    # --- 6. Setup Data & Trainer ---
    print(f"==== Loading Data: {args.dataset} ====")
    cali_data = build_calibration_dataset(
        get_dataset,
        dataset_name=args.dataset,
        num_trajectories=args.traj_num,
        random_calibration=random_calibration,
        calibration_seed=args.calibration_seed,
        data_path=args.data_path,
        fix_input_to_time_step=None,
        time_step_size=1, 
        max_num_time_steps=target_cali_time, 
    )
    print(f"==== Calibration selection: {describe_calibration_selection(cali_data, random_calibration)} ====")
    trainer = get_trainer(model_path=args.model_path, batch_size=256, dataset=cali_data)
    fisher_cache = f"MySVD_origin/fisher_cache/{args.dataset}/{calibration_tag}_order{args.order_control}_{args.gradient_loss}/Ldict_origin.pt"
    profile_cache = f"MySVD_origin/input_cache/{args.dataset}/{calibration_tag}/profiling_mat_origin.pt"
    os.makedirs(os.path.dirname(fisher_cache), exist_ok=True)

    Ldict = None
    if args.use_fisher == "y":
        if os.path.exists(fisher_cache) and not args.overwrite_cache:
            print(f"==== Loading cached Ldict from {fisher_cache} ====")
            Ldict = torch.load(fisher_cache)
        else:
            if os.path.exists(fisher_cache):
                print(f"==== Overwriting cached Ldict at {fisher_cache} ====")
            print("==== Collecting Fisher Information ====")
            transpose = getattr(cali_data, 'transpose', False)
            calib_loader = DataLoader(cali_data, batch_size=32, shuffle=False)
            if args.use_prev_time == "True":
                prev_data = PrevTimeDataset(cali_data)
                prev_loader = DataLoader(prev_data, batch_size=32, shuffle=False)
                Ldict = collect_fisher_dict_time_sob(trainer, args.dataset, calib_loader, prev_loader, args.DEV,
                                                gradient_loss=args.gradient_loss, order_control=args.order_control,
                                                constants=cali_data.constants, transpose=transpose,
                                                balance_ratio=int(args.balance_ratio))
            else:
                Ldict = collect_fisher_dict_linear_only(trainer, args.dataset, calib_loader, args.DEV,
                                                gradient_loss=args.gradient_loss, order_control=args.order_control,
                                                constants=cali_data.constants, transpose=transpose,
                                                balance_ratio=int(args.balance_ratio))
            torch.save(Ldict, fisher_cache)
            print(f"Ldict saved to {fisher_cache}")

    # --- 8. Profiling ---
    if os.path.exists(profile_cache) and not args.overwrite_cache:
        print(f"==== Loading cached profiling_mat from {profile_cache} ====")
        profiling_mat = torch.load(profile_cache)
    else:
        if os.path.exists(profile_cache):
            print(f"==== Overwriting cached profiling_mat at {profile_cache} ====")
        profiling_mat = profile_svd_physics(trainer, cali_data, args.DEV)
        os.makedirs(os.path.dirname(profile_cache), exist_ok=True)
        torch.save(profiling_mat, profile_cache)
        print(f"profiling_mat saved to {profile_cache}")
    rank_dir = f"rank_results/{args.model_name}/{args.dataset}/MySVD_origin_Rank/{calibration_tag}_order{args.order_control}_{args.gradient_loss}"
    rank_path = args.rank_path or f"{rank_dir}/rank_{args.rank_ratio}_origin.json"
    if args.overwrite_cache or not os.path.exists(rank_path):
        if os.path.exists(rank_path):
            print(f"==== Overwriting cached rank json at {rank_path} ====")
        rank_dict = get_global_greedy_rank_dict(
            trainer, Ldict, profiling_mat, args.rank_ratio, args.DEV, eps_whiten=1e-6
        )
        save_dir = os.path.dirname(rank_path)
        os.makedirs(save_dir, exist_ok=True)
        with open(rank_path, "w") as f:
            json.dump(rank_dict, f, indent=4)
    print(f"[MySVD_origin] rank file: {rank_path}")


    # rank_dict = get_global_greedy_energy_dict(trainer, Ldict, profiling_mat, args.rank_ratio, args.DEV, eps_whiten=1e-6)
    # rank_path = f"rank_results/{args.model_name}/{args.dataset}/MySVD_Rank/traj{args.traj_num}_order{args.order_control}/energy_{args.rank_ratio}.json"
    # save_dir = os.path.dirname(rank_path)
    # os.makedirs(save_dir, exist_ok=True)
    # with open(rank_path, "w") as f:
    #     json.dump(rank_dict, f, indent=4)
