#coding:utf8
import os
import sys
import argparse
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

from poseidon_main.scOT.inference import get_trainer, rollout
from poseidon_main.scOT.problems.base import get_dataset
 
from MySVD_origin.helper.input_collector import *
from MySVD_origin.helper.rank_selector import *
from MySVD_origin.helper.fisher_collector import *
from MySVD_origin.helper.prevds_collector import *
from MySVD_origin.helper.calibration_dataset import (
    build_calibration_dataset,
    calibration_cache_tag,
    describe_calibration_selection,
)

# 压缩方法导入
from MySVD_origin.compression_method.fast_compress import svd_compress_with_collector
from MySVD_origin.compression_method.compress_clean_input import *
from MySVD_origin.compression_method.compress_comb_input import mysvd_comb, mysvd_comb_ablation
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- 1. 基础配置 (统一参数名) ---
    parser.add_argument('--model_name', type=str, default='Poseidon-L')
    parser.add_argument('--dataset', type=str, default='wave.Gaussians')
    parser.add_argument('--traj_num', type=int, default=64)
    parser.add_argument('--random_calibration', type=str, default='n', choices=['y', 'n'],
                        help='Use the same seeded random trajectory subset as rank selection.')
    parser.add_argument('--calibration_seed', type=int, default=2026,
                        help='Seed used when --random_calibration y.')
    parser.add_argument('--use_phyloss', type=str, default='no_phyloss')
    parser.add_argument('--final_time', type=int, default=15)
    parser.add_argument('--time_back', type=int, default=0)
    parser.add_argument('--time_mode', type=str, default="alltime")
    parser.add_argument('--energy_convert', type=str, default="")
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--uniform', type=str, default='n', choices=['y', 'n'], help='Whether to use uniform rank')
    # --- 2. 压缩配置 (统一参数名) ---
    parser.add_argument('--rank_ratio', type=float, default=0.2, help='Uniform rank ratio or Energy threshold')
    parser.add_argument('--rank_type', type=str, default='MySVD_origin_Rank')
    parser.add_argument('--DEV', type=str, default='cuda')

    # --- 3. MySVD 特定参数 ---
    parser.add_argument('--use_fisher', type=str, default='y', choices=['y', 'n'])
    parser.add_argument('--use_prev_time', type=str, default="False")
    parser.add_argument('--balance_ratio', type=float, default=10.0)
    parser.add_argument('--order_control', type=int, default=2)
    parser.add_argument('--gradient_loss', type=str, default='l1', choices=['mse', 'l1'])
    parser.add_argument('--skip_layer', type=str, default='n')
    # --- 4. 路径覆盖 ---
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='poseidon_main/camlab-ethz/down_streams')
    parser.add_argument('--rank_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()
    random_calibration = args.random_calibration == 'y'
    calibration_tag = calibration_cache_tag(
        args.traj_num,
        random_calibration=random_calibration,
        calibration_seed=args.calibration_seed,
        final_time=args.final_time,
    )
    target_cali_time = args.final_time - args.time_back
    if args.skip_layer == "n": skip_layer = False
    else: skip_layer = True
    # --- 5. 路径自动构建 (与其他脚本对齐) ---
    args.rank_path = args.rank_path or (
        f"rank_results/{args.model_name}/{args.dataset}/MySVD_origin_Rank/"
        f"{calibration_tag}_order{args.order_control}_{args.gradient_loss}/"
        f"rank_{args.rank_ratio}_origin.json"
    )
    if args.uniform != "y":
        if not os.path.exists(args.rank_path): raise FileNotFoundError(f"No Path: {args.rank_path}")

    time_str = "" if args.time_back == 0 else f"_time{target_cali_time}"
    fn = f"{args.gradient_loss}_order{args.order_control}_alpha{args.alpha}" if args.use_fisher == "y" else "no_fisher"
    if args.uniform == "y": fn += "_unirank"
    # if args.use_prev_time == "False": fn += "_no_prevt"
    calibration_suffix = f"_{calibration_tag}" if random_calibration else ""
    save_path = args.save_path or (
        f"compression/compressed_ft_models/{args.dataset}/{args.rank_ratio}/"
        f"MySVD_origin/{fn}{time_str}{calibration_suffix}_origin"
    )

    args.model_path = args.model_path or (
        f"checkpoints/{args.model_name}/{args.use_phyloss}/{args.dataset}/origin/none"
    )

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

    # --- 7. Fisher Collection ---
    fisher_cache = (
        f"MySVD_origin/fisher_cache/{args.dataset}/{calibration_tag}_"
        f"order{args.order_control}_{args.gradient_loss}/Ldict_origin.pt"
    )
    os.makedirs(os.path.dirname(fisher_cache), exist_ok=True)
    Ldict = None
    if args.use_fisher == "y":
        if os.path.exists(fisher_cache):
            print(f"==== Loading cached Ldict from {fisher_cache} ====")
            Ldict = torch.load(fisher_cache)
        else:
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

    # --- Compress ---
    print(f"==== Running Physics-Aware SVD [{args.rank_type}] ====")
    os.makedirs(save_path, exist_ok=True)
    
    if args.alpha == 0:
        profile_mat = torch.load(
            f"MySVD_origin/input_cache/{args.dataset}/{calibration_tag}/profiling_mat_origin.pt"
        )
        model = mysvd_clean_input(
                trainer, dev=args.DEV, ranks_json_path=args.rank_path, calib_dataset=cali_data, 
                profile_mat=profile_mat, Ldict=Ldict
            )
    else:
        model = mysvd_comb(
            trainer, dev=args.DEV, ranks_json_path=args.rank_path, calib_dataset=cali_data, 
            Ldict=Ldict, alpha=args.alpha, save_path=save_path
        )
        # model = mysvd_comb_ablation(
        #     trainer, dev=args.DEV, ranks_json_path=args.rank_path, calib_dataset=cali_data, 
        #     Ldict=Ldict, alpha=args.alpha, uniform_rank=(args.uniform == "y"), use_fisher=(args.use_fisher == "y"),
        #     save_path=save_path,
        # )

    # --- 9. Save ---
    torch.save(model, f"{save_path}/compressed_model_origin.pt")
    print(f"Success! Saved to {save_path}")
