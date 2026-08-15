"""
MySVD Compression for VICON
============================
【改动说明】
  新增 use_trace_align 超参 (hydra CLI: use_trace_align=y/n):
    y → 按 dataset_type 分桶收集统计量，合并前做 trace 能量对齐
    n → 按 dataset_type 分桶收集统计量，合并时简单平均 (无缩放)
  该超参贯穿 Step 1 (profiling) / Step 2 (Fisher) / Step 5 (compress) 三个阶段。
"""

import os
import copy
import json
import math
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from VICON.src import utils
from VICON.src.dataset import all_datasets
import VICON.src.models as models
from VICON.src.trainer import Trainer
from MySVD.convert_attn import convert_attn_to_linear
from MySVD.helper.fisher_collector import *
from MySVD.helper.input_collector import *
from MySVD.helper.rank_selector import *
from MySVD.helper.whitener import *
from MySVD.helper.calibration_dataset import (
    build_calibration_loaders,
    calibration_cache_tag,
    describe_calibration_selection,
)
from MySVD.compress.clean_compress import *
from MySVD.compress.comb_compress import *

# =====================================================================
# Main entry point (hydra-based, aligned with svdllm.py)
# =====================================================================

@hydra.main(version_base=None, config_path="../VICON/configs/", config_name="default")
def main(cfg: DictConfig):
    # --- CLI overrides via hydra ---
    rank_ratio      = cfg.get("rank_ratio", 0.8)
    calibration_num = cfg.get("calibration_num", 2048)
    dev             = cfg.get("DEV", "cuda")
    use_fisher      = cfg.get("use_fisher", "y")        # y / n
    rank_type       = cfg.get("rank_type", "MySVD_Rank") # MySVD_Rank / Energy_Rank
    gradient_loss   = cfg.get("gradient_loss", "l1")
    balance_ratio   = cfg.get("balance_ratio", 10.0)
    compress_mode   = cfg.get("compress_mode", "comb") # ACES / DIRTY /comb
    alpha           = cfg.get("alpha", 0.7)
    random_calibration_str = cfg.get("random_calibration", "n")  # y / n
    calibration_seed = int(cfg.get("calibration_seed", 2026))
    random_calibration = (random_calibration_str.lower() == "y")

    # 【新增】trace 对齐开关
    use_trace_align_str = cfg.get("use_trace_align", "y")  # y / n
    use_trace_align = (use_trace_align_str.lower() == "y")

    batch_size = 128
    max_calib_batches = int(calibration_num / batch_size)

    trace_tag = "trace" if use_trace_align else "notrace"
    calibration_tag = calibration_cache_tag(
        calibration_num,
        random_calibration,
        calibration_seed,
    )
    print(f"[MySVD] use_trace_align = {use_trace_align} ({trace_tag})")

    # ============================================
    # 1. Build model & trainer
    # ============================================
    if cfg.model.type == "crop":
        model = models.ICON_CROPPED(cfg.model)
    else:
        model = models.ICON_UNCROPPED(cfg.model)

    trainer = Trainer(model, cfg.model, cfg.opt, cfg.loss, trainable_mode=cfg.trainable_mode)

    ckpt_path = cfg.get("ckpt_path", "VICON/output/ckpts/train/20260220-085512/200000_params.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            trainer.model.load_state_dict(checkpoint)
        print(f"Loaded pretrained weights: {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    convert_attn_to_linear(trainer.model)

    # ============================================
    # 2. Calibration data
    # ============================================
    print("Initializing calibration dataset (split: train)...")
    dataset_seed = calibration_seed if random_calibration else cfg.train_seed
    calib_datasets = all_datasets(cfg.datasets, 0, dataset_seed, "train")
    calib_loaders = build_calibration_loaders(
        calib_datasets,
        calibration_num=calibration_num,
        batch_size=batch_size,
        num_workers=cfg.dataset_workers,
        random_calibration=random_calibration,
        calibration_seed=calibration_seed,
    )
    print(
        "[MySVD] Calibration selection: "
        + describe_calibration_selection(
            calib_datasets,
            calibration_num=calibration_num,
            random_calibration=random_calibration,
            calibration_seed=calibration_seed,
        )
    )

    # ============================================
    # 3. Fisher information (Ldict)
    # ============================================
    # 【改动】cache 路径加入 trace_tag，避免 trace/notrace 混用缓存
    cache_dir = cfg.get("cache_dir", "MySVD")
    fisher_cache = (
        f"{cache_dir}/fisher_cache/{calibration_tag}"
        f"_{gradient_loss}_{trace_tag}/Ldict.pt"
    )
    Ldict = None
    if use_fisher == "y":
        if os.path.exists(fisher_cache):
            print(f"Loading cached Ldict from {fisher_cache}")
            Ldict = torch.load(fisher_cache, map_location='cpu')
        else:
            Ldict = collect_mysvd_fisher_vicon(
                trainer=trainer,
                calib_loaders=calib_loaders,
                dev=dev,
                max_calib_batches=max_calib_batches,
                gradient_loss=gradient_loss,
                balance_ratio=balance_ratio,
                use_trace_align=use_trace_align,   # ← 传递超参
            )
            os.makedirs(os.path.dirname(fisher_cache), exist_ok=True)
            torch.save(Ldict, fisher_cache)
            print(f"Ldict saved to {fisher_cache}")

    # ============================================
    # 4. Rank file path
    # ============================================
    rank_root = cfg.get("rank_root", "rank_results/MySVD_Rank")
    rank_path = cfg.get(
        "rank_path",
        f"{rank_root}/{calibration_tag}_{gradient_loss}_{trace_tag}/rank_{rank_ratio}.json",
    )

    # ============================================
    # 5. Run MySVD compression
    # ============================================
    print(f"==== Running MySVD [{rank_type}] on VICON ====")

    # 【改动】input profiling cache 也加入 trace_tag
    input_cache = (
        f"{cache_dir}/input_cache/{calibration_tag}_{trace_tag}/profiling_mat.pt"
    )
    if os.path.exists(input_cache):
        print(f"Loading cached profiling_mat from {input_cache}")
        Rdict = torch.load(input_cache, map_location='cpu')
    else:
        # 如果没有缓存，需要重新 profile（带 trace 超参）
        Rdict = profile_svd_vicon(
            trainer, calib_loaders, cfg, dev,
            max_calib_batches=max_calib_batches,
            use_trace_align=use_trace_align,   # ← 传递超参
        )
        os.makedirs(os.path.dirname(input_cache), exist_ok=True)
        torch.save(Rdict, input_cache)
        print(f"Saved profiling_mat to {input_cache}")

    fn = (
        f"{calibration_tag}_{gradient_loss}"
        f"_br{balance_ratio}_{trace_tag}"
        if use_fisher == "y"
        else f"no_fisher_{calibration_tag}_{trace_tag}"
    )

    if compress_mode == "clean_input":
        fn += "_clean_input"
        compressed_model = mysvd_clean_input(
            trainer=trainer, dev=dev, ranks_json_path=rank_path,
            Ldict=Ldict, Rdict=Rdict,
        )
    else:
        compressed_model = mysvd_comb(
            trainer=trainer,
            calib_loaders=calib_loaders,
            dev=dev,
            ranks_json_path=rank_path,
            Ldict=Ldict,
            max_calib_batches=max_calib_batches,
            alpha=alpha,
            use_trace_align=use_trace_align,   # ← 传递超参
        )

    # ============================================
    # 6. Save
    # ============================================
    save_path = cfg.get(
        "save_path",
        f"compressed_model/{rank_ratio}/MySVD/{fn}"
    )
    os.makedirs(save_path, exist_ok=True)

    final_save_path = f"{save_path}/compressed_model.pth"
    torch.save(compressed_model.state_dict(), final_save_path)
    print(f"Success! Model saved to {final_save_path}")


if __name__ == "__main__":
    main()
