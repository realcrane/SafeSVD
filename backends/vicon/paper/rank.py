from MySVD.helper import *
import os
import torch
import torch.nn as nn
import math
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import json
from MySVD.convert_attn import convert_attn_to_linear
from VICON.src import utils
from VICON.src.dataset import all_datasets
import VICON.src.models as models
from VICON.src.trainer import Trainer
from VICON.src.rollout_utils import rollout_one_traj
from VICON.src.rollout_strategy import gen_strategy_list
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

@hydra.main(version_base=None, config_path="../VICON/configs/", config_name="default")
def main(cfg: DictConfig):
    calibration_num = cfg.get("calibration_num", 2048)
    rank_ratio      = cfg.get("rank_ratio", 0.8)
    gradient_loss   = cfg.get("gradient_loss", "l1")
    balance_ratio   = cfg.get("balance_ratio", 10.0)
    use_trace_align_str = cfg.get("use_trace_align", "n")  # y / n
    random_calibration_str = cfg.get("random_calibration", "n")  # y / n
    calibration_seed = int(cfg.get("calibration_seed", 2026))

    use_trace_align = (use_trace_align_str.lower() == "y")
    random_calibration = (random_calibration_str.lower() == "y")
    trace_tag = "trace" if use_trace_align else "notrace"
    calibration_tag = calibration_cache_tag(
        calibration_num,
        random_calibration,
        calibration_seed,
    )
    print(f"[Rank] use_trace_align = {use_trace_align} ({trace_tag})")

    batch_size = 128
    max_calib_batches = int(calibration_num / batch_size)

    # 【改动】cache 路径加入 trace_tag
    cache_dir = cfg.get("cache_dir", "MySVD")
    fisher_cache = (
        f"{cache_dir}/fisher_cache/{calibration_tag}"
        f"_{gradient_loss}_{trace_tag}/Ldict.pt"
    )
    profile_cache = (
        f"{cache_dir}/input_cache/{calibration_tag}_{trace_tag}/profiling_mat.pt"
    )

    os.makedirs(os.path.dirname(fisher_cache), exist_ok=True)
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
        print(f"Loaded Model: {ckpt_path}")
    else:
        print(f"No file exist: {ckpt_path}")

    print("Loading Training Dataset...")
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
        "[Rank] Calibration selection: "
        + describe_calibration_selection(
            calib_datasets,
            calibration_num=calibration_num,
            random_calibration=random_calibration,
            calibration_seed=calibration_seed,
        )
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    original_attns = convert_attn_to_linear(trainer.model)

    if os.path.exists(profile_cache):
        print(f"==== Loading cached Input from {profile_cache} ====")
        Rdict = torch.load(profile_cache)
    else:
        Rdict = profile_svd_vicon(
            trainer, calib_loaders, cfg, dev,
            max_calib_batches=max_calib_batches,
            use_trace_align=use_trace_align,       # ← 传递超参
        )
        os.makedirs(os.path.dirname(profile_cache), exist_ok=True)
        torch.save(Rdict, profile_cache)

    if os.path.exists(fisher_cache):
        print(f"==== Loading cached Ldict from {fisher_cache} ====")
        Ldict = torch.load(fisher_cache)
    else:
        Ldict = collect_mysvd_fisher_vicon(
            trainer=trainer,
            calib_loaders=calib_loaders,
            dev=dev,
            max_calib_batches=max_calib_batches,
            gradient_loss=gradient_loss,
            balance_ratio=balance_ratio,
            use_trace_align=use_trace_align,       # ← 传递超参
        )
        os.makedirs(os.path.dirname(fisher_cache), exist_ok=True)
        torch.save(Ldict, fisher_cache)
        print(f"Ldict saved to {fisher_cache}")

    rank_dict = get_global_greedy_rank_dict(trainer, Ldict, Rdict, rank_ratio, dev)

    # 【改动】rank 输出路径也加入 trace_tag
    rank_root = cfg.get("rank_root", "rank_results/MySVD_Rank")
    rank_path = cfg.get(
        "rank_path",
        f"{rank_root}/{calibration_tag}_{gradient_loss}_{trace_tag}/rank_{rank_ratio}.json",
    )
    save_dir = os.path.dirname(rank_path)
    os.makedirs(save_dir, exist_ok=True)
    with open(rank_path, "w") as f:
        json.dump(rank_dict, f, indent=4)
    print("Done")

if __name__ == "__main__":
    main()
