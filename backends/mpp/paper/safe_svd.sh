#!/bin/bash
#SBATCH --job-name=launch_mpp_safe_svd
#SBATCH --gpus=0
#SBATCH --time=00:20:00
#SBATCH --output=launch_mpp_safe_svd.out

# Run this launcher from physic_model_comp/mpp.

YAML_CONFIG='multiple_physics_pretraining/config/mpp_avit_L_config.yaml'
CONFIG='basic_config'
CKPT_PATH='models/MPP_AViT_L'

CALI_NUMS=(16)
CALIBRATION_SEEDS=(7 137 1204 1729 65537)
# 0 1 42 2026 3407
RANK_RATIOS=(0.2 0.4 0.6 0.8)
ALPHAS=(0.7)

USE_TRACE_BALANCE='y'
USE_FISHER='y'
FISHER_LOSS='l1'
BALANCE_RATIO=10.0
DECAY_ALPHA=1.0
DEV='cuda'
JOB_TIME='08:00:00'

mkdir -p logs/safe_svd

for cali_num in "${CALI_NUMS[@]}"; do
    for seed in "${CALIBRATION_SEEDS[@]}"; do
        for rank_ratio in "${RANK_RATIOS[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                RUN_ID="cali${cali_num}_seed${seed}_rank${rank_ratio}_alpha${alpha}_${FISHER_LOSS}_trace${USE_TRACE_BALANCE}"
                LOG_FILE="logs/safe_svd/${RUN_ID}.out"

                echo "Submitting MPP safe MySVD: calibration_num=$cali_num, seed=$seed, rank_ratio=$rank_ratio, alpha=$alpha"

                sbatch --job-name="mpp_safe_${RUN_ID}" \
                    --gpus=1 \
                    --time="$JOB_TIME" \
                    --output="$LOG_FILE" \
                    --wrap="
                            set -euo pipefail
                            source /lus/lfs1aip2/scratch/u6ey/chunleili.u6ey/Dev/miniforge3/etc/profile.d/conda.sh
                            conda activate mpp_new
                            export PYTHONPATH=$(pwd)

                            python -u MySVD/safe_svd.py \
                                --yaml_config '$YAML_CONFIG' \
                                --config '$CONFIG' \
                                --ckpt_path '$CKPT_PATH' \
                                --calibration_num $cali_num \
                                --calibration_seed $seed \
                                --rank_ratio $rank_ratio \
                                --alpha $alpha \
                                --DEV '$DEV' \
                                --use_trace_balance '$USE_TRACE_BALANCE' \
                                --use_fisher '$USE_FISHER' \
                                --fisher_loss '$FISHER_LOSS' \
                                --balance_ratio $BALANCE_RATIO \
                                --decay_alpha $DECAY_ALPHA
                    "

                sleep 0.5
            done
        done
    done
done

echo "All MPP safe MySVD jobs submitted. Use 'squeue -u $USER' to check status."
