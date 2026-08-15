#!/bin/bash
#SBATCH --job-name=launch_vicon_safe_svd
#SBATCH --gpus=0
#SBATCH --time=00:20:00
#SBATCH --output=launch_vicon_safe_svd.out

# Run this launcher from physic_model_comp/vicon.

CALI_NUMS=(2048)

# 'n': preserve the original fixed-prefix calibration behavior.
# 'y': submit one complete rank-selection + compression job per seed.
RANDOM_CALIBRATION='y'
CALIBRATION_SEEDS=(7 137 1204 1729 65537)
# 0 1 42 2026 3407

RANK_RATIOS=(0.2 0.4 0.6 0.8)
ALPHAS=(0.7)
GRADIENT_LOSS='l1'
BALANCE_RATIO=10.0
USE_TRACE_ALIGN='y'
USE_FISHER='y'
COMPRESS_MODE='comb'
DEV='cuda'
JOB_TIME='08:00:00'

if [[ "$RANDOM_CALIBRATION" == 'y' ]]; then
    ACTIVE_SEEDS=("${CALIBRATION_SEEDS[@]}")
else
    # calibration_seed is ignored in fixed-prefix mode.
    ACTIVE_SEEDS=(0)
fi

mkdir -p logs/safe_svd

for cali_num in "${CALI_NUMS[@]}"; do
    for seed in "${ACTIVE_SEEDS[@]}"; do
        if [[ "$RANDOM_CALIBRATION" == 'y' ]]; then
            CALIBRATION_TAG="random_seed${seed}"
        else
            CALIBRATION_TAG='fixed'
        fi

        for rank_ratio in "${RANK_RATIOS[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                RUN_ID="cali${cali_num}_${CALIBRATION_TAG}_rank${rank_ratio}_alpha${alpha}_${GRADIENT_LOSS}_trace${USE_TRACE_ALIGN}"
                LOG_FILE="logs/safe_svd/${RUN_ID}.out"

                echo "Submitting VICON safe MySVD: calibration_num=$cali_num, calibration=$CALIBRATION_TAG, rank_ratio=$rank_ratio, alpha=$alpha"

                sbatch --job-name="vicon_safe_${RUN_ID}" \
                    --gpus=1 \
                    --time="$JOB_TIME" \
                    --output="$LOG_FILE" \
                    --wrap="
                            set -euo pipefail
                            source /lus/lfs1aip2/scratch/u6ey/chunleili.u6ey/Dev/miniforge3/etc/profile.d/conda.sh
                            conda activate vicon
                            export PYTHONPATH=$(pwd)

                            python -u MySVD/safe_svd.py \
                                --calibration_num $cali_num \
                                --random_calibration '$RANDOM_CALIBRATION' \
                                --calibration_seed $seed \
                                --rank_ratio $rank_ratio \
                                --gradient_loss '$GRADIENT_LOSS' \
                                --balance_ratio $BALANCE_RATIO \
                                --use_trace_align '$USE_TRACE_ALIGN' \
                                --DEV '$DEV' \
                                --use_fisher '$USE_FISHER' \
                                --compress_mode '$COMPRESS_MODE' \
                                --alpha $alpha
                    "

                sleep 0.5
            done
        done
    done
done

echo "All VICON safe MySVD jobs submitted. Use 'squeue -u $USER' to check status."
