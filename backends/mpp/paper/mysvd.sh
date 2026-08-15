#!/bin/bash
#SBATCH --job-name=mpp_mysvd
#SBATCH --gpus=0
#SBATCH --time=01:00:00
#SBATCH --output=mpp_mysvd.out


cali_num=16
RANK_RATIO=(0.2 0.4 0.6 0.8)
use_trace="y"
for rank_ratio in "${RANK_RATIO[@]}"; do
    RUN_ID="${rank_ratio}"
    LOG_FILE="logs/mysvd/${RUN_ID}.out"

    echo "Submitting job for Rank Ratio: $rank_ratio..."

    sbatch --job-name="mpp_mysvd_${RUN_ID}" \
        --gpus=1 \
        --time=05:00:00 \
        --output="$LOG_FILE" \
        --wrap="
                source ~/miniconda3/etc/profile.d/conda.sh
                conda activate mpp
                export PYTHONPATH=$(pwd)
                python -u MySVD/mysvd.py \
                        --rank_ratio $rank_ratio \
                        --calibration_num $cali_num \
                        --alpha 0.7 \
                        --use_trace_balance $use_trace \

            "
    sleep 0.5
done