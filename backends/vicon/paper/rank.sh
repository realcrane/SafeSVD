#!/bin/bash
#SBATCH --job-name=launch_myrank
#SBATCH --gpus=0
#SBATCH --time=01:00:00
#SBATCH --output=launch_myrank.out

CALI_NUMS=(2048)
RANK_RATIO=(0.8 0.6 0.4 0.2)
use_trace=n
for cali_num in "${CALI_NUMS[@]}"; do
    for rank in "${RANK_RATIO[@]}"; do
        RUN_ID="cali${cali_num}_order${order}_rank${rank}"
        LOG_FILE="logs/myrank/${RUN_ID}.out"

        echo "Submitting job for RANK_RATIO: $rank, Calibration Number: $cali_num..."

        sbatch --job-name="myrank_${RUN_ID}" \
            --gpus=1 \
            --time=05:00:00 \
            --output="$LOG_FILE" \
            --wrap="
                    source ~/miniconda3/etc/profile.d/conda.sh
                    conda activate vicon
                    export PYTHONPATH=$(pwd)
                    
                    python -u MySVD/rank.py \
                    +calibration_num=$cali_num \
                    +rank_ratio=$rank \
                    +use_trace_align=$use_trace
            "
        
        sleep 0.5
    done
done

echo "All jobs submitted. Use 'squeue -u $USER' to check status."