#!/bin/bash
#SBATCH --job-name=launch_mysvd
#SBATCH --gpus=0
#SBATCH --time=01:00:00
#SBATCH --output=launch_mysvd.out

CALI_NUMS=(2048)
RANK_RATIO=(0.2 0.4 0.6 0.8)
ALPHAs=(0.7)
COMP_MODE="comb"
use_trace="y"
for cali_num in "${CALI_NUMS[@]}"; do
    for rank in "${RANK_RATIO[@]}"; do
        for alpha in "${ALPHAs[@]}"; do
          RUN_ID="cali${cali_num}_order${order}_rank${rank}_alpha${alpha}"
          LOG_FILE="logs/mysvd/${RUN_ID}.out"

          echo "Submitting job for RANK_RATIO: $rank, Calibration Number: $cali_num..., Alpha: $alpha"

          sbatch --job-name="mysvd_${RUN_ID}" \
              --gpus=1 \
              --time=24:00:00 \
              --output="$LOG_FILE" \
              --wrap="
                      source ~/miniconda3/etc/profile.d/conda.sh
                      conda activate vicon
                      export PYTHONPATH=$(pwd)
                      
                      python -u MySVD/mysvd.py \
                      +calibration_num=$cali_num \
                      +rank_ratio=$rank \
                      +compress_mode=$COMP_MODE \
                      +alpha=$alpha \
                      +use_trace_align=$use_trace \
              "
          
          sleep 0.5
      done
    done
done

echo "All jobs submitted. Use 'squeue -u $USER' to check status."