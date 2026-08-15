#!/bin/bash
#SBATCH --job-name=launch_mysvd_origin
#SBATCH --gpus=0
#SBATCH --time=01:00:00
#SBATCH --output=launch_mysvd_origin.out

# --- 1. 配置参数列表 ---
MODEL="Poseidon-L"
CONFIG_LIST=(
    # "fluids.incompressible.BrownianBridge|20|False|NS-BB"
    # "fluids.incompressible.PiecewiseConstants|20|False|NS-PwC"
    # "fluids.incompressible.VortexSheet|20|False|NS-SVS"
    # "fluids.compressible.RichtmyerMeshkov|20|False|CE-RM"
    # "fluids.compressible.RiemannKelvinHelmholtz|20|False|CE-RPUI"
    "wave.Gaussians|15|False|Wave-Gauss"
    # "wave.Layer|15|False|Wave-Layer"
)

TRAJ_NUMS=(64)
RANK_RATIO=(0.2)
ORDER_CONTROL=(2)
USE_FISHER='y'
# Matches the 5.24 mysvd_rank.py default.  Change to l1 only for a deliberate
# like-for-like comparison against a current l1 run.
LOSS='l1'
TIME_BACK=0

mkdir -p logs/mysvd_rank_origin

# --- 2. 开始循环分发 ---
for config in "${CONFIG_LIST[@]}"; do
    IFS="|" read -r ds ftime useptime short_name<<< "$config"
    
    for traj in "${TRAJ_NUMS[@]}"; do
        for rank_ratio in "${RANK_RATIO[@]}"; do
            for order in "${ORDER_CONTROL[@]}"; do

                RUN_ID="${short_name}_${traj}_${rank_ratio}_order${order}"
                LOG_FILE="logs/mysvd_rank_origin/${RUN_ID}_origin.out"

                echo "Submitting job for Dataset: $ds, Rank Ratio: $rank_ratio..."

                sbatch --job-name="mysvd_rank_origin_${RUN_ID}" \
                    --gpus=1 \
                    --time=1:00:00 \
                    --output="$LOG_FILE" \
                    --wrap="
                            source ~/miniconda3/etc/profile.d/conda.sh
                            conda activate poseidon
                            export PYTHONPATH=$(pwd)
                            
                            python -u MySVD_origin/mysvd_rank.py \
                                --dataset '$ds' \
                                --traj_num $traj \
                                --use_fisher '$USE_FISHER' \
                                --order_control $order \
                                --gradient_loss '$LOSS' \
                                --rank_ratio $rank_ratio \
                                --use_prev_time '$useptime' \
                                --final_time $ftime
                    "
                
                sleep 0.5
            done
        done
    done
done

echo "All jobs submitted. Use 'squeue -u $USER' to check status."
