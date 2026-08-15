#!/bin/bash
#SBATCH --job-name=launch_mysvd_origin
#SBATCH --gpus=0
#SBATCH --time=01:00:00
#SBATCH --output=launch_mysvd_origin.out

# --- 1. 配置参数列表 ---
MODEL="Poseidon-L"
CONFIG_LIST=(
    # "fluids.incompressible.BrownianBridge|20|False|NS-BB"
    "fluids.incompressible.PiecewiseConstants|20|False|NS-PwC"
    # "fluids.incompressible.VortexSheet|20|False|NS-SVS"
    # "fluids.compressible.RichtmyerMeshkov|20|False|CE-RM"
    # "fluids.compressible.RiemannKelvinHelmholtz|20|False|CE-RPUI"
    # "wave.Gaussians|15|False|Wave-Gauss"
    # "wave.Layer|15|False|Wave-Layer"
)

TRAJ_NUMS=(64)
RANK_RATIO=(0.2)
ORDER_CONTROL=(2)
ALPHAs=(0.7)
LOSS="l1"
RANK_PATH="rank_results/Poseidon-L/fluids.incompressible.PiecewiseConstants/MySVD_Rank/traj64_order2/rank_0.2.json"
# abalation config
USE_FISHER='y'
UNIFORM='n'
TIME_BACK=0

mkdir -p logs/mysvd_origin

# --- 2. 开始循环分发 ---
for config in "${CONFIG_LIST[@]}"; do
    IFS="|" read -r ds ftime useptime short_name<<< "$config"
    for traj in "${TRAJ_NUMS[@]}"; do
        for rank_ratio in "${RANK_RATIO[@]}"; do
            for order in "${ORDER_CONTROL[@]}"; do
                for alpha in "${ALPHAs[@]}"; do

                    RUN_ID="${short_name}_${rank_ratio}_order${order}_alpha${alpha}"
                    LOG_FILE="logs/mysvd_origin/${RUN_ID}_origin.out"

                    echo "Submitting job for Dataset: $ds, Rank Ratio: $rank_ratio..."

                    sbatch --job-name="mysvd_origin_${RUN_ID}" \
                        --gpus=1 \
                        --time=24:00:00 \
                        --output="$LOG_FILE" \
                        --wrap="
                                set -euo pipefail
                                source /lus/lfs1aip2/scratch/u6ey/chunleili.u6ey/Dev/miniforge3/etc/profile.d/conda.sh
                                conda activate poseidon
                                export PYTHONPATH=$(pwd)
                                
                                python -u MySVD_origin/mysvd.py \
                                    --dataset '$ds' \
                                    --final_time $ftime \
                                    --traj_num $traj \
                                    --use_fisher '$USE_FISHER' \
                                    --order_control $order \
                                    --rank_ratio $rank_ratio \
                                    --use_prev_time '$useptime' \
                                    --uniform '$UNIFORM' \
                                    --alpha $alpha \
                                    --gradient_loss '$LOSS' \
                                    --rank_path '$RANK_PATH' \
                        "
                
                    sleep 0.5
                done
            done
        done
    done
done

echo "All jobs submitted. Use 'squeue -u $USER' to check status."
