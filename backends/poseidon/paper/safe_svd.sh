#!/bin/bash
#SBATCH --job-name=launch_safe_mysvd_origin
#SBATCH --gpus=0
#SBATCH --time=00:20:00
#SBATCH --output=launch_safe_mysvd_origin.out

# Submit one job per complete rank-selection + compression pipeline.
MODEL="Poseidon-L"
CONFIG_LIST=(
    # dataset|final_time|use_prev_time|short_name|order_control
    "fluids.incompressible.PiecewiseConstants|20|False|NS-PwC|2"
    # "fluids.compressible.RichtmyerMeshkov|20|False|CE-RM|2"
    # "wave.Gaussians|15|False|Wave-Gauss|2"
)

TRAJ_NUMS=(64)

# Set to 'n' to use the first traj_num trajectories.  With 'y', one complete
# and reproducible pipeline is submitted for every seed below.
RANDOM_CALIBRATION='n'
CALIBRATION_SEEDS=(0)

RANK_RATIOS=(0.2)
ALPHAS=(0.7)
LOSS='l1'
USE_FISHER='y'
UNIFORM='n'
TIME_BACK=0
JOB_TIME='24:00:00'

if [[ "$RANDOM_CALIBRATION" == 'y' ]]; then
    ACTIVE_SEEDS=("${CALIBRATION_SEEDS[@]}")
else
    ACTIVE_SEEDS=(0)
fi

mkdir -p logs/safe_svd_origin

for config in "${CONFIG_LIST[@]}"; do
    IFS='|' read -r dataset final_time use_prev_time short_name order_control <<< "$config"
    for traj_num in "${TRAJ_NUMS[@]}"; do
        for seed in "${ACTIVE_SEEDS[@]}"; do
            if [[ "$RANDOM_CALIBRATION" == 'y' ]]; then
                calibration_tag="random_seed${seed}"
            else
                calibration_tag='fixed'
            fi
            for rank_ratio in "${RANK_RATIOS[@]}"; do
                for alpha in "${ALPHAS[@]}"; do
                    run_id="${short_name}_traj${traj_num}_${calibration_tag}_${rank_ratio}_order${order_control}_alpha${alpha}_ftime${final_time}"
                    log_file="logs/safe_svd_origin/${run_id}.out"
                    echo "Submitting MySVD_origin: dataset=${dataset}, traj_num=${traj_num}, seed=${seed}, rank_ratio=${rank_ratio}, alpha=${alpha}"

                    sbatch --job-name="safe_origin_${run_id}" \
                        --gpus=1 \
                        --time="$JOB_TIME" \
                        --output="$log_file" \
                        --wrap="
                            set -euo pipefail
                            source /lus/lfs1aip2/scratch/u6ey/chunleili.u6ey/Dev/miniforge3/etc/profile.d/conda.sh
                            conda activate poseidon
                            export PYTHONPATH=$(pwd)
                            
                            python -u MySVD_origin/safe_svd.py \\
                                --model_name '$MODEL' \\
                                --dataset '$dataset' \\
                                --traj_num $traj_num \\
                                --random_calibration '$RANDOM_CALIBRATION' \\
                                --calibration_seed $seed \\
                                --final_time $final_time \\
                                --time_back $TIME_BACK \\
                                --rank_ratio $rank_ratio \\
                                --use_fisher '$USE_FISHER' \\
                                --use_prev_time '$use_prev_time' \\
                                --order_control $order_control \\
                                --gradient_loss '$LOSS' \\
                                --uniform '$UNIFORM' \\
                                --alpha $alpha
                        "
                    sleep 0.5
                done
            done
        done
    done
done

echo 'All safe MySVD_origin jobs submitted.'
