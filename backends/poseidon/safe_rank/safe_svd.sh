#!/bin/bash
#SBATCH --job-name=launch_safe_svd
#SBATCH --gpus=0
#SBATCH --time=00:10:00
#SBATCH --output=launch_safe_svd.out

# --- 1. Batch configuration ---
MODEL="Poseidon-L"
CONFIG_LIST=(
    # dataset|use_prev_time|short_name|order
    # "fluids.incompressible.BrownianBridge|False|NS-BB|2"
    "fluids.incompressible.PiecewiseConstants|False|NS-PwC|2"
    # "fluids.incompressible.VortexSheet|False|NS-SVS|2"
    # "fluids.compressible.RichtmyerMeshkov|False|CE-RM|1"
    # "fluids.compressible.RiemannKelvinHelmholtz|False|CE-RPUI|1"
    # "wave.Gaussians|False|Wave-Gauss|1"
    # "wave.Layer|False|Wave-Layer|1"
    # "elliptic.Helmholtz.time|False|Helmholtz|2"
    # "elliptic.poisson.Gaussians.time|False|Poisson-Gauss|2"
)

# traj_num|final_time|whitening_mode|whitening_eps_mode|whitening_eps|rank_source|experiment_tag
# rank_source=safe selects SAFE-SVD ranks; rank_source=mysvd reuses the matching
# MySVD zero-rank JSON and writes into a distinct output directory.
EXPERIMENT_LIST=(
    # Controlled reconstruction check: identical MySVD ranks and raw-Gram eps.
    # "64|20|new|absolute|1e-07|safe|abs1e-07"
    # End-to-end SAFE-SVD comparison under the same sum-equivalent whitening.
    "64|20|new|absolute|1e-12|safe|abs1e-12"
    "64|20|new|absolute|1e-15|safe|abs1e-15"
    "64|20|new|absolute|1e-18|safe|abs1e-18"
    # # Complete the current absolute-eps baseline at the longer time horizon.
    # "16|20|new|absolute|1e-08|safe|abs1e-08"
    # Test whether the sum-equivalent result scales to more trajectories.
    # "64|20|new|absolute|1e-09|safe|abs1e-09"
)

# Each value launches one complete rank-selection + compression job.
# 'default' uses the first traj_num training trajectories in order; a number
# samples traj_num complete trajectories randomly with that number as its seed.
CALIBRATION_SEEDS=(default)

RANK_RATIOS=(0.2)
ALPHAS=(0.7)
LOSS='l1'
COMPRESSION_BACKEND='GPU'
GPU_PROFILE='n'
JOB_TIME='10:00:00'

mkdir -p logs/safe_svd

# --- 2. Submit one sequential pipeline job for every configuration/experiment. ---
for config in "${CONFIG_LIST[@]}"; do
    IFS="|" read -r ds useptime short_name order <<< "$config"
    for experiment in "${EXPERIMENT_LIST[@]}"; do
        IFS="|" read -r traj ftime whitening_mode whitening_eps_mode whitening_eps rank_source experiment_tag <<< "$experiment"
        for seed in "${CALIBRATION_SEEDS[@]}"; do
            if [[ "$seed" == "default" ]]; then
                CALIBRATION_TAG="fixed"
            else
                CALIBRATION_TAG="random_seed${seed}"
            fi

            for rank_ratio in "${RANK_RATIOS[@]}"; do
                for alpha in "${ALPHAS[@]}"; do
                    RUN_ID="${short_name}_traj${traj}_${CALIBRATION_TAG}_${rank_ratio}_order${order}_alpha${alpha}_ftime${ftime}_${whitening_mode}_${experiment_tag}"
                    LOG_FILE="logs/safe_svd/${RUN_ID}.out"

                    GPU_PROFILE_ARG=""
                    if [[ "$GPU_PROFILE" == "y" ]]; then
                        GPU_PROFILE_ARG="--gpu_profile"
                    fi

                    RANK_SOURCE_ARGS=""
                    if [[ "$rank_source" == "mysvd" ]]; then
                        PIPELINE_CALIBRATION_TAG="traj${traj}"
                        if [[ "$seed" != "default" ]]; then
                            PIPELINE_CALIBRATION_TAG+="_random_seed${seed}"
                        fi
                        PIPELINE_CALIBRATION_TAG+="_ftime${ftime}"
                        WHITENING_TAG="whitening_${whitening_mode}_covmean_rankscale_eps${whitening_eps_mode}${whitening_eps}"
                        MYSVD_RANK_PATH="rank_results/${MODEL}/${ds}/MySVD_Rank/${PIPELINE_CALIBRATION_TAG}_order${order}_${LOSS}/rank_${rank_ratio}_zero.json"
                        MYSVD_SAVE_PATH="compression/compressed_ft_models/${ds}/${rank_ratio}/SAFE_SVD/${PIPELINE_CALIBRATION_TAG}_${LOSS}_order${order}_alpha${alpha}_${WHITENING_TAG}_MySVD_Rank"
                        RANK_SOURCE_ARGS=" --rank_path '${MYSVD_RANK_PATH}' --save_path '${MYSVD_SAVE_PATH}'"
                    elif [[ "$rank_source" != "safe" ]]; then
                        echo "Unknown rank_source: $rank_source" >&2
                        exit 1
                    fi

                    echo "Submitting SAFE-SVD job: dataset=$ds, traj_num=$traj, calibration=$CALIBRATION_TAG, final_time=$ftime, rank_ratio=$rank_ratio, alpha=$alpha, whitening=${whitening_mode}/${whitening_eps_mode}/${whitening_eps}"

                    sbatch --job-name="safe_svd_${RUN_ID}" \
                        --gpus=1 \
                        --time="$JOB_TIME" \
                        --output="$LOG_FILE" \
                        --wrap="
                                set -euo pipefail
                                source /lus/lfs1aip2/scratch/u6ey/chunleili.u6ey/Dev/miniforge3/etc/profile.d/conda.sh
                                conda activate poseidon
                                export PYTHONPATH=$(pwd)

                                python -u SAFE_SVD/safe_svd.py \
                                    --model_name '$MODEL' \
                                    --dataset '$ds' \
                                    --traj_num $traj \
                                    --calibration_seed $seed \
                                    --final_time $ftime \
                                    --rank_ratio $rank_ratio \
                                    --order_control $order \
                                    --gradient_loss '$LOSS' \
                                    --alpha $alpha \
                                    --compression_backend '$COMPRESSION_BACKEND' \
                                    --whitening_mode '$whitening_mode' \
                                    --whitening_eps_mode '$whitening_eps_mode' \
                                    --whitening_eps '$whitening_eps'$RANK_SOURCE_ARGS $GPU_PROFILE_ARG
                        "

                    sleep 0.5
                done
            done
        done
    done
done

echo "All safe SAFE-SVD jobs submitted. Use 'squeue -u $USER' to check status."
