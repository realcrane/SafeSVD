#!/bin/bash
#SBATCH --job-name=ablation1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/mysvd/ablation1.out

uniform='y'
use_fisher='n'
order=0
alpha=0.7

source ~/miniconda3/etc/profile.d/conda.sh
conda activate poseidon
export PYTHONPATH=$(pwd)

python -u MySVD/mysvd.py \
    --dataset fluids.incompressible.PiecewiseConstants \
    --final_time 20 \
    --traj_num 64 \
    --use_fisher $use_fisher \
    --order_control $order \
    --rank_ratio 0.2 \
    --uniform $uniform \
    --alpha $alpha