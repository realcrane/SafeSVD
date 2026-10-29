# SAFE-SVD

Official code for the NeurIPS 2026 paper **“SAFE-SVD: Sensitivity-Aware Fidelity-Enforcing SVD for Physics Foundation Models”**

## Installation

### Environment

```bash
conda create -n 
conda activate safe-svd
python -m pip install -r requirement.txt
```

### Repostority

We include three models: Poseidon, VICON and MPP for our experiments. Download their source code fisrt. NOTE: The latest VICON version has a bit difference to our experimental one, so we directly provide the old version in our repostority.

```bash
git clone git@github.com:camlab-ethz/poseidon.git
git clone git@github.com:PolymathicAI/multiple_physics_pretraining.git
```

### Model

We also provide origin models for download.

```bash
python model_download/model_download.py
```

### Dataset

We provide the way to download Poseidon, VICON and MPP datasets. You can also download them by yourself regarding to their repos link1, link2 and link3 respectively.

### VICON

```bash
mkdir data/raw_data
cd data
git clone https://github.com/pdebench/PDEBench.git
cd PDEBench/pdebench/data_download
python download_direct.py \ 
	--root_folder  ../../../raw_data \ # /Path/to/raw_data/
	--pde_name 2d_cfd \ # for VICON and MPP
	--pde_name swe \ # MPP
	--pde_name 2d_reacdiff \ # MPP
	--pde_name ns_incom # MPP
cd ../../../raw_data
git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D-conditoned #MPP
# Data preprocess
cd ../.. # Path/to/SAFE-SVD
python data/prepare/vicon/convert_PDEBench_highVis.py
python data/prepare/vicon/convert_PDEBench_lowVis.py
python data/prepare/vicon/split_PDEBench.py
cp -i configs/mpp/data/mpp_avit_L_config.yaml multiple_physics_pretraining/config/mpp_avit_L_config.yaml
```

## Quick Start

### Compression

```bash
python -m safe_svd.main \
  --model poseidon \
  --model_path models/poseidon/fluids.incompressible.PiecewiseConstants \
  --data_path poseidon/camlab-ethz/down_streams \
  --dataset fluids.incompressible.PiecewiseConstants \
  --DEV cuda \
  --rank_ratio 0.2 \
  --rank_selector legacy \
  --traj_num 16 \
  --calibration_seed default \
  --final_time 10 \
  --alpha 0.7 \
  --save_path ./artifacts/poseidon/0.2/
```

### Evaluation

```bash
python evaluation/poseidon/evaluate.py \
	--poseidon-root ./poseidon \
	--data-root poseidon/camlab-ethz/down_streams \
	--checkpoint models/poseidon/fluids.incompressible.PiecewiseConstants \
	--model artifacts/poseidon/0.2/compressed_model.pt
```

## Citation

If this code is used in academic work, cite this paper and the original SAFE-SVD work.

```bash
@article{hong2026safe,
  title={SAFE-SVD: Sensitivity-Aware Fidelity-Enforcing SVD for Physics Foundation Models},
  author={Hong, Chengjie and He, Feixiang and Zeng, Yiheng and Kang, Lulu and Wang, He},
  journal={arXiv preprint arXiv:2605.17985},
  year={2026}
}
```