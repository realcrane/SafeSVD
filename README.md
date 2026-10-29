# SAFE-SVD

Official code for the NeurIPS 2026 paper **[“SAFE-SVD: Sensitivity-Aware Fidelity-Enforcing SVD for Physics Foundation Models”](https://arxiv.org/abs/2605.17985)**.

SAFE-SVD compresses physics foundation models while preserving predictive accuracy and physical fidelity. It combines sensitivity estimates from physics-aware losses, sequential layer compression, and adaptive rank allocation. This repository supports **Poseidon**, **VICON**, and **Multiple Physics Pretraining (MPP)**.

## Installation

### 1. Set up the environment

The dependency snapshot in [`requirement.txt`](requirement.txt) targets **Linux x86_64, Python 3.11, and PyTorch 2.5.1 with CUDA 12.1**. Use an NVIDIA driver compatible with this CUDA build. The commands below assume Git and Conda are installed.

```bash
git clone https://github.com/Evave1204/SAFE-SVD-v0.git
cd SAFE-SVD-v0

conda create -n safe-svd python=3.11 -y
conda activate safe-svd
python -m pip install -r requirement.txt
python -m pip install -e . --no-deps
```

Run subsequent commands from the SAFE-SVD repository root unless a command explicitly changes directory. Relative paths in the examples are resolved from that directory.

[`environment-unified.yml`](environment-unified.yml) describes a separate CUDA 12.8/PyTorch 2.11 environment. Use one dependency specification per environment; the examples here use `requirement.txt`.

### 2. Obtain the upstream model source

Download only the source needed for your model:

```bash
# Poseidon
git clone https://github.com/camlab-ethz/poseidon.git poseidon
python -m pip install -e ./poseidon --no-deps

# MPP: use the nested location to avoid an existing local config directory.
mkdir -p mpp
git clone https://github.com/PolymathicAI/multiple_physics_pretraining.git mpp/multiple_physics_pretraining
```

Poseidon's `--no-deps` installation preserves the shared environment instead of installing its older PyTorch pins. MPP is imported directly from its checkout; no separate installation is needed. The flat location `multiple_physics_pretraining/` is also supported if it contains the complete source, including `models/avit.py`.

VICON's research source is already included in [`VICON/`](VICON/). Its implementation differs from the latest upstream version, so use the bundled copy with the supplied checkpoint. See [`VICON/VENDORED.md`](VICON/VENDORED.md) for provenance and reproducibility details.

**Current source limitation:** `safe_svd/main.py` imports `safesvd.adapters.mpp_paths`, but that module's source is absent from this checkout. The equivalent path-discovery helper exists in `evaluation/mpp/paths.py`. Repair this import or restore the missing module before running the unified compression entry point. Installing the upstream models does not supply this SAFE-SVD module.

### 3. Download the original checkpoints

Download the checkpoints provided in the [SAFE-SVD model repository](https://huggingface.co/Evave7/SAFE-SVD):

```bash
# Authenticate first if the model repository requires access approval.
hf auth login
python model_download/model_download.py
```

For only the Poseidon checkpoint used in the quick start:

```bash
python model_download/model_download.py \
  --poseidon-only \
  --dataset fluids.incompressible.PiecewiseConstants
```

The downloader creates this layout. Poseidon checkpoints are directories containing their configuration and weights; VICON and MPP checkpoints are files.

```text
models/
├── poseidon/
│   └── fluids.incompressible.PiecewiseConstants/
│       ├── config.json
│       └── ...
├── VICON/
│   └── model.pth
└── MPP/
    └── MPP_AViT_L
```

Existing files are skipped unless `--force` is supplied. Use `--output-dir` to change the destination, then pass the corresponding checkpoint path to compression and evaluation. If access is denied, obtain access to the model repository or supply a compatible checkpoint using `--model_path`.

## Dataset preparation

Prepare only the datasets required by your chosen model. Downloads and assembled files can require substantial disk space. Keep raw data separate from VICON's converted data.

### Poseidon

The quick start uses **NS-PwC**, whose code identifier is `fluids.incompressible.PiecewiseConstants`. Download its [dataset repository](https://huggingface.co/datasets/camlab-ethz/NS-PwC), then assemble the chunks into the NetCDF file expected by Poseidon:

```bash
hf download camlab-ethz/NS-PwC \
  --repo-type dataset \
  --local-dir data/poseidon/NS-PwC

mkdir -p poseidon/camlab-ethz/down_streams
python data/poseidon/NS-PwC/assemble_data.py \
  --input_dir data/poseidon/NS-PwC \
  --output_file poseidon/camlab-ethz/down_streams/NS-PwC.nc
```

For other tasks, use the [Poseidon dataset guide](https://github.com/camlab-ethz/poseidon#datasets) and follow the assembly instructions in the corresponding dataset repository. Set `--dataset` to its **code identifier**, use a matching fine-tuned checkpoint, and point `--data_path` at the directory containing the assembled files.

### Shared PDEBench downloads for VICON and MPP

VICON uses PDEBench's 2D CFD data and PDEArena's incompressible Navier–Stokes data. The supplied MPP configuration uses PDEBench's 2D CFD, shallow-water, diffusion–reaction, and incompressible Navier–Stokes datasets.

```bash
mkdir -p data/raw_data
git clone https://github.com/pdebench/PDEBench.git data/PDEBench

# Run here because the downloader reads its metadata CSV from this directory.
cd data/PDEBench/pdebench/data_download
python download_direct.py \
  --root_folder ../../../raw_data \
  --pde_name 2d_cfd \
  --pde_name swe \
  --pde_name 2d_reacdiff \
  --pde_name ns_incom
cd ../../../..
```

For VICON alone, keep only `--pde_name 2d_cfd`. See the [PDEBench download instructions](https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download) for alternative download methods.

### VICON conversion and directory layout

The bundled preprocessing scripts' standalone entry points contain historical relative paths. The following command instead calls their functions with explicit paths. It creates `data/VICON` and splits each converted CFD category into 80% train, 10% validation, and 10% test files.

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from VICON.dataset_prepare.convert_PDEBench_highVis import convert as convert_high
from VICON.dataset_prepare.convert_PDEBench_lowVis import convert as convert_low
from VICON.dataset_prepare.split_PDEBench import split

raw = Path("data/raw_data/2D/CFD")
output = Path("data/VICON")
if output.exists():
    raise SystemExit("data/VICON already exists; use a fresh conversion directory.")

convert_high(str(raw / "2D_Train_Rand"), str(output / "2D_Train_Rand"), 500)
convert_low(
    [str(raw / "2D_Train_Rand"), str(raw / "2D_Train_Turb")],
    str(output / "2D_Train_Turb"),
    50,
)
np.random.seed(42)
split(str(output / "2D_Train_Rand"))
split(str(output / "2D_Train_Turb"))
PY

# Git LFS must be installed for the PDEArena data files.
git lfs install
git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D-conditoned \
  data/VICON/NavierStokes-2D-conditoned
```

The expected layout is:

```text
data/VICON/
├── 2D_Train_Rand/
│   ├── train/*.h5
│   ├── valid/*.h5
│   └── test/*.h5
├── 2D_Train_Turb/
│   ├── train/*.h5
│   ├── valid/*.h5
│   └── test/*.h5
└── NavierStokes-2D-conditoned/
    └── *.h5
```

This creates a local split; seed 42 does **not** establish that it matches the paper's historical split. For these newly prepared data, the example below uses `--vicon_file_order filesystem`. Use the default research manifest only with data whose category file sets, split assignments, and contents match the research data. Matching filenames alone is insufficient. See [`VICON/VENDORED.md`](VICON/VENDORED.md) and the [preprocessing notes](VICON/dataset_prepare/README.md), including the upstream note about a corrupted PDEArena file.

If PDEArena is stored elsewhere, set `--config_override datasets.types.NS2D.folder=/absolute/path` during compression and the corresponding `--config-override` during evaluation.

### MPP configuration

MPP reads the **raw PDEBench files**, not VICON's converted files. With `--data_root data/raw_data`, the supplied [`mpp_avit_L_config.yaml`](safe_svd/configs/mpp_avit_L_config.yaml) expects:

```text
data/raw_data/2D/
├── shallow-water/
├── NS_incom/
├── CFD/
│   ├── 2D_Train_Rand/
│   └── 2D_Train_Turb/
└── diffusion-reaction/
```

Pass this YAML explicitly to both compression and evaluation; no copy into the upstream repository is needed. Relative data paths in the YAML are resolved under the data root. If your downloads use a different layout, edit a copy and use that same configuration for both stages. Keep the architecture and state configuration compatible with the checkpoint.

## Compression and evaluation

The unified pipeline performs input profiling, Fisher collection, rank allocation, sequential compression, and checkpoint saving. Resolve the source limitation noted above before running it.

These examples demonstrate the workflow with explicit settings. Reproducing a particular paper result also requires the corresponding checkpoint, data split, calibration settings, and numerical configuration.

### Poseidon quick start

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
  --fisher_batch_size 8 \
  --capture_batch_size 8 \
  --alpha 0.7 \
  --save_path artifacts/poseidon/0.2

python evaluation/poseidon/evaluate.py \
  --poseidon-root ./poseidon \
  --data-root poseidon/camlab-ethz/down_streams \
  --checkpoint models/poseidon/fluids.incompressible.PiecewiseConstants \
  --model artifacts/poseidon/0.2/compressed_model.pt \
  --dataset fluids.incompressible.PiecewiseConstants \
  --num-trajectories 64 \
  --final-time 10 \
  --device cuda \
  --output artifacts/poseidon/0.2/metrics.json
```

`--traj_num` selects complete **training trajectories** for calibration, not a batch size. `--calibration_seed default` uses the first trajectories; an integer selects a random subset. Evaluation's `--num-trajectories` selects test trajectories independently. `--final_time` controls the maximum time-step index for non-elliptic calibration data; evaluation uses `--final-time`.

### VICON

For the locally prepared data layout above:

```bash
python -m safe_svd.main \
  --model vicon \
  --model_path models/VICON/model.pth \
  --data_root data/VICON \
  --DEV cuda \
  --rank_ratio 0.8 \
  --rank_selector legacy \
  --calibration_num 2048 \
  --batch_size 64 \
  --random_calibration n \
  --calibration_seed 42 \
  --vicon_file_order filesystem \
  --save_path artifacts/vicon/0.8

python evaluation/vicon/evaluate.py \
  --data-root data/VICON \
  --checkpoint models/VICON/model.pth \
  --model artifacts/vicon/0.8/compressed_model.pth \
  --batch-size 64 \
  --protocol research \
  --device cuda \
  --output artifacts/vicon/0.8/metrics.json
```

Keep `rank_dict.json` alongside the compressed VICON checkpoint: the evaluator needs it to reconstruct the low-rank layers. Evaluation uses the test split. The default `research` protocol advances the dataset RNG across metric passes and models; `--protocol paired` replays samples for a paired comparison and changes that protocol.

### MPP

```bash
python -m safe_svd.main \
  --model mpp \
  --model_path models/MPP/MPP_AViT_L \
  --data_root data/raw_data \
  --yaml_config safe_svd/configs/mpp_avit_L_config.yaml \
  --config basic_config \
  --DEV cuda \
  --rank_ratio 0.4 \
  --rank_selector legacy \
  --calibration_num 16 \
  --calibration_seed 42 \
  --batch_size 1 \
  --save_path artifacts/mpp/0.4

python evaluation/mpp/evaluate.py \
  --mpp-root mpp/multiple_physics_pretraining \
  --data-root data/raw_data \
  --checkpoint models/MPP/MPP_AViT_L \
  --model artifacts/mpp/0.4/compressed_model.pt \
  --yaml-config "$PWD/safe_svd/configs/mpp_avit_L_config.yaml" \
  --config-name basic_config \
  --batch-size 1 \
  --max-batches 64 \
  --device cuda \
  --output artifacts/mpp/0.4/metrics.json
```

MPP evaluation resolves a relative `--yaml-config` under `--mpp-root`, so this example supplies an absolute YAML path. It evaluates up to 64 batches per validation file with shuffle seed 0. `--calibration_num` controls calibration samples per file independently of this evaluation limit.

All three evaluators measure the original checkpoint first, then the compressed model, and write task/physical metrics and relative changes to JSON. Omit `--model` to evaluate only the original checkpoint.

## Parameters, outputs, and resuming

`--rank_ratio` is the retained **parameter budget for eligible Linear weights**, not a uniform rank fraction or the retained fraction of the entire model. For example, `0.2` requests a 20% budget for those weights; biases and other uncompressed parameters are excluded. The `legacy` selector follows the original allocation rule and may exceed the requested budget when enforcing minimum ranks. Use `safe` for allocation within the budget. The Poseidon CLI defaults to `safe`; the examples explicitly select `legacy`.

See [`safe_svd/PARAMETERS.md`](safe_svd/PARAMETERS.md) for the full parameter reference and model-specific behavior. Some help strings and reference defaults lag the implementation; inspect the resolved arguments for your command:

```bash
python -m safe_svd.main --model poseidon --dry_run
python -m safe_svd.main --help
```

Append `--dry_run` to a compression command to print paths and settings without loading checkpoints or datasets. It does not verify that assets exist or that the model fits in memory.

| Output file | Purpose |
| --- | --- |
| `compressed_model.pt` | Complete compressed Poseidon or MPP model. |
| `compressed_model.pth` | Compressed VICON state dictionary. |
| `rank_dict.json` | Layer ranks; required alongside the VICON model. |
| `run_config.json` | Resolved arguments, paths, and run metadata. |
| `compressing_model.pt` | Intermediate checkpoint, removed after successful final saving. |
| `gpu_profile.csv` | Stage timing and memory when `--gpu_profile` is enabled. |

Input/Fisher statistics and rank allocations are cached under `artifacts/cache/` and `artifacts/ranks/` by default. MPP and VICON also write `rank_run_config.json` for their separate rank-selection stage.

Rerunning the same command resumes an unfinished compression checkpoint and reuses compatible completed caches. Keep the same output directory and effective model/data/algorithm settings. A completed final model causes an error instead of being overwritten; use a new `--save_path` for another run. `--overwrite_cache` forces recomputation and permits replacing the final output.

Use `--pipeline_stage rank_selection` to stop after rank allocation. For compression from existing statistics and ranks, use `--pipeline_stage compression` with explicit `--profile_cache`, `--fisher_cache`, and `--rank_cache` paths. Reuse caches only for compatible checkpoints, data, and calibration settings; generated path tags do not fully verify data or weight contents.

For CUDA out-of-memory errors, reduce Poseidon's `--fisher_batch_size` during Fisher collection, or its `--batch_size` and `--capture_batch_size` during compression. `--compression_backend low-gpu` changes compression calculations to float32 and affects numerical results; it does not reduce Fisher's backward batch size. Keep VICON batch size, workers, and sampling settings unchanged when resuming or comparing research runs. Further memory and recovery guidance is in the parameter reference.

## Citation and acknowledgments

If you use SAFE-SVD in your research, please cite:

```bibtex
@article{hong2026safe,
  title={SAFE-SVD: Sensitivity-Aware Fidelity-Enforcing SVD for Physics Foundation Models},
  author={Hong, Chengjie and He, Feixiang and Zeng, Yiheng and Kang, Lulu and Wang, He},
  journal={arXiv preprint arXiv:2605.17985},
  year={2026},
  url={https://arxiv.org/abs/2605.17985}
}
```

We thank the authors of [Poseidon](https://github.com/camlab-ethz/poseidon), [VICON](https://github.com/Eydcao/VICON), [MPP](https://github.com/PolymathicAI/multiple_physics_pretraining), [PDEBench](https://github.com/pdebench/PDEBench), and [PDEArena](https://github.com/microsoft/pdearena) for their models, code, and datasets. Please also cite the upstream works you use.

This repository is released under the [MIT License](LICENSE). Upstream code, model checkpoints, and datasets retain their respective licenses.
