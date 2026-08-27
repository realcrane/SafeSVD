# SAFE-SVD v0

## Prepare.
### Repo. clone
Before downloads, clone Poseidon, VICON and MPP repositories under SAFE-SVD.
For Poseidon, use an unmodified checkout of the official repository; do not
reuse the local `poseidon_main` research fork:

```bash
git clone https://github.com/camlab-ethz/poseidon.git poseidon
```

The resulting structure is `SAFE-SVD-v0/poseidon/`,
`SAFE-SVD-v0/vicon/VICON/`, and
`SAFE-SVD-v0/mpp/multiple_physics_pretraining/`.  ViCON needs the full
upstream checkout containing `VICON/src` and `VICON/configs`; MPP needs the
full checkout containing `multiple_physics_pretraining/models/avit.py`.
The small research snapshots under `physic_model_comp/` do not contain those
gitignored model-source directories, so they are useful as a reference but
cannot run the adapters on their own.

### Environment
```bash
conda create -n poseidon -c conda-forge --override-channels --no-default-packages python=3.11
conda activate poseidon
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# Install the PyTorch CUDA wheel appropriate for the machine first.  PyTorch
# is deliberately not listed in requirements.txt because this choice depends
# on the NVIDIA driver/GPU.  Do not mix a pip CUDA wheel with conda cudatoolkit.
# Example: follow https://pytorch.org/get-started/locally/ for the desired CUDA wheel.
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

# Install the GPU-independent Poseidon/SAFE-SVD dependencies.
python -m pip install -r requirements.txt
python -m pip install -e ./poseidon
python -m pip install -e .
```

## Dataset Downloads
### Poseidon
We provide python files to downlaod poseidon downstream datasets, run
```python
python downloads/poseidon/datasets/downstreams.py
```

### VICON
Download and unpack the data following the upstream ViCON instructions, then
pass its root through `--data-root`.  The adapter applies that root to the
Hydra dataset configuration.  If a configuration uses non-standard field
names, pass one or more upstream Hydra overrides after `--`, for example
`-- --config-override datasets.NS2D.path=/datasets/vicon/ns2d`.

### MPP
Download and unpack the data following the upstream Multiple Physics
Pretraining instructions, then pass the parent data directory through
`--data-root`.  SAFE-SVD rebases the `train_data_paths` and `valid_data_paths`
in MPP's YAML under that directory.

## Foundation and Finetuned Model Downloads
We have totally n finetuned poseidon models, and one VICON and MPP foundation model for user to easy strat compression. Please run the command below for downloads.
```python
python downloads/model_download.py
```

## Poseidon compression

Place the unmodified upstream Poseidon checkout at `poseidon/`, downstream data
at `poseidon/camlab-ethz/down_streams/`, and use the *concrete* fine-tuned
checkpoint directory as `--checkpoint`.  That directory must contain
`config.json` (passing the parent `models/poseidon/` directory is not enough).

Install both packages once in the same Poseidon environment:

```bash
cd /path/to/SAFE-SVD-v0
python -m pip install -e ./poseidon
python -m pip install -e .
```

Then launch compression with that environment's `python`:

```bash
cd /path/to/SAFE-SVD-v0
python -m safesvd.cli compress \
  --model poseidon \
  --checkpoint /path/to/models/poseidon/Poseidon-L-finetuned \
  --data-root /path/to/SAFE-SVD-v0/poseidon/camlab-ethz/down_streams \
  --rank-ratio 0.2 \
  --rank-selector safe \
  --calibration-size 64 \
  --dataset fluids.incompressible.PiecewiseConstants \
  --final-time 20 \
  -- \
  --model_name Poseidon-L \
  --fast
```

The default output path is addressed by the selector and compression settings,
for example
`artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_default_ftime20/`.
For Poseidon, `--calibration-seed -1` uses the fixed prefix of training
trajectories. Pass a non-negative seed, such as `--calibration-seed 7`, to
sample a reproducible, different set of complete trajectories; its output
directory is `traj64_seed7_ftime20`.

`--rank-selector safe` is the default. It charges every mandatory rank-one
factor before allocation, admits only integer ranks that strictly save
parameters, and compares each next singular direction with an explicit
full-rank tail-energy transition. Use `--rank-selector legacy` only to
reproduce the former MySVD-compatible allocation.

The Poseidon backend normally uses the Python interpreter that launches the
public CLI. If that interpreter belongs to another native PyTorch stack (for
example AIMET), point the backend at a dedicated Poseidon environment:

```bash
SAFE_SVD_RUNTIME_PYTHON=/path/to/poseidon-env/bin/python \
python -m safesvd.cli compress ...
```

Backend subprocesses enable Python's faulthandler by default so native crashes
include the active Python stack in stderr.

Poseidon compression always uses MySVD_fast's 512-sample activation-capture
chunk and consumes each teacher/student pair before capturing the next one.
`--fast` is retained as a no-op compatibility flag. Persistent artifacts are backend-scoped:
`ranks/poseidon/{safe,legacy}/`, `cache/poseidon/`, and the run-local
`compressed/` folder. Rank files are selector-scoped so legacy ranks cannot be
reused by the budget-consistent SAFE selector; profiling and Fisher caches stay
shared because selector choice does not change those statistics.
The VICON and MPP routes use the equivalent `ranks/vicon/{safe,legacy}`,
`ranks/mpp/{safe,legacy}`, `cache/vicon`, and `cache/mpp` roots. SAFE-SVD
imports only public `scOT.*` modules and keeps
all Fisher, calibration, rank-selection, and low-rank replacement code inside
SAFE-SVD; the `poseidon/` source tree is never edited.

## Poseidon evaluation

Evaluate a compressed model with Poseidon's original
`get_dataset -> get_trainer -> rollout` path followed by the physical PDE
metrics from `poseidon_analysis/physical_loss.py`:

```bash
python evaluation/poseidon/evaluate.py \
  --poseidon-root ./poseidon \
  --data-root ./poseidon/camlab-ethz/down_streams \
  --checkpoint /path/to/original-finetuned-checkpoint \
  --model /path/to/run/compressed/compressed_model.pt \
  --dataset fluids.incompressible.PiecewiseConstants
```

The JSON output is saved as `metrics.json` in the corresponding run directory
by default. Its compressed-model `increase` values are percentages rounded to
six decimal places.

## ViCON and MPP compression

The same top-level command now routes ViCON and MPP through SAFE-SVD's shared
whitening, Fisher/rank selection, sequential low-rank replacement, resumable
checkpointing, and portable compressed-state format.  Each model retains its
upstream loader and physical loss conventions.

```bash
# ViCON: ./vicon must contain the full upstream ./vicon/VICON checkout.
python -m safesvd.cli compress \
  --model vicon \
  --source-root ./vicon \
  --checkpoint ./models/VICON/model.pth \
  --data-root /datasets/vicon \
  --rank-ratio 0.2 \
  --calibration-size 2048 \
  --calibration-seed 42 \
  --device cuda \
  -- --batch-size 128

# MPP: ./mpp must contain ./mpp/multiple_physics_pretraining.
python -m safesvd.cli compress \
  --model mpp \
  --source-root ./mpp \
  --checkpoint ./models/MPP/MPP_AViT_L \
  --data-root /datasets/mpp \
  --rank-ratio 0.2 \
  --calibration-size 1024 \
  --calibration-seed 42 \
  --device cuda \
  -- --batch-size 4
```

`--calibration-size` is the requested number of samples per ViCON dataset
category and the target number of samples drawn from each MPP physical-data
file.  To preserve the paper routes, ViCON consumes
`floor(calibration-size / batch-size)` batches (so it must be at least one
batch); MPP consumes `ceil(calibration-size / batch-size)` batches per data
file.  The adapter caches profile/Fisher statistics and ranks under `cache/vicon` or
`cache/mpp` and `ranks/vicon` or `ranks/mpp`; the run directory contains
`compressed/compressed_model.pth` (ViCON) or `compressed/compressed_model.pt`
(MPP).

## Unified evaluation

Evaluate the original checkpoint and, optionally, the compressed checkpoint
on identical upstream test data.  The output JSON records aggregate and
per-dataset physical metrics; `--max-batches` is useful for a quick smoke
test.

```bash
python -m safesvd.cli evaluate \
  --model vicon \
  --source-root ./vicon \
  --checkpoint ./models/VICON/model.pth \
  --compressed-model artifacts/vicon/paper/<run>/compressed/compressed_model.pth \
  --data-root /datasets/vicon \
  --max-batches 10

python -m safesvd.cli evaluate \
  --model mpp \
  --source-root ./mpp \
  --checkpoint ./models/MPP/MPP_AViT_L \
  --compressed-model artifacts/mpp/paper/<run>/compressed/compressed_model.pt \
  --data-root /datasets/mpp \
  --max-batches 10
```

ViCON reports its upstream masked MSE and relative loss plus task-specific
Sobolev and, where applicable, incompressible-flow divergence and vorticity
metrics.  MPP reports relative error, Sobolev error, and the corresponding incompressible NS,
compressible NS, SWE, or diffusion-reaction physical metrics.

<!-- SAFE-SVD is a unified, reproducible interface for applying SVD-based
compression to three scientific foundation models:

| Route | Base model | Weight source | Status in v0 |
| --- | --- | --- | --- |
| `poseidon` | Poseidon | Official upstream Hugging Face collection | Paper-reproduction reference |
| `vicon` | VICON | SAFE-SVD Hugging Face model repository | Paper-reproduction reference |
| `mpp` | Multiple Physics Pretraining (MPP) | SAFE-SVD Hugging Face model repository | Paper-reproduction reference |

The public route exposes both the budget-consistent `safe` rank selector and
the MySVD-compatible `legacy` selector. Selector names are recorded in run
manifests and artifact paths so published recipes remain reproducible.

## Repository policy

This repository contains code, configuration, small metadata files, and
reproduction instructions. It must not contain foundation-model checkpoints,
datasets, generated rank caches, or compressed checkpoints. Those assets are
downloaded into a user-selected local directory and are excluded by
`.gitignore`.

The exact model identifiers, expected checkpoint filenames, SHA-256 hashes,
and source revision are recorded in [`configs/model_registry.yaml`](configs/model_registry.yaml).
Fill in every `TODO` before the public release.

## Quick-start contract

Each route has the same conceptual workflow:

```text
base checkpoint + calibration data
        -> rank selection -> compression -> evaluation -> metrics.json
```

The public interface has one stable shape across all three models:

```bash
safe-svd compress \
  --model poseidon \
  --checkpoint /path/to/poseidon/checkpoint \
  --data-root /path/to/poseidon/data \
  --rank-ratio 0.2 \
  --calibration-size 64
```

Python callers use the same contract:

```python
from safesvd import SafeSVDConfig, compress

result = compress(SafeSVDConfig(
    model="mpp",
    checkpoint=Path("/path/to/checkpoint"),
    data_root=Path("/path/to/data"),
    output_dir=Path("artifacts/mpp-paper"),
    rank_ratio=0.2,
    calibration_size=16,
    calibration_seed=42,
))
```

The public modules are `safesvd.config`, `safesvd.pipeline`, and
`safesvd.adapters`. The copied paper source lives under `backends/` and is
strictly an internal compatibility layer—not a user entry point:

```text
backends/poseidon/paper/
backends/vicon/paper/
backends/mpp/paper/
```

The improved SAFE rank-selection is implemented in the shared
`safesvd.core.ranks` module and is used consistently by all three adapters.

Install the lightweight routing package from this directory:

```bash
pip install -e /path/to/Project/SAFE-SVD-v0
```

The model-specific runtime dependencies remain those of the selected route.

## Assets

Read [`docs/models.md`](docs/models.md) to obtain base checkpoints and
[`docs/datasets.md`](docs/datasets.md) to obtain and prepare calibration/test
data. Dataset licences and access restrictions remain those of the original
providers; downloading a dataset is the user's responsibility.

## Release checklist

Before changing this repository from private to public, complete every item in
[`docs/release-checklist.md`](docs/release-checklist.md), especially licence
review, a clean-machine download test, and a paper-regression run for all
three routes. -->
