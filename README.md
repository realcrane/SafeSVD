# SAFE-SVD v0

## Prepare.
### Repo. clone
Before downloads, clone Poseidon, VICON and MPP repositories under SAFE-SVD.
For Poseidon, use an unmodified checkout of the official repository; do not
reuse the local `poseidon_main` research fork:

```bash
git clone https://github.com/camlab-ethz/poseidon.git poseidon
```

The resulting structure is `SAFE-SVD/poseidon/`, `SAFE-SVD/vicon/`, and
`SAFE-SVD/mpp/`.

### Environment
```bash
conda create -n ...
pip install ...
```

## Dataset Downloads
### Poseidon
We provide python files to downlaod poseidon downstream datasets, run
```python
python downloads/poseidon/datasets/downstreams.py
```

### VICON
Please refer to vicon repo for the downloading. (wait to try again)

### MPP
Please refer to mpp repo for the downloading. (wait to try again)

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
  --calibration-size 64 \
  --dataset fluids.incompressible.PiecewiseConstants \
  --final-time 20 \
  -- \
  --model_name Poseidon-L \
  --fast
```

The default output path is addressed by the compression settings, for example
`artifacts/poseidon/paper/0.2/fluids.incompressible.PiecewiseConstants/traj64_default_ftime20/`.
For Poseidon, `--calibration-seed -1` uses the fixed prefix of training
trajectories. Pass a non-negative seed, such as `--calibration-seed 7`, to
sample a reproducible, different set of complete trajectories; its output
directory is `traj64_seed7_ftime20`.

Poseidon compression always uses MySVD_fast's 512-sample activation-capture
chunk and consumes each teacher/student pair before capturing the next one.
`--fast` is retained as a no-op compatibility flag. Persistent artifacts are backend-scoped:
`ranks/poseidon/`, `cache/poseidon/`, and the run-local `compressed/` folder.
Future routes use the equivalent `ranks/vicon`, `ranks/mpp`, `cache/vicon`,
and `cache/mpp` roots. SAFE-SVD imports only public `scOT.*` modules and keeps
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
  --model /path/to/run/compressed/compressed_model_origin.pt \
  --dataset fluids.incompressible.PiecewiseConstants
```

The JSON output is saved as `metrics.json` in the corresponding run directory
by default. Its compressed-model `increase` values are percentages rounded to
six decimal places.

<!-- SAFE-SVD is a unified, reproducible interface for applying SVD-based
compression to three scientific foundation models:

| Route | Base model | Weight source | Status in v0 |
| --- | --- | --- | --- |
| `poseidon` | Poseidon | Official upstream Hugging Face collection | Paper-reproduction reference |
| `vicon` | VICON | SAFE-SVD Hugging Face model repository | Paper-reproduction reference |
| `mpp` | Multiple Physics Pretraining (MPP) | SAFE-SVD Hugging Face model repository | Paper-reproduction reference |

`v0` is intentionally a **paper-compatible release**. The optimized
`MySVD_fast` implementation and the new SAFE rank-selection method will be
added later as explicitly labelled optional presets; neither silently changes
the paper route.

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

Poseidon's improved SAFE rank-selection and accelerated compression source is
preserved separately at `backends/poseidon/safe_rank/`; it is not the
default `paper` route.

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
