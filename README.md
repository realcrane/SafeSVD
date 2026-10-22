# SAFE-SVD v0

## Prepare.
### Repo. clone
VICON is included in this repository under `VICON/`, using the research
`ICON_CROPPED` architecture that matches our VICON checkpoint. No separate
VICON clone is needed. See [source provenance](VICON/VENDORED.md).
Before downloads, clone the Poseidon and MPP repositories under SAFE-SVD.
For Poseidon, use an unmodified checkout of the official repository; do not
reuse the local `poseidon_main` research fork:

```bash
git clone https://github.com/camlab-ethz/poseidon.git poseidon
```

The resulting structure is `SAFE-SVD-v0/poseidon/`,
the bundled `SAFE-SVD-v0/VICON/`, and
`SAFE-SVD-v0/mpp/multiple_physics_pretraining/`. MPP needs the
full checkout containing `multiple_physics_pretraining/models/avit.py`.
Poseidon and MPP are upstream source dependencies. VICON's model, data and
rollout source are maintained directly in this repository.

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

### VICON and MPP: from official sources, without a research checkout

This workflow starts with this repository and public upstream sources. It does
not require SMC, another user's data directory, or hard-linked research assets.
Model weights are a separate download from `Evave7/SAFE-SVD`; they are not
downloaded by the dataset commands below.

#### 1. Get the pinned official source repositories

Run these Bash commands at the SAFE-SVD repository root. If a target checkout
already exists, inspect it instead of overwriting it. The helper rejects a
different revision or tracked local modifications to these upstream sources.

```bash
# VICON source is already bundled under ./VICON; keep that version.
git clone https://github.com/PolymathicAI/multiple_physics_pretraining.git mpp/multiple_physics_pretraining
git -C mpp/multiple_physics_pretraining checkout --detach e751fc25ae5274c3ddc869bf73afd6fcd4163cb5
git clone https://github.com/pdebench/PDEBench.git data/upstream/PDEBench
git -C data/upstream/PDEBench checkout --detach 4ff3e3a4aa1561721b5571fa3a048a0a463e0568
```

Install Git LFS using your platform's package manager. The data workflow needs
Python with `h5py`, `numpy`, `pandas`, `tqdm`, and a compatible `torch` /
`torchvision` pair; PDEBench's original downloader imports torchvision.
This step does not install all dependencies needed for model compression.

```bash
python -m pip install h5py numpy pandas tqdm
git lfs version
python -c 'from torchvision.datasets.utils import download_url; import h5py, numpy, pandas, tqdm'
python downloads/datasets/prepare_official.py plan
```

#### 2. Download the original PDEBench files

The [MPP repository](https://github.com/PolymathicAI/multiple_physics_pretraining)
defines dataset paths, loader types, filters, and an 80%/10%/10% sample split in
its `basic_config`. It does not provide a separate turnkey download command.
Its AViT-L configuration uses shallow-water, incompressible NS, compressible
CFD Rand (128 and 512), CFD Turb, and diffusion-reaction. Follow the
[PDEBench download instructions](https://github.com/pdebench/PDEBench/blob/4ff3e3a4aa1561721b5571fa3a048a0a463e0568/pdebench/data_download/README.md)
for these sources:

```bash
python downloads/datasets/prepare_official.py download-pdebench
```

This uses the **unmodified official** `parse_metadata` and `download_url`
functions for `2d_cfd`, `ns_incom`, `swe`, and `2d_reacdiff`. Metadata selection
runs in `data/upstream/PDEBench/pdebench/data_download`, because the official
script reads its CSV relative to the current directory. Files are downloaded
to `data/official/mpp/PDEBench_data` with the original MD5 check intact.

The SAFE-SVD wrapper adds up to three attempts per file (60 seconds between
attempts), retains failed files in `data/official/quarantine/pdebench`, and
writes a receipt only after the original checksum check passes. Unchanged
verified files can be reused on restart; changing their size, modification
time, inode, or device invalidates the receipt. Two attempts with the same
wrong checksum stop retries for that file, while other files continue. Any
unresolved failure makes the stage fail; five failed files stop the stage
early for investigation. The upstream source checkout is not edited.

The pinned complete `2d_cfd` manifest has **17 files**, not only the 8 Rand/Turb
files selected by the training config. The complete raw download therefore
has **293 files**: 17 CFD + 274 NS + 1 shallow-water + 1 diffusion-reaction.
The NS manifest has no `ns_incom_inhom_2d_512-49.h5`; do not invent a missing URL.

**Capacity:** this is a multi-TB download, not a quick setup command. Check disk
space and quota first, allow additional space for VICON conversions and Git LFS
storage, and use tmux for long-running stages. Retained failures also use disk
space. The wrapper retries whole files and reuses verified receipts; it does
not add HTTP range resume to PDEBench's downloader.

#### 3. Download VICON's PDEArena data with Git LFS

The [bundled VICON preparation guide](VICON/dataset_prepare/README.md)
uses `pdearena/NavierStokes-2D-conditoned` and excludes a known corrupted file.

```bash
python downloads/datasets/prepare_official.py download-pdearena
```

The wrapper clones that official dataset repository with Git LFS smudging
initially disabled, checks out revision
`777ab1693937a050f4e22c271c4c4797b24952ab`, then runs `git lfs pull` for HDF5 files
except `NavierStokes2D_train_496019_0.38774_32.h5`. Instead of downloading and
deleting the known-bad payload, it retains the small excluded LFS pointer in
`data/official/excluded/`, outside the loader directory. This yields 116 usable
NS2D files (78 train / 19 valid / 19 test), materialized as regular files.

#### 4. Run VICON's original conversion and splitting functions

```bash
python downloads/datasets/prepare_official.py preprocess-vicon --split-seed 0
```

The wrapper calls, in order, the original functions in
`convert_PDEBench_highVis.py`, `convert_PDEBench_lowVis.py`, and
`split_PDEBench.py`; it supplies paths rather than editing their hard-coded
example paths. The algorithms are not reimplemented:

- High viscosity: 128-resolution Rand files, 500 trajectories per block.
- Low viscosity: 512-resolution Rand and Turb files, the upstream 4-by-4 spatial
  mean to 128 resolution, 50 trajectories per block.
- Each category: 80 blocks, divided into 64 train / 8 valid / 8 test files.

The original split script does not fix a random seed. The wrapper seeds NumPy
and records the actual split-to-original-block mapping in each category's
`conversion_receipt.json`. A seed alone does not fix filesystem glob order,
and a fresh split is **not claimed to reproduce the original paper's exact
split**. Keep the mapping when comparing experiments. Existing conversions
are never silently overwritten or reshuffled; failed staging directories are
retained for inspection. The upstream converters can require several GB of RAM.

#### 5. Adapt MPP's paths and verify the prepared files

```bash
python downloads/datasets/prepare_official.py write-mpp-config
python downloads/datasets/prepare_official.py verify
```

MPP's official AViT-L YAML uses `~/PDEBench/...`, not this repository's data
layout. `write-mpp-config` creates `data/official/configs/mpp_avit_L.yaml` by
changing only the six training and six validation root paths in `basic_config`.
The model architecture, data categories, filters, and split fractions remain
unchanged. The upstream checkout is not edited.

`verify` requires current full-MD5 download receipts for all 293 PDEBench files
and computes the SHA-256 of each of the 116 materialized PDEArena files against
its pinned Git LFS pointer. It also checks HDF5 readability, VICON converted
field shapes, split counts, and finite first/last trajectory endpoint fields.
It expects 293 raw PDEBench + 116 PDEArena + 160 converted CFD files,
**569 HDF5 files in total**. These checks do not inspect every physical value
or test model accuracy. Successful stages append provenance to
`data/official/official_preparation.jsonl`; a failed stage is not recorded as
successful.

#### Output layout and compression configuration

All stages accept `--data-root /absolute/new/output` to change the output root;
use the same value for every stage. The default `data/official` keeps any older
server-prepared data separate and never treats those files as a completed fresh
download.

```text
data/official/
├── mpp/PDEBench_data/2D/                     # official raw downloads
├── vicon/VICON/NavierStokes-2D-conditoned/   # official Git LFS checkout
├── vicon/VICON/PDEBench/2D/CFD/converted/
│   ├── 2D_Train_Rand/{train,valid,test}/
│   └── 2D_Train_Turb/{train,valid,test}/
├── configs/mpp_avit_L.yaml
└── official_preparation.jsonl
```

For MPP, use `--data-root /path/to/SAFE-SVD-v0/data/official/mpp` and append
`-- --yaml-config /path/to/SAFE-SVD-v0/data/official/configs/mpp_avit_L.yaml`
to the top-level compression/evaluation command. Select `basic_config`.

For VICON, use `--data-root /path/to/SAFE-SVD-v0/data/official/vicon` and append
the following upstream overrides after `--`. The current adapter does not
automatically rebase the `folder` fields:

```text
--config-override datasets.types.NS2D.folder=/path/to/SAFE-SVD-v0/data/official/vicon/VICON/NavierStokes-2D-conditoned
--config-override datasets.types.COMPRESSIBLE2D.folder=/path/to/SAFE-SVD-v0/data/official/vicon/VICON/PDEBench/2D/CFD/converted/2D_Train_Rand
--config-override datasets.types.EULER2D.folder=/path/to/SAFE-SVD-v0/data/official/vicon/VICON/PDEBench/2D/CFD/converted/2D_Train_Turb
```

To run the small-grid offline integration tests against the official functions:

```bash
python downloads/datasets/test_prepare_official.py
python downloads/datasets/test_pdebench_retry.py
python downloads/datasets/test_diagnose_pdebench_download.py
python downloads/datasets/test_verify_official.py
python downloads/datasets/test_resume_after_diagnosis.py
```

If a full-size file fails MD5 validation, retain it and independently download
the same official object to new paths for comparison:

```bash
python downloads/datasets/diagnose_pdebench_download.py \
  --manifest data/upstream/PDEBench/pdebench/data_download/pdebench_data_urls.csv \
  --filename '<exact filename from the manifest>' \
  --previous-md5 '<MD5 calculated for the first failed copy>' \
  --output /absolute/new/independent-copy.hdf5 \
  --report /absolute/new/diagnosis.json
```

The diagnostic records the incoming-byte MD5 separately from a full reread of
the persisted file. It also attempts an S3 multipart ETag comparison, which
is evidence only when the computed ETag exactly matches. It does not overwrite
the original copy or accept a file with the wrong published checksum.
Transient HTTP/TLS errors can resume within the same running process; the
server must return the exact requested range and the same object ETag. Use
`--network-retries` and `--retry-delay` to configure this bounded retry policy.
A failed process retains its partial copy and failure report; use new paths
for a new independent attempt.

For the default `data/official` layout, `resume_after_diagnosis.py` can wait
for the report and continue preparation automatically. It accepts only a
complete independent transfer whose incoming and persisted MD5 both equal
the original manifest value, retains the first copy in quarantine, and writes
a verification receipt before starting the full pipeline. Its
`--diagnostic-status` input is the companion task-status file generated by the
`downloads/datasets/run_logged_task.sh`; `--run-dir` must be an unused absolute
log directory. For example, inside a persistent tmux session, use an explicit
Bash interpreter (also on servers whose login shell is csh):

```bash
/bin/bash downloads/datasets/run_logged_task.sh diagnosis "$PWD/logs/diagnosis-1" \
  python downloads/datasets/diagnose_pdebench_download.py \
  --manifest data/upstream/PDEBench/pdebench/data_download/pdebench_data_urls.csv \
  --filename '<exact filename from the manifest>' \
  --output /absolute/new/independent-copy.hdf5 \
  --report /absolute/new/diagnosis.json
# Companion status: logs/diagnosis-1/status.diagnosis.txt
```

Unexpected or mismatching results stop this controller for investigation.

To run the complete sequence in a persistent terminal, after the source and
environment setup above, use the following Bash wrapper. The log directory
must not already exist. This command downloads the full multi-TB dataset; it
does **not** remove existing data and does not start compression.

```bash
tmux new -s official-data
bash downloads/datasets/run_official_pipeline.sh "$(command -v python)" "$PWD/logs/official-data-run-1"
```

Create the `logs` parent directory first if it is absent. The wrapper runs the
offline tests, then the download, conversion, configuration, and verification
stages in sequence. It stops on the first failure. Check `status.txt` and
`pipeline.log` in the chosen log directory; only a final `SUCCESS` means that
all preparation stages and the structural check passed. It is not a model
loading or compression-accuracy test. Re-running with a new log directory
uses the upstream downloaders' existing-file checks; it is not automatically
a clean download. Incomplete conversion directories still require inspection.

**Validation boundary:** a successful plan, dependency check, or small-grid test
is not a full dataset download or a compression test. Only report full data
preparation after all download, conversion, and verification stages succeed.
Older server reuse records (`data/vicon_mpp_manifest.json`, or archived copies
under `data/preparation_history/`) are migration records, not evidence that
this clean-source workflow has completed.

**Recovery record (2026-10-10):** the first complete copy of official PDEBench
object `164688` had MD5 `a0511a3b532c4fb4814e895b5d888c0a`, independently
reproduced by reading the same file on two hosts. A subsequent independent
transfer on `barnacle` failed with `DECRYPTION_FAILED_OR_BAD_RECORD_MAC` and
was retained. On `crested`, both the complete received bytes and a complete
post-write disk reread matched the original manifest MD5
`b2733888745a1d64e36df813e979910b`; its multipart ETag also matched the server.
The source checksum is therefore not the cause of this failure. The exact
low-level cause of the first corrupted copy remains unproven; a TLS error on
another attempt alone does not establish that cause. The controller preserved
the first copy in `data/official/quarantine/first_attempt/`, promoted only the
fully verified independent copy, and resumed preparation. Evidence is in
`data/official/diagnostics/164688/attempt3_crested.json` and
`logs/data_official_recovery_20261010/`; the active full pipeline uses
`logs/data_official_full_resume_20261010/`. A later read-only check confirmed
that `barnacle` rebooted, interrupting its PDEArena transfer; its old `RUNNING`
status is stale. The full pipeline resumes missing Git LFS objects using the
same pinned official checkout. This recovery record is not a claim that all
datasets or loader checks have finished.

## Foundation and Finetuned Model Downloads
We have totally n finetuned poseidon models, and one VICON and MPP foundation model for user to easy strat compression. Please run the command below for downloads.
```python
python downloads/model_download.py
```

## Unified entry point: `safe_svd/main.py`

This entry point loads the selected model and calibration data, then runs input
collection, Fisher collection, rank selection, and compression in order. All
compression and calibration helpers live in this repository. It uses official
model checkouts under `mpp/multiple_physics_pretraining/` and `poseidon/scOT/`,
and the bundled research VICON source under `VICON/`.
It does not load another research repository or the `poseidon_main` fork.
There is no `--source_root` option.

From the SAFE-SVD repository root:

```bash
python -m safe_svd.main --model mpp \
  --ckpt_path ./models/MPP/MPP_AViT_L \
  --yaml_config ./data/official/configs/mpp_avit_L.yaml \
  --data_root ./data/official/mpp --rank_ratio 0.4

python -m safe_svd.main --model vicon \
  --ckpt_path ./models/VICON/model.pth \
  --data_root /path/to/vicon_data --rank_ratio 0.8

python -m safe_svd.main --model poseidon \
  --model_path ./models/poseidon/fluids.incompressible.PiecewiseConstants \
  --data_path ./poseidon/camlab-ethz/down_streams \
  --dataset fluids.incompressible.PiecewiseConstants --rank_ratio 0.2
```

VICON's `--data_root` should contain the dataset subdirectories named in its
Hydra config; use repeated `--config_override datasets.types.NAME.folder=PATH`
for a different layout. MPP's YAML lists its train/validation directories;
relative entries are resolved under `--data_root` and absolute entries are
used as written. Explicit command-line paths are relative to the invocation
directory. Default checkpoint paths match `model_download/model_download.py`.
Default input/Fisher caches, ranks, and compressed models are written under
this repository's `artifacts/{cache,ranks,compressed}/`, even when invoked
from another directory.

Use `python -m safe_svd.main --help` to see all options and append `--dry_run`
to inspect the resolved configuration without loading any model or data.
See [all parameter definitions](safe_svd/PARAMETERS.md) for defaults, supported
models, batch-size controls, cache reuse and Poseidon memory guidance.
`python safe_svd/main.py ...` and the installed `safe-svd-unified ...` command
use the same entry point. MPP/VICON default to `--rank_selector legacy` and
Poseidon to `safe`; `--rank_selector safe` selects the shared strict-budget
allocator for any model.

## VICON research alignment

Both `python -m safe_svd.main --model vicon` and `python -m safesvd.cli compress
--model vicon` use the same compression implementation. For the original MySVD
route, use `legacy` ranks, fixed calibration (`random_calibration=n`), batch size
128, alpha 0.7, L1 Fisher with balance ratio 10, and trace alignment enabled.
These are VICON CLI defaults; the public CLI still requires `--calibration-size`
(use 2048 to match the research run). `safe` remains an explicit alternative.
Input/Fisher Gram accumulation and float32 dewhitening match the research code;
alpha zero still uses the sequential comb algorithm.

New automatically generated caches and default outputs are isolated from older
VICON runs (`vicon_research_v2` or a versioned cache-key hash). Caches preserve
calibration RNG state so cold and warm runs consume the same subsequent data.
Old caches lacking that state require recomputation. Old models are not modified:
compression changes require a new run and a new output directory. Explicit
output/cache paths remain the caller's responsibility.

Evaluate a newly compressed checkpoint with:

```bash
python -m safesvd.cli evaluate --model vicon \
  --checkpoint ./models/VICON/model.pth --data-root /path/to/vicon_data \
  --compressed-model /path/to/new_run/compressed_model.pth \
  --output /path/to/new_run/metrics_research.json -- --protocol research
```

The default `research` protocol follows a fresh research evaluation: native
origin loss → origin physics → compressed loss → compressed physics, advancing
the same test dataset RNG stream (test seed 84, batch size 128). It does **not**
reuse historical CSV baselines. Historical CSVs may have skipped earlier models,
so their samples cannot necessarily be reconstructed from the seed alone.

For identical time-pair samples across models and metric passes, use
`-- --protocol paired` and a separate `metrics_paired.json` output. JSON records
the protocol, test seed, batch limits, label energies and per-dataset batch
counts. `paired` is a controlled comparison, not the original advancing-stream
protocol. Linear parameter counts include native fused Q/K/V projections without
converting the origin model's forward pass.

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
# ViCON uses the research source bundled under ./VICON.
python -m safesvd.cli compress \
  --model vicon \
  --source-root . \
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
using the upstream validation/test splits and research evaluation protocol.
The output JSON records aggregate and
per-dataset physical metrics; `--max-batches` is useful for a quick smoke
test.

```bash
python -m safesvd.cli evaluate \
  --model vicon \
  --source-root . \
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

MPP defaults to batch size 1 and at most 64 batches per validation file,
with shuffle seed 0. VICON defaults to batch size 128 and all test batches,
using `test_seed` from its configuration (84 by default). For each VICON
model, the loss pass precedes the physics pass. The same Dataset instances
are shared across passes and models, so time-pair RNG state advances in the
same order as the original research evaluator.

For VICON, `--data-root` contains `2D_Train_Rand`, `2D_Train_Turb`, and
`NavierStokes-2D-conditoned`. Dataset `folder` paths are resolved below this
root; legacy absolute paths and paths containing `..` use their basename.
Explicit `--config-override datasets.types.<name>.folder=...` values take
precedence (pass backend options after `--`).

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
