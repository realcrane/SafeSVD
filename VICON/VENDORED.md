# VICON source included in SAFE-SVD

This directory is part of the SAFE-SVD Git repository, not a submodule or a
separate checkout. Keep it when cloning, distributing or installing SAFE-SVD.
Model weights, datasets, caches and experiment outputs are downloaded or
generated separately and must not be committed here.

## Source

- Upstream: https://github.com/Eydcao/VICON
- Research working-tree snapshot: `physic_model_comp/vicon/VICON`, retrieved
  from the experiment server on 2026-10-16.
- Research repository HEAD: `b2a98a25e9c4e2b560593f3ce62ee858c0bdbc4d`.
- The snapshot includes the research working tree's local modifications; the
  commit alone does not reproduce it. In particular, it retains the cropped
  model and the original data and rollout implementations used in experiments.
- `src/models.py` snapshot SHA-256:
  `e11b4dc3358c1b88b72b2a4e963dba5caf5dae23f6a55862db43d89d4640fa91`.
- Compression entry-point alignment was checked against the server's
  `MySVD/safe_svd.py`, `rank.py`, and `mysvd.py` on 2026-10-23. That wrapper
  defaults to fixed calibration (`random_calibration=n`), calibration seed 42,
  and rank ratio 0.8; both stage scripts use batch size 64.

The upstream MIT license is retained in `LICENSE`.

## SAFE-SVD integration

The model default is `type: crop`, which constructs `ICON_CROPPED`. Its four
position embeddings match `models/VICON/model.pth`: input/output patch
embeddings of shape `(64, 1024)` and input/output function embeddings of shape
`(10, 1024)`. This checkpoint was verified to load strictly on CPU against the
research snapshot before vendoring.

The vendored model and dataset implementations retain the research behavior.
Integration includes package imports, dataset folder defaults, package metadata,
and a separate file-order manifest applied by SAFE-SVD's calibration loader:

- `configs/datasets/default.yaml` uses category folder names. The unified
  entry point rebases them under `--data_root`; explicit
  `--config_override datasets.types.NAME.folder=/absolute/path` takes priority.
- `pyproject.toml` no longer lists `torchdata`. This snapshot uses
  `torch.utils.data.IterDataPipe` directly. The historical Poetry lock file is
  retained as source provenance, not as SAFE-SVD's installation instructions.
- Install runtime dependencies using SAFE-SVD's root `requirements.txt`, and
  install SAFE-SVD from its root. No separate VICON clone or installation is
  required. Configuration YAML files and the order JSON are included in the wheel.
- `configs/research_calibration_order.json` records the research server's
  initial train-file enumeration on 2026-10-23. The unified CLI applies it by
  default using paths relative to each locally configured category folder.
  Categories, their order, and file sets must match. No research source tree
  or source data path is accessed at runtime. Use `--vicon_file_order PATH`
  for another manifest; `--vicon_file_order filesystem` keeps the current
  directory enumeration and does not guarantee research sampling alignment.
- Automatically generated VICON cache/rank/output tags include
  `vicon_research_v3`. Default cache keys include configuration content hashes
  and a file-order fingerprint, in addition to checkpoint/path/settings
  identity. The manifest establishes ordering, not equal HDF5 contents or
  equal historical data splits.

Run compression from the SAFE-SVD root:

```bash
python -m safe_svd.main --model vicon \
  --ckpt_path models/VICON/model.pth \
  --data_root /path/to/converted \
  --config_override datasets.types.NS2D.folder=/path/to/NavierStokes-2D-conditoned \
  --rank_ratio 0.8 --rank_selector legacy \
  --calibration_num 2048 --batch_size 64 \
  --random_calibration n --calibration_seed 42 \
  --save_path artifacts/vicon_r08_research_v3
```

For VICON, CLI `--pipeline_stage all` starts separate `rank_selection` and
`compression` subprocesses, matching the original wrapper's stage boundary.
The second process reloads the model and datasets and reads the statistics and
ranks. Input/Fisher caches contain factors and version metadata (Input also
records sample counts), with no RNG snapshot; loading them does not restore
the preceding stage's RNG state. Old v1/v2 structured caches are rejected.
Explicit original factor dictionaries can be read, but users must verify their
checkpoint, data, sampling, and numerical settings before reuse. Existing
caches and outputs should be preserved in separate paths.

Compression checkpoints separately save the RNG and path-order state from
before calibration-batch prefetch. Resuming a v3 checkpoint restores that state
to rebuild the same prefetched samples; it does not reuse v1/v2 checkpoints.
Keep data, batch size, workers, and sampling settings unchanged when resuming.

Matching these execution rules does not guarantee byte-identical historical
weights or equal full evaluation results. The research train dataset generates
its seed from the global NumPy RNG and ignores the supplied base seed. Input
profiling seeds that global RNG only when it actually runs; a cache hit skips
the seeding, so a historical compression process may have used an unrecorded
random state. Numerical equivalence requires matching the actual samples,
checkpoint and data contents, caches, and effective settings, then comparing
intermediate statistics/ranks and final metrics.

Training-specific paths in `configs/default.yaml` are historical; compression
loads the explicitly supplied `--ckpt_path` instead. The Python `main()` API
does not spawn subprocesses: for VICON `all` it restores the entry loader/global
RNG state before compression instead of continuing the rank-selection stream.

## Alignment validation (2026-10-23)

The local CPU regression suite passed 85 tests without skips against the latest
server MySVD snapshot. A three-category, two-layer fixture compared input and
Fisher induced metrics, ranks, compressed effective operators and predictions.
With Torch 2.4.1, all measured maximum absolute and relative L2 differences were
zero; ten rank dictionaries matched exactly. The comb comparison also covered
alpha 0/0.7/1, trace on/off, and Fisher present/absent. Test tolerances are
atol=1e-6/rtol=1e-5 for input/Fisher and atol=2e-6/rtol=2e-5 for comb/attention;
rank and RNG replay checks are exact. These are controlled CPU fixtures, not a
full-checkpoint compression/evaluation result.

The validation report is saved under
`artifacts/validation/vicon_research_v3_20261023.json` when available (generated
artifacts are not included in the source package).

The server's old `data/processed_data/2D/CFD/converted` copy had different CFD
train/valid/test assignments and renumbered files. In each CFD category, only
53 of the research training split's 64 files belonged to that copy's train
split. None of the research test split's eight files belonged to its test split.
Matching filenames alone therefore did not establish matching data.

A unique mapping for all 80 files per CFD category was established using
48 exact sampled values plus field shape/dtype per file. A new server-only
`data/VICON` tree reproduces the research split and logical filenames using
160 links to SAFE's existing CFD files and a link to SAFE's NS2D directory.
It does not depend on the research repository and leaves existing data intact.
Its `research_alignment.json` records the mapping. The unified CLI prefers
this tree; commands that explicitly select the old processed-data root must
instead pass `--data_root data/VICON` for this comparison.

From the actual new tree, three categories, two consecutive passes, and two
samples per category/pass (batch=1, controlled seed=42) produced 72 tensors
with identical SHA-256 and maximum absolute difference zero, including time
pairs. This checks real input samples, not every HDF5 byte or full batch-64
compression. The same 85-test CPU suite passed on the server with Torch
2.5.1+cu121; one additional test covers the new default-data-root preference.
