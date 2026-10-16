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

The upstream MIT license is retained in `LICENSE`.

## SAFE-SVD integration

The model default is `type: crop`, which constructs `ICON_CROPPED`. Its four
position embeddings match `models/VICON/model.pth`: input/output patch
embeddings of shape `(64, 1024)` and input/output function embeddings of shape
`(10, 1024)`. This checkpoint was verified to load strictly on CPU against the
research snapshot before vendoring.

The research Python implementations are copied unchanged. Integration changes
are limited to an explicit package initializer, the three dataset folder
defaults, package metadata, and ignore rules:

- `configs/datasets/default.yaml` uses category folder names. The unified
  entry point rebases them under `--data_root`; explicit
  `--config_override datasets.types.NAME.folder=/absolute/path` takes priority.
- `pyproject.toml` no longer lists `torchdata`. This snapshot uses
  `torch.utils.data.IterDataPipe` directly. The historical Poetry lock file is
  retained as source provenance, not as SAFE-SVD's installation instructions.
- Install runtime dependencies using SAFE-SVD's root `requirements.txt`, and
  install SAFE-SVD from its root. No separate VICON clone or installation is
  required. Configuration YAML files are included in SAFE-SVD's Python wheel.
- Automatically generated VICON cache/rank/output tags include
  `vicon_research_v1`, separating this runtime from previous defaults.

Run compression from the SAFE-SVD root:

```bash
python -m safe_svd.main --model vicon \
  --ckpt_path models/VICON/model.pth \
  --data_root /path/to/converted \
  --config_override datasets.types.NS2D.folder=/path/to/NavierStokes-2D-conditoned \
  --rank_ratio 0.8 --rank_selector legacy \
  --calibration_num 2048 --batch_size 128 \
  --save_path artifacts/vicon_r08
```

This preserves the research model, file ordering, preprocessing and rollout
defaults. Training-specific paths in `configs/default.yaml` are historical;
compression loads the explicitly supplied `--ckpt_path` instead.
