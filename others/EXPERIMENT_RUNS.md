# Experiment Run Ledger

This is the canonical ledger for SAFE-SVD experiments launched from this
repository. It records only SAFE-SVD compression and SAFE-SVD evaluation runs;
unrelated projects and other low-rank methods are out of scope.

Last reconciliation: 2026-10-25 (Asia/Shanghai), targeted MPP OOM/research alignment and VICON research-v3 completion audit; earlier run records below retain their prior check dates.

The remote path `/cs/research/vecg/EasyPDE/SAFE-SVD-v0` is on a shared
filesystem, so identical artifacts observed from several hosts are recorded
once. The local `check_ucl_gpus` probe still reported jump-host DNS errors, but
direct read-only SSH through `knuckles.cs.ucl.ac.uk` reached `brent`, `ruddy`,
and `smew`. This reconciliation verified that the seed-1 and seed-1204
compression jobs reached 815/815 and wrote final compressed models. Their
processes have exited; seed-1204's tmux session remains at an idle shell. No
`metrics.json` exists for either run, so physical evaluation has not been
verified. `brent` and `ruddy` were selected because each had an idle NVIDIA
GeForce RTX 3090 Ti before launch.

The 2026-10-22 reconciliation additionally inspected the completed VICON
`artifacts/vicon_r08` result through the shared filesystem on `mandarin`. Its
compression artifact is complete, but its origin/compressed smoke evaluation
uses different stateful test time-pair traversals; see the completed-run entry
below for the rank hashes and metric evidence.

## 未开始

### `poseidon_legacy_seed1204_r0.2`

- Status: `未开始`
- Outcome: `pending`
- Task: Poseidon SAFE-SVD compression artifact found without a live process.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; calibration size `64`; calibration seed `1204`; manifest `rank_selector=null`; `Poseidon-L`; `--fast`; final time `20`.
- Host / GPU: shared filesystem; original host not recorded; no matching process on reachable hosts.
- tmux: none found.
- Command: original CLI not retained; manifest preserves the backend command/configuration.
- Log: no independent log found.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed1204_ftime20/compressed/compressing_model.pt` (574,458,794 B only).
- Submitted / started: manifest mtime `2026-08-27 03:23:45 BST`.
- Completed: not completed.
- Last checked: `2026-08-28 16:40:51 BST`.

#### Results

- No final model or metrics file was found. The temporary checkpoint indicates an interrupted or abandoned compression, but the cause is unknown.

### `poseidon_timestamp_20260824T072736Z`

- Status: `未开始`
- Outcome: `pending`
- Task: Poseidon SAFE-SVD timestamped run found as a manifest-only artifact.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; calibration size `7`; calibration seed `7`; `Poseidon-L`; `--fast`.
- Host / GPU: shared filesystem; original host not recorded; no matching process on reachable hosts.
- tmux: none found.
- Command: original CLI not retained; manifest preserves the backend command/configuration.
- Log: no independent log found.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/20260824T072736Z/` (manifest only).
- Submitted / started: manifest mtime `2026-08-24 08:27:36 BST`.
- Completed: not completed.
- Last checked: `2026-08-28 16:40:51 BST`.

#### Results

- No compression output, checkpoint, or metrics file was found; this is an empty or failed run of unknown cause.

## 正在运行

## 已完成

### `mpp_align_r08_seed42_rank_v3`

- Status: `已完成`.
- Task: Compare all48 AViT-L legacy ranks with the research selector using identical original weights/Input factors and each implementation's newly collected full Fisher factors.
- Hyperparameters: ratio0.8, legacy selector, calibration16/file, seed42, batch1, L1/balance10/trace inherited from the paired Fisher diagnostic. Requires its full comparison to pass first.
- Intended host / GPU: smew, GPU0, NVIDIA GeForce RTX3090Ti; launch only after Fisher exits and GPU is free.
- tmux: `mpp_align_r08_seed42_rank_v3`.
- Working directory: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0`.
- Exact command:

  ```bash
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /cs/research/vecg/EasyPDE/miniconda3/envs/aimet_env/bin/python -u /cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/validation/mpp_alignment_20261025/rank_alignment.py --safe-root /cs/research/vecg/EasyPDE/SAFE-SVD-v0 --reference-root /cs/research/vecg/EasyPDE/Scientific-Foundational-Model-compression/physic_model_comp/mpp --rank-ratio 0.8 --output /cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/validation/mpp_alignment_20261025/rank_comparison.json
  ```

- Log: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/validation/mpp_alignment_20261025/rank.log`.
- Output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/validation/mpp_alignment_20261025/rank_comparison.json`; wrapper status `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/validation/mpp_alignment_20261025/rank_status.json`.
- Planned: 2026-10-25T10:01:45.376625+00:00.
- Scope: actual rank allocation and parameter budget; no final compressed weights or evaluation produced.

- Started: `2026-10-25T10:10:22.951090+00:00`; process PID415160.
- Verified host/GPU: smew-l.cs.ucl.ac.uk GPU0 RTX3090Ti, process used2516MiB and GPU utilization100%; reference rank log advancing3/48.
- Fisher prerequisite passed exactly before launch; this run uses a new tmux/log/report, and preserves all existing model outputs.

- Outcome: `success`.
- Completed: `2026-10-25T10:12:35.798997+00:00`, exit0. No matching GPU process remained; GPU0 returned to151MiB,0% usage. Existing tmux session is retained at its shell.

#### Results

- Checked at completion on smew GPU0 RTX3090Ti, tmux `mpp_align_r08_seed42_rank_v3`. Source: `artifacts/validation/mpp_alignment_20261025/rank_comparison.json`, `rank_status.json`, `rank.log`. Exact command is retained above and in the report.
- All48 actual AViT-L legacy ranks at ratio0.8 match; rank difference count0, represented parameter difference0. Original checkpoint/Input factors match, and each side used its own newly collected exact-equal full Fisher factors.
- Reference/Safe represented linear weights: 161060864/161060864; target budget 161061273; represented total parameters 366789900/366789900.
- Reference rank selection 55.670s; SAFE 55.198s; full diagnostic 132.276s.
- Related validation: full cold Fisher all48 factor pairs exact (absolute/relative differences0), all176 inputs/counts/RNG equal; existing Input all48 pairs exact.24 MPP CPU tests passed against current server research source;48-layer full-model rank validation passed on the real GPU.
- Scope limit: this audit did not run new full sequential AViT-L compression or its final physical evaluation. Small-model sequential compression, resume and four-category evaluation match the research implementation in regression tests.

### `mpp_align_cali16_seed42_fisher_v3`

- Status: `已完成`.
- Task: Cold full-dataset MPP Fisher validation of SAFE research-v3 against the verified research collector, matching Fisher-entry RNG; no existing Input/Fisher/rank cache is read.
- Hyperparameters: calibration16 per file, seed42 (sampler and diagnostic entry RNG), batch1, L1, balance10, decay1, trace enabled, checkpointing from each YAML (both True), AMPFalse, n_steps16, 11 files / 176 batches per implementation. CPU threads1; FP32 reduction then FP64 whitening. No rank/compression/evaluation is included in this diagnostic.
- Host / GPU: smew-l.cs.ucl.ac.uk, GPU0, NVIDIA GeForce RTX3090Ti, confirmed idle (151MiB,0%,no compute processes) before launch.
- tmux: `mpp_align_cali16_seed42_fisher_v3`.
- Working directory: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0`.
- Exact command:

  ```bash
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /cs/research/vecg/EasyPDE/miniconda3/envs/aimet_env/bin/python -u /cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/validation/mpp_alignment_20261025/gpu_alignment.py --safe-root /cs/research/vecg/EasyPDE/SAFE-SVD-v0 --reference-root /cs/research/vecg/EasyPDE/Scientific-Foundational-Model-compression/physic_model_comp/mpp --checkpoint /cs/research/vecg/EasyPDE/SAFE-SVD-v0/models/MPP/MPP_AViT_L --scope full --implementation compare --calibration-num 16 --batch-size 1 --seed 42 --checkpointing config --expected-files 11 --output-dir /cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/validation/mpp_alignment_20261025/fisher_full
  ```

- Log: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/validation/mpp_alignment_20261025/fisher_full.log`.
- Outputs: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/validation/mpp_alignment_20261025/fisher_full/{reference,safe}/{report.json,fisher_factors.pt}`, `comparison.json`; wrapper status `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/validation/mpp_alignment_20261025/fisher_full_status.json`.
- Planned: 2026-10-25T09:52:19.190491+00:00.

- Started: `2026-10-25T09:52:27.609994+00:00`; wrapper PID411560, reference child PID411561. Verified GPU use22338MiB and live progress through batch17/176.

- Outcome: `success`.
- Completed: `2026-10-25T10:08:22.156705+00:00`, exit0. Both children completed all176 batches; no OOM. GPU returned to151MiB idle and no compute processes remained.

#### Results

- Checked at completion on smew GPU0 RTX3090Ti, tmux `mpp_align_cali16_seed42_fisher_v3`. Source: `artifacts/validation/mpp_alignment_20261025/fisher_full/comparison.json` and each side's `report.json`/`fisher_factors.pt`; wrapper status `fisher_full_status.json` and log `fisher_full.log`.
- All176 batch tensor hashes and ordering, all48 layers' gradient counts, CPU/CUDA RNG before/after are identical. Both configured checkpointing=True with24 checkpointed blocks, calibration16/file, batch1, n_steps16, AMPFalse.
- All48 (L,L-inverse) pairs are **exactly equal**, max absolute difference `0.0`, max relative Frobenius difference `0.0`. Comparison tolerance was atol1e-6/rtol1e-5, but observed differences are zero.
- Reference Fisher394.669s, peak allocated20.52894GiB; SAFE402.613s, peak allocated20.46253GiB. Peak allocator-reserved22.80859375GiB/23.04296875GiB respectively; this is distinct from allocated memory.
- Existing Input cache comparison separately found48/48 factor pairs exact, max absolute/relative difference0. Full original checkpoint file SHA256 also matches.
- This establishes full-dataset Fisher equality with matched entry RNG; it does not establish new final compressed-model metrics. The production calibration seed remains a sampler seed and does not silently reset stochastic DropPath RNG.

### `mpp_legacy_r0.8_cali16_seed42_bs1_v2_oom`

- Status: `已完成`.
- Outcome: `failed` (Fisher forward CUDA OOM; no compressed model produced).
- Task: Diagnose the user's SAFE MPP run against the verified research implementation.
- Host / GPU: `smew-l.cs.ucl.ac.uk`, GPU 0, NVIDIA GeForce RTX 3090 Ti (24564 MiB).
- tmux: existing user session `mpp:0.0`; idle shell at inspection; no matching process remains.
- Working directory: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0`.
- Exact command provided by user:

  ```bash
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python -m safe_svd.main --model mpp --ckpt_path models/MPP/MPP_AViT_L --yaml_config multiple_physics_pretraining/config/mpp_avit_L_config.yaml --config basic_config --data_root data/raw_data --rank_ratio 0.8 --rank_selector legacy --calibration_num 16 --calibration_seed 42 --batch_size 1 --alpha 0.7 --use_fisher y --gradient_loss l1 --balance_ratio 10 --decay_alpha 1 --use_trace_balance y --save_path ./artifacts/mpp/0.8/mpp --DEV cuda
  ```

- Hyperparameters: legacy ratio0.8, calibration16 per file, seed42, batch1, alpha0.7, Fisher L1, balance10, decay1, trace enabled, FP64 compression. Actual YAML had gradient_checkpointing=False, n_steps16, enable_amp=False.
- Started: exact process start not retained; run_config written `2026-10-25T08:42:21.406597+00:00`.
- Completed: exact failure time not retained; confirmed terminated at `2026-10-25T09:51:12.740548+00:00`.
- Log: existing tmux scrollback; sanitized traceback and run metadata in `artifacts/validation/mpp_alignment_20261025/inspection.json`.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/mpp/0.8/mpp/run_config.json`; no final model or compression checkpoint present at inspection.

#### Results

- Checked `2026-10-25T09:51:12.740548+00:00`. Input cache existed, Fisher/rank caches did not. Failure is Step 2 forward in AViT's non-checkpoint branch, requesting64MiB with63.88MiB free; allocated22.56GiB and process23.33GiB. Compression had not started.
- Research and SAFE checkpoints have identical full-file SHA256 `52b99bdc842f2b109984f85c85f05bc46c179f0dc9b9453350804a5bf9843ba5`. Model/data source files match. All11 per-file datasets' first sampled batch tensors/order match; effective config mismatch was gradient_checkpointing (SAFE False, reference True).
- `env VAR=value && python ...` does not pass the variable to Python; use `env VAR=value python ...`. This does not replace fixing the checkpoint configuration.
- New local/remote alignment work uses version3 cache paths and separate validation outputs; the failed run's existing artifacts are preserved. The original research repository is read-only.

### `vicon_research_v3_eac2915da5b9_r0.8_bs64_seed42`

- Status: `已完成`.
- Outcome: `success` (compression and evaluation artifacts produced; research equivalence is **not** established).
- Task: Diagnose slow VICON compression after research-alignment corrections.
- Hyperparameters: VICON; legacy ranks; ratio `0.8`; calibration `2048` per
  category; batch `64`; seed label `42`; fixed calibration; alpha `0.7`;
  Fisher `l1`; balance `10`; trace enabled; FP64 compression; alignment v3.
- Host / GPU: `ruddy-l.cs.ucl.ac.uk`, GPU 0, NVIDIA GeForce RTX 3090 Ti.
- tmux: `vicon_safe`, pane `vicon_safe:0.0`.
- Working directory: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0`.
- Exact parent launch command (PID `89975`):

  ```bash
  python -m safe_svd.main --model vicon --ckpt_path models/VICON/model.pth --data_root data/processed_data/2D/CFD/converted/ --rank_ratio 0.8 --rank_selector legacy --calibration_num 2048 --batch_size 64 --alpha 0.7 --use_fisher y --gradient_loss l1 --balance_ratio 10 --use_trace_align y --save_path ./artifacts/vicon/0.8/vicon --DEV cuda --overwrite_cache
  ```

- Started: parent `2026-10-23 07:43:55 BST`; compression child PID `94754`
  started `08:31:55 BST`. The child uses `aimet_env/bin/python`,
  `--pipeline_stage compression`, and the resolved v3 cache paths recorded in
  the diagnostic snapshot and `run_config.json`.
- Log: tmux pane, stdout/stderr `/dev/pts/1`; no independent compression log
  was identified. Read-only diagnostic snapshot:
  `artifacts/validation/vicon_speed_20261023.json`.
- Output directory: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/vicon/0.8/vicon`.
  Planned checkpoint: `compressing_model.pt`, every 6 enumerated Linear layers.
- Caches: input/Fisher/rank use the `vicon_research_v3_eac2915da5b9_cali2048`
  prefix; exact paths and command arguments are retained in the snapshot.
- Previous check: `2026-10-23 09:01:47 BST` / `16:01:47 Asia/Shanghai`.
- Last checked: `2026-10-25T15:18:55+08:00`; ruddy pane is at an idle shell and no matching compression/evaluation process remains.
- Completion evidence: final compression model mtime `2026-10-23 10:30:06 BST` / `17:30:06 Asia/Shanghai`; metrics mtime `2026-10-23 11:22:00 BST` / `18:22:00 Asia/Shanghai`. These are artifact-write times, not independently recorded process exit times.
- tmux final progress: `62/62 [1:57:44, 113.94s/layer]`, followed by final model and metrics success messages.

#### Historical diagnostics (2026-10-23, while running)

- User-reported progress: `1/62 [23:02, 1382.43s/layer]`, postfix `q_proj`.
  Verified at 08:59:49 BST: `4/62 [26:12]`; at 09:01:47 BST:
  `5/62 [27:48]`, postfix `transformer.layers.0.self_attn.out_proj`.
  The first displayed per-layer estimate is not a steady-state measurement.
- GPU utilization was 100% at both checks; GPU memory was 6368 then 7970 MiB.
  Only compression PID 94754 was listed as a GPU compute process. Its RSS was
  about 55.6 GiB and process swap was zero. Host available memory was about
  64.5 GiB; vmstat showed no ongoing swap-in/out or I/O wait in sampled seconds.
- SAFE's first actual layer includes all CPU raw-batch prefetch. The original
  research implementation prefetches before its layer progress bar. Both
  retain 96 batches (3 categories x 32), approximately 52.5 GiB of raw tensors
  with the configured default dimensions. Prefetch and first-layer compute
  were not separately timed, so their individual contributions are unknown.
- Both implementations use paired teacher/student capture and three FP64
  Gram products for each compressed layer. The current compression loop,
  shared accumulator and VICON collector match local files and have no
  uncommitted changes relative to HEAD. No large new Step-4 computation was
  identified in the recent alignment changes. Batch 64 versus the older 128
  default increases iteration/launch overhead without increasing total samples.
- Current ranks: `pre_proj=211`, `post_proj=null`, first `q_proj=160`.
  The previous output at the same path also compressed pre_proj (rank 209),
  so newly enabling pre_proj compression does not explain this run's timing.
- Existing `compressed_model.pth`, `rank_dict.json`, and `metrics.json` in this
  output directory predate the current run. They are not completion evidence
  or evaluation results for the running v3 experiment. No new checkpoint was
  present at the last snapshot, before the first 6-layer save.
- Input alignment caveat: the explicit `--data_root` still selects the old
  `data/processed_data/2D/CFD/converted` tree, bypassing the separately prepared
  `data/VICON` split/content mapping. The v3 file-order manifest alone does not
  correct the old CFD split/content mapping. This is a reproduction issue,
  not an established cause of slowness; no running configuration was changed.

#### Results

- Checked: `2026-10-25T15:18:55+08:00`, read-only on `ruddy` (run/pane) and `mandarin` (shared artifacts and CPU diagnostics). Run assignment: ruddy GPU 0, RTX 3090 Ti, tmux `vicon_safe:0.0`. No new GPU run was launched.
- Sources: `artifacts/vicon/0.8/vicon/{run_config.json,rank_dict.json,compressed_model.pth,metrics.json}` and tmux scrollback. Full comparison evidence: `artifacts/validation/vicon_audit_20261025/audit.json`.
- Exact successful evaluation command (cwd repository root; protocol research, test seed 84, server default batch 64, no batch cap, no config overrides):

  ```bash
  python evaluation/vicon/evaluate.py --vicon-root ./VICON --data-root data/processed_data/2D/CFD/converted --checkpoint models/VICON/model.pth --model artifacts/vicon/0.8/vicon/compressed_model.pth --output artifacts/vicon/0.8/vicon/metrics.json
  ```

- Compressed total parameters: **70,310,912**; linear parameters: **70,118,400**. Origin: 87,823,104 / 87,630,592. Loss and physics batch counts: COMPRESSIBLE2D 63, EULER2D 7, NS2D 10.

| Dataset | Origin loss | Compressed loss | Origin Sobolev | Compressed Sobolev | Compressed relative loss | Compressed relative Sobolev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COMPRESSIBLE2D | 0.000623345317 | 0.000638427246 | 1.652218400 | 1.613612085 | 0.000376542468 | 0.079213999 |
| EULER2D | 0.023612800453 | 0.022755171571 | 13.725354545 | 13.507794960 | 0.004895116918 | 0.454087620 |
| NS2D | 0.002886012942 | 0.002695847163 | 597.719259679 | 590.429572768 | 0.001006982739 | 0.775848677 |

- Compressed NS2D continuity 4.639865589 (relative 0.601918330), vorticity error 0.978040880 (relative 0.211678682). Full precision metrics and increases are retained in the audit JSON.
- Interpretation: **not an equivalent reproduction**. Both compression and evaluation used the old converted root. Across the two CFD categories, old-root test sampled fingerprints match 0/16 reference files; mapped `data/VICON` matches 16/16. Current CFD test filename enumeration order matches, but evaluation has no test manifest. Fingerprints sample 48 scalar values/file, not full-file content.
- Original checkpoint equality: all 129 state tensors are exactly equal. Four selected layers' reconstructed input covariance relative Frobenius differences are 2.09%-4.12%; Fisher differences are 14.99%-40.66%. The mismatch therefore exists before rank selection.
- SAFE versus the named reference checkpoint/rank: **31/62 ranks differ**, and every checkpoint shape agrees with its own rank JSON. Net parameter delta -512 is exactly the sum of rank changes (pre_proj -16,896; attention +71,680; linear1 -21,504; linear2 -33,792), not a counting convention.
- Reference CSV reuses Origin Euler Sobolev **30.581628** without checkpoint/data/settings provenance and can skip its two sampling passes. SAFE always recomputes origin. Consequently the CSV improvement percentages and this JSON's percentages use different baselines/pass histories.
- Controlled CPU validation: 19 compression-alignment tests passed; 18 metric formula cases max relative difference 2.5073e-7; four attention/load comparisons exactly match; eight sampling assertions passed. These do not establish full real-data experiment equality.
- Remaining limits: original cache-hit compression does not fix global NumPy state; matching seed label alone is insufficient. Historical CSV/RNG provenance is incomplete. No full GPU reproduction rerun was performed. Product and original research code were left unchanged; only local/remote audit records and this ledger were updated.

### `vicon_legacy_fixed_r0.8`

- Status: `已完成`
- Outcome: `success` (compression and smoke evaluation); evaluation comparison is not sample-aligned.
- Task: VICON SAFE-SVD compression compared with the research `physic_model_comp/vicon/MySVD` result.
- Hyperparameters: rank ratio `0.8`; rank selector `legacy`; calibration size `2048`; fixed-prefix calibration (`random_calibration=n`, train seed `42`); alpha `0.7`; Fisher `l1`; balance ratio `10`; trace alignment enabled; spectral whitening with absolute epsilon `1e-6`; device `cuda`.
- Host / GPU: original launch host and GPU were not recorded; artifacts are on the shared filesystem and were inspected read-only through `mandarin`.
- tmux: not recorded.
- Command:

  ```bash
  python -m safe_svd.main \
    --model vicon \
    --ckpt_path models/VICON/model.pth \
    --data_root data/processed_data/2D/CFD/converted/ \
    --rank_ratio 0.8 \
    --rank_selector legacy \
    --calibration_num 2048 \
    --batch_size 128 \
    --alpha 0.7 \
    --use_fisher y \
    --gradient_loss l1 \
    --balance_ratio 10 \
    --use_trace_align y \
    --save_path ./artifacts/vicon_r08 \
    --DEV cuda
  ```

- Log: no independent log was found during this inspection.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/vicon_r08/compressed_model.pth`; ranks at the adjacent `rank_dict.json`; run metadata at `run_config.json`; evaluation at `metrics_smoke.json`.
- Submitted / started: not recorded.
- Completed: compressed model mtime `2026-10-16 12:02:12`; smoke metrics mtime `2026-10-22 03:20:13` (remote filesystem time).
- Last checked: `2026-10-22`.

#### Results

- Source: `artifacts/vicon_r08/{run_config.json,rank_dict.json,metrics_smoke.json}` and the research MySVD checkpoint/rank/CSV, checked `2026-10-22`.
- Compression retained `70,311,680 / 87,823,104` total parameters and `70,119,168 / 87,630,592` Linear parameters (about 80%, i.e. about 20% reduction).
- The source checkpoint is byte-identical to the research MySVD checkpoint: both SHA-256 values are `d0af2ed2f0e3eb935ed62dfad350f218b6fb927098ecd6db17d3dd2877eab91e`.
- The selected ranks are not byte-identical: SAFE-SVD rank SHA-256 is `e93fe3ed5d44e00a6ba15186308ad9f363d2ef3462b9e58f88834e8707beef8a`, while research `cali2048_l1_trace/rank_0.8.json` is `3b05e359d2a1e95c7ae5d7725667b37d488d67af47e469f7d12027c0b8d16b91`.
- The smoke evaluator did not feed origin and compressed models identical test time pairs. For example, the inferred COMPRESSIBLE2D label energy is about `1.891` for origin (`loss/relative_loss`) but `1.310` for compressed. Therefore the reported `-91.28%` loss change and other origin/compressed percentages are not trustworthy as paired-model comparisons.
- The research CSV also persists and reuses an old origin row while its zero-worker iterable test datasets carry state across passes, so it is not a stable gold reference for cross-run comparison. A deterministic, cached test-batch replay is required before attributing remaining differences to compression.

##### Audit update — 2026-10-22 16:35 (Asia/Shanghai)

- Scope: read-only comparison of the existing run and current implementations;
  no compression or evaluation job was launched. The status remains `已完成`,
  outcome `success` for artifact generation, with experiment equivalence **not
  established**. The command and paths above remain the recorded run identity.
- Inspection host: `mandarin-l.cs.ucl.ac.uk`, GPU 0 NVIDIA GeForce RTX 3090 Ti
  (153 MiB allocated, 0% utilization at inspection; no GPU computation used by
  this audit). There was no tmux server and no matching SAFE-SVD process on this
  host. This does not recover the original launch host/GPU/tmux, which remain
  unknown. No independent compression log or exact smoke-evaluation command
  was available; the smoke JSON omits protocol, batch size and batch limit.
- `poseidon` resolves to
  `/cs/research/vecg/EasyPDE/miniconda3/envs/aimet_env/bin/python`;
  PyTorch `2.5.1+cu121`, CUDA `12.1`, NumPy `1.26.4`.
- Local and remote SAFE HEAD are both
  `79fd5d12c797eef52387659ca27d4d50d4d22d0e`. Hashes of `safe_svd/main.py`,
  `safe_svd/compress.py`, `safesvd/adapters/vicon_runtime.py`, and the untracked
  `evaluation/vicon/evaluate.py` match between local and remote. Existing
  unrelated working-tree changes were preserved.
- The server research and SAFE copies of `VICON/src/{models,dataset,dataset_utils,
  trainer,crop_utils,utils,model_utils}.py` and model/loss/rollout default configs
  are byte-identical. The input checkpoint SHA-256 above was rechecked and also
  matches. The local research `vicon/VICON` checkout is a different, incompatible
  snapshot; its uncropped architecture and sorted-file loader must not be used
  to diagnose the server experiment. The server research MySVD and evaluation
  entry scripts do match the local research copies.

**Confirmed differences in the existing artifacts and effective inputs**

- `run_config.json` identifies `vicon_research_v1` caches, with
  `profile_cache=true`, `fisher_cache=false`, `rank_cache=false` before the run.
  Both saved input/Fisher caches lack `vicon_rng_state`. The current code uses
  `vicon_research_v2` and rejects those old payloads rather than silently treating
  them as equivalent. Thus the existing `vicon_r08` output is not evidence of
  current v2 end-to-end behavior. The old cache-hit path skipped profiling's
  seeding and RNG advancement before computing Fisher.
- Rank dictionaries have 62 entries each, with **32 different ranks**, not merely
  different JSON formatting. Examples (research -> SAFE): `pre_proj` 217 -> 212,
  `transformer.layers.2.self_attn.k_proj` 208 -> 231,
  `transformer.layers.9.linear2` 468 -> 442.
- Both implementations enumerate paths without sorting, but their separate
  data directories enumerate differently. Both CFD training directories have
  64 files and the same filename set: research starts `0.h5,1.h5,10.h5,11.h5`,
  SAFE starts `0.h5,1.h5,2.h5,3.h5`. NS2D has the same filename sets (78 train,
  19 test), but both train and test enumeration orders differ. The CFD test
  directories have eight files each in matching order. These are distinct
  files rather than identical inodes; full data-content equality was not
  established. Same seed plus different initial order does not select the same
  files, trajectories or time pairs. Sources: the configured research folders
  under `physic_model_comp/vicon/VICON/` and SAFE
  `data/processed_data/2D/CFD/converted/`.
- CPU comparisons of existing caches use induced metrics `R @ R.T` and `L.T @ L`
  in float64, avoiding arbitrary eigenvector signs/bases. Relative difference
  means `norm(SAFE-reference) / norm(reference)` in Frobenius norm. Values below
  are observed differences, not an accepted numerical tolerance:

  | Layer | Input max absolute difference | Input relative difference | Fisher max absolute difference | Fisher relative difference |
  | --- | ---: | ---: | ---: | ---: |
  | `pre_proj` | 0.0634976 | 5.93783% | 3.05840e-15 | 67.8299% |
  | `post_proj` | 3.42233 | 1.80057% | 1.48388e-14 | 12.2939% |
  | `transformer.layers.0.self_attn.q_proj` | 10.3373 | 6.14236% | 2.26749e-15 | 77.9681% |
  | `transformer.layers.2.self_attn.k_proj` | 3.94737 | 3.41770% | 3.71949e-14 | 108.220% |

  Fisher matrices have very small absolute scale, so their absolute differences
  alone are misleading. For the last row, reference/SAFE traces are respectively
  `2.19358216e-13` / `4.52297611e-13`. Statistics already differ before rank
  allocation; this is not explained by JSON serialization or SVD sign choices.
  Cache sources are the exact paths in `run_config.json`, compared with research
  `MySVD/input_cache/cali2048_trace/profiling_mat.pt` and
  `MySVD/fisher_cache/cali2048_l1_trace/Ldict.pt`.

**Current pipeline differences that remain after the v2 changes**

- Compression sampling: research `MySVD/safe_svd.py` launches `rank.py` and
  `mysvd.py` as separate processes. The latter builds fresh datasets and, with
  both statistics caches present, starts comb capture from a fresh stream.
  SAFE runs input -> Fisher -> comb on the same loaders and restores stage-end
  RNG on cache hits (`safe_svd/main.py:214,258`). With workers=0, cold SAFE comb
  normally sees a third traversal, whereas the research compression process
  begins a first traversal. Splitting SAFE into its two CLI stages does not
  eliminate this difference because RNG is restored from caches.
- Historical reproducibility: the shared train loader initializes RNG from
  global `np.random.randint`, not the supplied dataset base seed
  (`VICON/src/dataset.py:108`). Research `mysvd.py` does not seed the global RNG
  when both caches hit. A seed label alone cannot reconstruct that historical
  compression input sequence. A CPU reproduction using the actual datasetBase
  class and tiny synthetic file contents confirms different file/trajectory
  selections across the separate-process versus continuous-stream boundary.
- Evaluation sampling: current `research` mode correctly advances through
  origin-loss -> origin-physics -> compressed-loss -> compressed-physics.
  Research `evaluation/eval_comp.py:584,628` can skip origin and already-written
  model rows based on CSV contents, and can evaluate several compressed models
  on one shared stream. SAFE always evaluates origin and one compressed model.
  Their compressed traversals align only when the original also evaluates
  origin first and this is the first compressed model, with matching data order
  and batch settings. A cached original row or earlier evaluated models changes
  that position. `--protocol paired` is a separate diagnostic and must not
  silently replace reproduction of the original research sequence.
- Cache/resume provenance remains incomplete: default cache identities omit
  YAML contents, data contents/order and code hashes; explicit caches mainly
  validate category/loader settings and restore saved `dp` paths. Resume checks
  ranks/settings/version but not checkpoint/data/statistics identity. These
  are potential reuse risks, not proven triggers for this completed run.
- Defaults must be compared by entry point: SAFE CLI and research `mysvd.py`
  default to ratio .8/fixed selection/seed label 2026, but the research wrapper
  defaults to ratio .6/random selection/seed42, and bare `rank.py` defaults to
  trace alignment off. With `use_fisher=n`, research still uses Fisher for rank
  selection while SAFE disables it for both stages. These nonmatching settings
  do not describe the recorded .8/Fisher-on/trace-on run above.

**Metrics and validation limits**

The reference row is `MySVD(cali2048_l1_br10.0_trace)` in research
`results/vicon_eval_results_0.8.csv`. SAFE values are from `metrics_smoke.json`.
They are recorded here as evidence, **not** as a matched-sample quality ranking:

| Metric | Research compressed | Existing SAFE compressed |
| --- | ---: | ---: |
| COMPRESSIBLE2D loss | 0.000670 | 0.000148919971 |
| EULER2D loss | 0.023385 | 0.024405857548 |
| NS2D loss | 0.002861 | 0.003099184250 |
| COMPRESSIBLE2D Sobolev | 1.798681 | 4.634216540 |
| EULER2D Sobolev | 22.018674 | 15.691371635 |
| NS2D Sobolev | 602.270929 | 585.248172740 |

- Fixed-input CPU comparisons against extracted original functions found exact
  input/Fisher induced metrics in two-layer examples with trace on/off, exact
  legacy ranks at five budget ratios, and comb operators/predictions within
  `atol=2e-6, rtol=2e-5` for alpha 0/.7/1 and trace on/off. Tiny cropped-attention
  and normalization comparisons had maximum absolute difference zero. Seven
  targeted existing tests passed. This confirms small-case math, not actual
  full-model/GPU/data equivalence.
- Original CSV rounds metrics to six decimal places (up to 5e-7 absolute
  rounding), while JSON retains precision. Early Python-float conversion in
  the port can cause small physics differences; a separate NumPy 2.4.3 toy
  comparison had relative differences below 8.5e-8. The server uses NumPy
  1.26.4, so this toy result is not a measured error bound for the server run.
- Next controlled comparison must preserve the original files/checkpoints,
  use fresh separate cache/output paths, record actual file/batch/RNG order
  separately for rank collection, compression and each research evaluation
  pass, then compare R/L metrics, ranks, effective weights and final metrics.
  Without that comparison, the contribution of each identified mismatch to
  the historical metric differences remains unquantified.

### `poseidon_safe_seed1_r0.2`

- Status: `已完成`
- Outcome: `success` (compression); physical evaluation not found.
- Task: Poseidon SAFE-SVD compression with calibration seed `1`.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; rank selector `safe`; calibration size `64`; calibration seed `1`; `Poseidon-L`; `--fast`; final time `20`; alpha `0.7`; device `cuda`.
- Host / GPU: `brent`, GPU 0, NVIDIA GeForce RTX 3090 Ti.
- tmux: `poseidon_safe_seed1_r0p2` (created `2026-08-29 08:13:34 BST`; no session at the last check because the process exited).
- Command:

  ```bash
  python -m safesvd.cli compress \
    --model poseidon \
    --checkpoint models/poseidon/fluids.incompressible.PiecewiseConstants \
    --data-root poseidon/camlab-ethz/down_streams \
    --calibration-seed 1 \
    --rank-ratio 0.2 \
    --rank-selector safe \
    --calibration-size 64 \
    --dataset fluids.incompressible.PiecewiseConstants \
    --final-time 20 \
    -- \
    --model_name Poseidon-L \
    --fast
  ```

- Log: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/logs/poseidon_safe_seed1_r0p2.log` (46,367 B; mtime `2026-08-29 14:38:35 BST`).
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed1_ftime20/compressed/compressed_model.pt` (532,493,038 B).
- Submitted / started: process start `2026-08-29 08:13:40 BST` (PID `217780`, backend PID `217781`).
- Completed: final model mtime `2026-08-29 14:38:35 BST`; log reports `815/815` and the saved model path.
- Last checked: `2026-08-31 02:34:27 BST`.

#### Results

- Source: final checkpoint and `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/logs/poseidon_safe_seed1_r0p2.log`, checked `2026-08-31 02:34:27 BST`.
- No `metrics.json` was found in the run root. Compression is complete, but no physical-evaluation metrics are available for this seed.

### `poseidon_safe_seed1204_r0.2`

- Status: `已完成`
- Outcome: `success` (compression); physical evaluation not found.
- Task: Poseidon SAFE-SVD compression with calibration seed `1204`.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; rank selector `safe`; calibration size `64`; calibration seed `1204`; `Poseidon-L`; `--fast`; final time `20`; alpha `0.7`; device `cuda`.
- Host / GPU: `ruddy`, GPU 0, NVIDIA GeForce RTX 3090 Ti; 21,801 MiB used and 0% utilization at the last check, consistent with no active run.
- tmux: `poseidon_safe_seed1204_r0p2` (created `2026-08-29 08:13:29 BST`; session remains at an idle shell after completion).
- Command:

  ```bash
  python -m safesvd.cli compress \
    --model poseidon \
    --checkpoint models/poseidon/fluids.incompressible.PiecewiseConstants \
    --data-root poseidon/camlab-ethz/down_streams \
    --calibration-seed 1204 \
    --rank-ratio 0.2 \
    --rank-selector safe \
    --calibration-size 64 \
    --dataset fluids.incompressible.PiecewiseConstants \
    --final-time 20 \
    -- \
    --model_name Poseidon-L \
    --fast
  ```

- Log: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/logs/poseidon_safe_seed1204_r0p2.log` (52,350 B; mtime `2026-08-29 15:52:56 BST`).
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed1204_ftime20/compressed/compressed_model.pt` (532,533,206 B).
- Submitted / started: process start `2026-08-29 08:13:34 BST` (PID `2369733`, backend PID `2369734`).
- Completed: final model mtime `2026-08-29 15:52:55 BST`; log reports `815/815` and the saved model path.
- Last checked: `2026-08-31 02:36:15 BST`.

#### Results

- Source: final checkpoint and `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/logs/poseidon_safe_seed1204_r0p2.log`, checked `2026-08-31 02:36:15 BST`.
- No `metrics.json` was found in the run root. Compression is complete, but no physical-evaluation metrics are available for this seed.

### `poseidon_safe_seed42_r0.2`

- Status: `已完成`
- Outcome: `success`
- Task: Poseidon SAFE-SVD compression followed by physical evaluation.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; rank selector `safe`; calibration size `64`; calibration seed `42`; `Poseidon-L`; `--fast`; final time `20`; alpha `0.7`; device `cuda`.
- Host / GPU: `ruddy`, GPU 0, NVIDIA GeForce RTX 3090 Ti; compression session is idle at the shell after completion.
- tmux: `new_rank_seed42` (created `2026-08-28 07:49:49 BST`).
- Command:

  ```bash
  python -m safesvd.cli compress \
    --model poseidon \
    --checkpoint models/poseidon/fluids.incompressible.PiecewiseConstants \
    --data-root poseidon/camlab-ethz/down_streams \
    --calibration-seed 42 \
    --rank-ratio 0.2 \
    --rank-selector safe \
    --calibration-size 64 \
    --dataset fluids.incompressible.PiecewiseConstants \
    --final-time 20 \
    -- \
    --model_name Poseidon-L \
    --fast
  ```

- Log: compression output was captured in tmux; no independent compression `.log` was found. Evaluation output was observed in the shared `brent` `svd` tmux session.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed42_ftime20/compressed/compressed_model.pt` (532,531,526 B); metrics at the same run root.
- Submitted / started: process start `2026-08-28 07:50:23 BST` (PID `104165`, backend PID `104166`).
- Completed: compressed model mtime `2026-08-28 18:49:37 BST`; evaluation metrics mtime `2026-08-29 07:41:57 BST`.
- Last checked: `2026-08-29 08:17:02 BST`.

#### Results

- Source: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed42_ftime20/metrics.json`, checked `2026-08-29 08:17:02 BST`.
- Compressed metrics: base loss `0.00240649`, continuity `0.0226094`, momentum `0.949849`, h1 error `0.0747666`, Sobolev s012 `5.43859`, vorticity error `0.0754400`.
- Interpretation: relative changes are base loss `+3.29%`, continuity `+0.65%`, momentum `-0.06%`, h1 `+1.53%`, Sobolev s012 `+2.38%`, and vorticity `+1.39%`; the evaluated compressed model remains close to the origin model.

### `poseidon_safe_seed7_r0.2`

- Status: `已完成`
- Outcome: `success`
- Task: Poseidon SAFE-SVD compression followed by physical evaluation.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; rank selector `safe`; calibration size `64`; calibration seed `7`; `Poseidon-L`; `--fast`; final time `20`; alpha `0.7`; device `cuda`.
- Host / GPU: `smew`, GPU 0, NVIDIA GeForce RTX 3090 Ti; compression session is idle at the shell after completion.
- tmux: `new_rank_seed7` (created `2026-08-28 14:27:52 BST`).
- Command:

  ```bash
  python -m safesvd.cli compress \
    --model poseidon \
    --checkpoint models/poseidon/fluids.incompressible.PiecewiseConstants \
    --data-root poseidon/camlab-ethz/down_streams \
    --calibration-seed 7 \
    --rank-ratio 0.2 \
    --rank-selector safe \
    --calibration-size 64 \
    --dataset fluids.incompressible.PiecewiseConstants \
    --final-time 20 \
    -- \
    --model_name Poseidon-L \
    --fast
  ```

- Log: compression output was captured in tmux; no independent compression `.log` was found. Evaluation output was observed in the shared `brent` `svd` tmux session.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed7_ftime20/compressed/compressed_model.pt` (532,533,726 B); metrics at the same run root.
- Submitted / started: process start `2026-08-28 14:28:24 BST` (PID `166151`, backend PID `166152`).
- Completed: compressed model mtime `2026-08-28 22:10:24 BST`; evaluation metrics mtime `2026-08-29 07:58:12 BST`.
- Last checked: `2026-08-29 08:17:02 BST`.

#### Results

- Source: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed7_ftime20/metrics.json`, checked `2026-08-29 08:17:02 BST`.
- Compressed metrics: base loss `0.00241412`, continuity `0.0225925`, momentum `0.949806`, h1 error `0.0749315`, Sobolev s012 `5.44372`, vorticity error `0.0756085`.
- Interpretation: relative changes are base loss `+3.62%`, continuity `+0.57%`, momentum `-0.07%`, h1 `+1.75%`, Sobolev s012 `+2.48%`, and vorticity `+1.62%`; the evaluated compressed model remains close to the origin model.

### `poseidon_safe_default_r0.2`

- Status: `已完成`
- Outcome: `success`
- Task: Poseidon SAFE-SVD compression followed by Poseidon physical evaluation.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; rank selector `safe`; calibration size `64`; calibration seed `-1` (fixed prefix); `Poseidon-L`; `--fast`; final time `20`.
- Host / GPU: `brent`, GPU 0, NVIDIA GeForce RTX 3090 Ti; tmux session is idle at the shell after completion.
- tmux: `new_rank` (created `2026-08-28 02:31:32 BST`).
- Command:

  ```bash
  python -m safesvd.cli compress \
    --model poseidon \
    --checkpoint models/poseidon/fluids.incompressible.PiecewiseConstants \
    --data-root poseidon/camlab-ethz/down_streams \
    --rank-ratio 0.2 \
    --rank-selector safe \
    --calibration-size 64 \
    --dataset fluids.incompressible.PiecewiseConstants \
    --final-time 20 \
    -- --model_name Poseidon-L --fast

  python evaluation/poseidon/evaluate.py \
    --poseidon-root poseidon \
    --data-root poseidon/camlab-ethz/down_streams \
    --checkpoint models/poseidon/fluids.incompressible.PiecewiseConstants \
    --model /cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_default_ftime20/compressed/compressed_model.pt
  ```

- Log: tmux pane; no independent `.log` found.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_default_ftime20/compressed/compressed_model.pt` (532,526,422 B); metrics in the same run's `metrics.json`.
- Submitted / started: tmux created `2026-08-28 02:31:32 BST`.
- Completed: final model mtime `2026-08-28 09:45:53 BST`; evaluation completed in the tmux pane afterward.
- Last checked: `2026-08-28 16:40:51 BST`.

#### Results

- Source: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_default_ftime20/metrics.json`, checked `2026-08-28 16:40:51 BST`.
- Compressed metrics: base loss `0.00238230`, continuity `0.0225261`, momentum `0.949995`, h1 error `0.0746340`, Sobolev s012 `5.37074`, vorticity error `0.0753724`.
- Interpretation: the physical metrics remain close to the origin model, while the recorded relative increases are largest for base loss (`+225.3%`), h1 (`+134.8%`), Sobolev s012 (`+109.8%`), and vorticity (`+130.2%`).

### `poseidon_legacy_default_r0.2`

- Status: `已完成`
- Outcome: `success`
- Task: Poseidon SAFE-SVD legacy compression artifact and physical evaluation.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; calibration size `64`; calibration seed `-1`; manifest `rank_selector=null`; `Poseidon-L`; `--fast`; final time `20`.
- Host / GPU: shared filesystem; original host not recorded; no live process found.
- tmux: no active session associated with this artifact.
- Command: original CLI not retained; manifest `backend_command_shell` is the authoritative recorded command.
- Log: no independent log found.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/0.2/fluids.incompressible.PiecewiseConstants/traj64_default_ftime20/compressed/compressed_model_origin.pt` (532,564,333 B).
- Submitted / started: manifest mtime `2026-08-26 03:48:18 BST`.
- Completed: final model mtime `2026-08-26 08:11:52 BST`.
- Last checked: `2026-08-28 16:40:51 BST`.

#### Results

- Source: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/0.2/fluids.incompressible.PiecewiseConstants/traj64_default_ftime20/metrics.json`, checked `2026-08-28 16:40:51 BST`.
- Compressed metrics: base loss `0.00239431`, continuity `0.0226081`, momentum `0.950005`, h1 error `0.0747979`, Sobolev s012 `5.43808`, vorticity error `0.0754854`.
- Interpretation: evaluation is present; this legacy selector has larger recorded increases in base loss (`+276.8%`) and Sobolev s012 (`+237.0%`) than the completed safe default run.

### `poseidon_legacy_seed42_r0.2`

- Status: `已完成`
- Outcome: `success`
- Task: Poseidon SAFE-SVD legacy compression and evaluation.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; calibration size `64`; calibration seed `42`; manifest `rank_selector=null`; `Poseidon-L`; `--fast`; final time `20`.
- Host / GPU: shared filesystem; original host not recorded; no live process found.
- tmux: no active session associated with this artifact.
- Command: original CLI not retained; manifest `backend_command_shell` is the authoritative recorded command.
- Log: no independent log found.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed42_ftime20/compressed/compressed_model_origin.pt` (532,567,092 B).
- Submitted / started: manifest mtime `2026-08-26 11:19:39 BST`.
- Completed: final model mtime `2026-08-26 20:17:26 BST`.
- Last checked: `2026-08-28 16:40:51 BST`.

#### Results

- Source: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed42_ftime20/metrics.json`, checked `2026-08-28 16:40:51 BST`.
- Compressed metrics: base loss `0.00243458`, continuity `0.0229036`, momentum `0.949885`, h1 error `0.0752439`, Sobolev s012 `5.61653`, vorticity error `0.0757916`.
- Interpretation: evaluation is present; the recorded Sobolev s012 increase is `+573.7%`, substantially above the safe default run.

### `poseidon_legacy_seed7_r0.2`

- Status: `已完成`
- Outcome: `success`
- Task: Poseidon SAFE-SVD legacy compression and evaluation.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; calibration size `64`; calibration seed `7`; manifest `rank_selector=null`; `Poseidon-L`; `--fast`; final time `20`.
- Host / GPU: shared filesystem; original host not recorded; no live process found.
- tmux: no active session associated with this artifact.
- Command: original CLI not retained; manifest `backend_command_shell` is the authoritative recorded command.
- Log: no independent log found.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed7_ftime20/compressed/compressed_model_origin.pt` (532,567,939 B).
- Submitted / started: manifest mtime `2026-08-26 11:17:39 BST`.
- Completed: final model mtime `2026-08-26 21:57:31 BST`.
- Last checked: `2026-08-28 16:40:51 BST`.

#### Results

- Source: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed7_ftime20/metrics.json`, checked `2026-08-28 16:40:51 BST`.
- Compressed metrics: base loss `0.00243951`, continuity `0.0228620`, momentum `0.949843`, h1 error `0.0753912`, Sobolev s012 `5.61041`, vorticity error `0.0759503`.
- Interpretation: evaluation is present; the Sobolev s012 and vorticity increases are `+562.1%` and `+207.9%`, respectively.

### `poseidon_legacy_seed1_r0.2`

- Status: `已完成`
- Outcome: `success`
- Task: Poseidon SAFE-SVD legacy compression; evaluation status unknown.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; calibration size `64`; calibration seed `1`; manifest `rank_selector=null`; `Poseidon-L`; `--fast`; final time `20`.
- Host / GPU: shared filesystem; original host not recorded; no live process found.
- tmux: no active session associated with this artifact.
- Command: original CLI not retained; manifest `backend_command_shell` is the authoritative recorded command.
- Log: no independent log found.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed1_ftime20/compressed/compressed_model_origin.pt` (532,557,568 B).
- Submitted / started: manifest mtime `2026-08-27 03:25:00 BST`.
- Completed: final model mtime `2026-08-27 11:37:38 BST`.
- Last checked: `2026-08-28 16:40:51 BST`.

#### Results

- Source: final checkpoint above, checked `2026-08-28 16:40:51 BST`.
- No `metrics.json` was found, so compression completion is confirmed but physical evaluation cannot be confirmed and no metric interpretation is available.

### `poseidon_evaluation_20260819T011617Z`

- Status: `已完成`
- Outcome: `success`
- Task: Poseidon SAFE-SVD compression with a subsequent physical evaluation.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; calibration size `64`; calibration seed `42`; `Poseidon-L`; `--fast`; final time `20`.
- Host / GPU: shared filesystem; original host not recorded; no live process found.
- tmux: no active session associated with this artifact.
- Command: original CLI not retained; manifest `backend_command_shell` and evaluation script metadata are the authoritative recorded commands.
- Log: no independent log found.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/20260819T011617Z/compressed/compressed_model_origin.pt` (532,535,629 B).
- Submitted / started: manifest mtime `2026-08-19 02:16:17 BST`.
- Completed: final model mtime `2026-08-19 11:32:21 BST`; evaluation metrics mtime `2026-08-24 06:06:23 BST`.
- Last checked: `2026-08-28 16:40:51 BST`.

#### Results

- Source: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/20260819T011617Z/evaluation/poseidon/metrics.json`, checked `2026-08-28 16:40:51 BST`.
- Compressed metrics: base loss `0.00238452`, continuity `0.0225081`, momentum `0.950313`, h1 error `0.0748036`, Sobolev s012 `5.35111`, vorticity error `0.0755629`.
- Interpretation: this evaluated run has the smallest recorded changes among the completed artifacts (base loss `+2.35%`, continuity `+0.20%`, h1 `+1.58%`, Sobolev s012 `+0.73%`, vorticity `+1.56%`).
