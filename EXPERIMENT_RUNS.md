# Experiment Run Ledger

This is the canonical ledger for SAFE-SVD experiments launched from this
repository. It records only SAFE-SVD compression and SAFE-SVD evaluation runs;
unrelated projects and other low-rank methods are out of scope.

Last reconciliation attempt: 2026-08-29 07:15:53 BST (2026-08-29 14:15:53 Asia/Shanghai).

The remote path `/cs/research/vecg/EasyPDE/SAFE-SVD-v0` is on a shared
filesystem, so identical artifacts observed from several hosts are recorded
once. This reconciliation could not reach any UCL host: local DNS failed to
resolve the configured jump host `knuckles.cs.ucl.ac.uk`, and direct SSH to
`brent-l.cs.ucl.ac.uk` also failed. `check_ucl_gpus` therefore returned zero
reachable hosts. The statuses and metrics below remain the latest trustworthy
artifact observations from `2026-08-28 16:40:51 BST`; no remote state change
could be verified in this check.

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

### `poseidon_safe_seed42_r0.2`

- Status: `正在运行`
- Outcome: `pending`
- Task: Poseidon SAFE-SVD compression.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; rank selector `safe`; calibration size `64`; calibration seed `42`; `Poseidon-L`; `--fast`; final time `20`; alpha `0.7`; device `cuda`.
- Host / GPU: `ruddy`, GPU 0, NVIDIA GeForce RTX 3090 Ti; about 4.75 / 24.56 GiB used and 84% utilization at last check.
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

- Log: tmux pane/stdout-stderr on `/dev/pts/2`; no independent `.log` found.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed42_ftime20/`; temporary `compressed/compressing_model.pt` is 883,747,949 B; final model and metrics are not present.
- Submitted / started: process start `2026-08-28 07:50:23 BST` (PID `104165`, backend PID `104166`).
- Completed: not completed.
- Last checked: `2026-08-28 16:40:51 BST` (latest successful check); recheck attempted `2026-08-29 07:15:53 BST` but UCL DNS/SSH was unavailable.
- Notes: tmux showed `592/815` layers (73%), current layer `decoder.layers.1.upsample.upsample`, with an estimated 2 h 49 min remaining. The remote login banner reported about 93% disk-quota usage. The current Poseidon runtime resumes automatically from `compressed/compressing_model.pt` when `--overwrite` is absent; no relaunch was attempted because the host was unreachable and the old process could not be ruled out.

### `poseidon_safe_seed7_r0.2`

- Status: `正在运行`
- Outcome: `pending`
- Task: Poseidon SAFE-SVD compression.
- Hyperparameters: dataset `fluids.incompressible.PiecewiseConstants`; rank ratio `0.2`; rank selector `safe`; calibration size `64`; calibration seed `7`; `Poseidon-L`; `--fast`; final time `20`; alpha `0.7`; device `cuda`.
- Host / GPU: `smew`, GPU 0, NVIDIA GeForce RTX 3090 Ti; about 5.97 / 24.56 GiB used and 98% utilization at last check.
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

- Log: tmux pane/stdout-stderr; no independent `.log` found.
- Checkpoint / output: `/cs/research/vecg/EasyPDE/SAFE-SVD-v0/artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/traj64_seed7_ftime20/`; temporary `compressed/compressing_model.pt` is 2,502,460,411 B; final model and metrics are not present.
- Submitted / started: process start `2026-08-28 14:28:24 BST` (PID `166151`, backend PID `166152`).
- Completed: not completed.
- Last checked: `2026-08-28 16:40:51 BST`.
- Notes: tmux showed `269/815` layers (33%), current layer `encoder.layers.2.blocks.5.intermediate.dense`, with an estimated 4 h 09 min remaining. The remote login banner reported about 93% disk-quota usage.

## 已完成

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
