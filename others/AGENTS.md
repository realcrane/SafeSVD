# Codex Project Instructions

## Remote experiment operations

Use the following workflow whenever the user asks Codex to launch, monitor, or
inspect experiments for this repository.

### Scope

The experiment ledger covers only SAFE-SVD experiments launched from this
repository (including its `safesvd` CLI and SAFE-SVD evaluation entry points).
Do not add experiments from other repositories or unrelated methods and
training jobs to the ledger.

### Discover available GPUs

1. Run `check_ucl_gpus` in the local terminal to inspect the remote machines.
   Its output may include completely idle machines.
2. When launching experiments, select only machines with available NVIDIA
   GeForce RTX 3090 Ti GPUs unless the user explicitly requests another GPU
   type.
3. When the task is only to inspect existing results, any reachable UCL GPU
   machine is acceptable. Prefer the host associated with the existing tmux
   session, process, log, checkpoint, or result path; an idle 3090 Ti is not
   required for a read-only result check.
4. For launches, match the number of selected machines to the number and size
   of requested jobs. Use one machine when it is sufficient; use multiple
   machines when the jobs can run independently and parallel execution is
   useful.
5. Before launching anything, check for existing matching processes, tmux
   sessions, logs, or completed outputs so the same experiment is not started
   twice.

### Prepare a remote SAFE-SVD run

1. SSH into each selected machine.
2. Run `cd svd` to enter the remote SAFE-SVD repository.
3. Use the existing `poseidon` environment setup command or shell helper to
   activate the required environment. Verify that the environment is active
   before launching the experiment.
4. Inspect the current repository's configs, launch scripts, README, and
   implementation before constructing the command. Use the launch interface
   implemented by the current code; do not invent stale flags or assume an old
   command still applies.
5. Confirm the intended dataset, method, seed, rank selector, rank bounds,
   regularization settings, and any other requested hyperparameters from the
   user's request and the current code.

### Launch persistent jobs with tmux

1. Every long-running remote experiment must run inside a persistent tmux
   session created with:

   ```bash
   tmux new -s <session_name>
   ```

2. Derive `<session_name>` from the task and the hyperparameters that
   distinguish the run. Keep it concise and shell-safe. A suitable pattern is:

   ```text
   <dataset>_<method>_<selector>_r<rank>_seed<seed>_<key-settings>
   ```

3. Check `tmux ls` before creating the session. Reuse a session only when it
   clearly belongs to the same requested run; otherwise choose a unique,
   informative suffix rather than disrupting an existing session.

4. Launch the repository's current experiment command inside the tmux session
   and ensure stdout/stderr are written to the repository's normal log path or
   to an explicitly named log file.

5. After launch, verify that the process is alive, the expected GPU is in use,
   and the log has begun updating. Then update the experiment ledger described
   below before reporting that the launch succeeded.

### Maintain the experiment run ledger

Use the repository-root `EXPERIMENT_RUNS.md` as the single source of truth for
remote experiment status, placement, logs, and collected results.

1. Before launching or inspecting jobs, read `EXPERIMENT_RUNS.md` and reconcile
   it with the current remote process, tmux, log, and output state.
2. Track every requested SAFE-SVD run under exactly one of these three headings:
   `未开始`, `正在运行`, or `已完成`.
3. Add a planned run to `未开始` before launch. Include its task, identifying
   hyperparameters, seed, intended command, and any known output path.
4. After a launch is verified, move the same entry from `未开始` to `正在运行`
   and record its hostname, GPU, tmux session, exact command, log path,
   checkpoint/result path, and start time.
5. When a run reaches a terminal state, move the same entry to `已完成`; do not
   leave duplicate copies under earlier headings. Record the completion time
   and an outcome such as `success`, `failed`, or `cancelled`, while preserving
   relevant failure evidence.
6. When the user asks to inspect results, first use the ledger to locate the
   run. For every completed run inspected, write the collected metrics and a
   concise interpretation into a `Results` subsection directly below that
   completed task. Include the check time and the source log, checkpoint, or
   result file. Do not only report the result in chat.
7. Preserve previous results and notes. Update the existing task entry instead
   of creating a second entry for the same task and hyperparameter set.

### Report and inspect results

For every launched or inspected run, report at least:

- remote hostname and GPU assignment;
- tmux session name;
- exact launch command;
- configuration or hyperparameter overrides;
- log and checkpoint/result paths;
- current ledger state: `未开始`, `正在运行`, or `已完成`, plus terminal outcome
  when applicable;
- key metrics when they are available;
- concise failure evidence and the likely cause when a run fails.

When the user asks only to check results, do not relaunch the experiment.
Inspect the existing tmux session, process state, logs, checkpoints, and result
files, update `EXPERIMENT_RUNS.md`, then summarize the latest trustworthy
status.

### Safety and coordination

- Do not kill processes, close tmux sessions, overwrite logs or checkpoints,
  or restart experiments unless the user explicitly asks.
- When launching a run, do not occupy a GPU that is already in active use
  merely because the host is otherwise reachable.
- For launches, prefer an idle 3090 Ti with enough free memory for the
  requested run. This restriction does not apply to read-only result checks.
- Avoid duplicate launches across machines by recording and comparing the
  task, hyperparameters, seed, hostname, tmux session, and output path in
  `EXPERIMENT_RUNS.md`.
- If the requested command, environment helper, remote directory, GPU state,
  or expected output is ambiguous, perform read-only checks first and report
  the ambiguity instead of guessing destructively.
