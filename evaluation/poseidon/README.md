# Poseidon evaluation

This directory mirrors the readable structure of Poseidon's
`poseidon_analysis/` evaluator:

- `evaluate.py` is the entry point: data setup, origin baseline,
  compressed-model evaluation, and JSON output.
- `physics.py` builds/aggregates the selected physical metrics.
- `PDE_calc.py` contains the PDE formulas for incompressible, compressible,
  wave, and elliptic datasets.

Performance profiling, plotting, and CSV result management are deliberately
not included. The rollout batch size defaults to `1024`, and the physical-loss
batch size to `10`, matching the original evaluator.

```bash
python evaluation/poseidon/evaluate.py \
  --poseidon-root ./poseidon \
  --data-root ./poseidon/camlab-ethz/down_streams \
  --checkpoint /path/to/original-finetuned-checkpoint \
  --model /path/to/run/compressed/compressed_model_origin.pt \
  --dataset fluids.incompressible.PiecewiseConstants
```

The JSON contains origin metrics, compressed metrics, and per-metric relative
increases. For a SAFE-SVD run it is written by default to
`<run>/evaluation/poseidon/metrics.json`. Omit `--model` to evaluate only the
original checkpoint. `--eval-obj oncomp` retains the original script's
fidelity mode, using original-model predictions as labels.
