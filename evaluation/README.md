# Evaluation JSON format

All three evaluators write one JSON file keyed by dataset name. Each dataset
uses Poseidon's result structure, including a flat `metrics` object:

```json
{
  "DATASET_NAME": {
    "dataset": "DATASET_NAME",
    "eval_obj": "ongt",
    "batch_size": 64,
    "origin": {
      "model": "/path/to/original/checkpoint",
      "metrics": {
        "base_loss": 0.1,
        "rel_base_loss": 0.2,
        "model_size": 100000,
        "linear_size": 90000
      }
    },
    "compressed": {
      "model": "/path/to/compressed_model.pt",
      "metrics": {
        "base_loss": 0.11,
        "rel_base_loss": 0.22,
        "model_size": 50000,
        "linear_size": 40000
      },
      "increase": {
        "base_loss": 10.0,
        "rel_base_loss": 10.0
      }
    }
  }
}
```

The `compressed` entry is omitted for baseline-only runs. Poseidon also uses
this outer dataset grouping, even when evaluating a single dataset. Consumers
should read `results[dataset]["origin"]["metrics"]` and, when present,
`results[dataset]["compressed"]["increase"]`.

- `model_size` and `linear_size` are parameter counts, not bytes.
- MPP's former `mse` and `rel_mse` fields are named `base_loss` and
  `rel_base_loss`; VICON's `losses` and `relative_losses` use those same names.
  Values and metric calculations are unchanged. Poseidon's loss fields retain
  their existing meaning, including `"-"` for an unavailable relative loss.
- Physical metric names and calculations remain model-specific. They appear
  directly alongside the loss and size fields, without a `physics` wrapper.
- `increase` follows Poseidon: `(compressed - origin) / origin * 100`, rounded
  to six decimal places. Size fields, nonnumeric values, missing metrics, and
  metrics with a zero baseline are omitted.
- Evaluation settings stay alongside `dataset`, `origin`, and `compressed`.
  MPP retains each dataset title and category, with batch counts in `batches`.
  VICON keeps pass counts and label-energy diagnostics in `sampling`. These
  metadata do not participate in percentage changes.

CLI arguments, output paths, evaluation splits, and sampling protocols are
unchanged. Existing JSON files are not rewritten automatically.
