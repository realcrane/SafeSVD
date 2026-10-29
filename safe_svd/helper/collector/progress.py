"""Calibration batch totals without reading examples or advancing loaders."""

from __future__ import annotations

import math

from torch.utils.data import DataLoader


def calibration_batch_count(data, model_type, options):
    """Return an exact sized-loader total, or None for unsized streams."""
    try:
        if model_type == "mpp":
            size = options["params"].batch_size
            limit = options.get("batches_per_dset", 16)
            counts = [len(subset) // size for group in data.sub_dsets
                      for subset in group.get_per_file_dsets()]
        elif model_type == "vicon":
            limit = options.get("max_calib_batches")
            counts = [len(loader) for loader in data.values()]
        else:
            limit = options.get("max_calib_batches")
            loader = data if isinstance(data, DataLoader) else options.get("fisher_loader")
            counts = [len(loader) if loader is not None else
                      math.ceil(len(data) / options.get("fisher_batch_size", 16))]
    except (TypeError, NotImplementedError):
        return None
    return sum(min(count, math.ceil(limit)) if limit is not None else count for count in counts)
