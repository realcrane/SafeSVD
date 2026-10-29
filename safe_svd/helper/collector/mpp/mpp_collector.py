"""MPP batch preparation and paired-input route."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from ..collector import CalibrationBatch, CaptureContext, CapturedPairs, capture_input, feature_rows


def iter_mpp_batches(dataset, device, options):
    """One preparation path for input profiling, Fisher, and compression."""
    params = options.get("params")
    if params is None:
        raise ValueError("MPP route_options requires params")
    limit = options.get("batches_per_dset", 16)
    if limit is not None and limit <= 0:
        raise ValueError("batches_per_dset must be positive or None")
    seed = options.get("calibration_seed", 42)
    for group in dataset.sub_dsets:
        for subset in group.get_per_file_dsets():
            loader = DataLoader(subset, batch_size=params.batch_size, num_workers=params.num_data_workers,
                                shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True)
            for index, batch in enumerate(loader):
                if limit is not None and index >= limit:
                    break
                inp_raw, bcs, target = (value.to(device) for value in batch)
                category = subset.get_name()
                labels = torch.tensor(dataset.subset_dict.get(category, [-1]), device=device)
                labels = labels.unsqueeze(0).expand(target.shape[0], -1)
                if inp_raw.ndim != 5:
                    raise ValueError("MPP input must have shape (batch, time, channels, height, width)")
                yield CalibrationBatch(category, (inp_raw.permute(1, 0, 2, 3, 4), labels, bcs, target))

def collect_mpp(context: CaptureContext, layer_name: str) -> CapturedPairs:
    """Mirror MySVD/mysvd.py: per-file shuffle, labels, and time-first input.

    calibration_data is MPP's training dataset; options requires params and
    accepts batches_per_dset (16) and calibration_seed (42).
    """
    features = context.teacher.get_submodule(layer_name).in_features
    for batch in iter_mpp_batches(context.calibration_data, context.device, context.options):
        inp, labels, bcs, _target = batch.payload
        h = capture_input(context.teacher, layer_name, lambda: context.teacher(inp, labels, bcs))
        hp = capture_input(context.student, layer_name, lambda: context.student(inp, labels, bcs))
        if h is None or hp is None:
            raise RuntimeError(f"MPP target {layer_name} was not reached by both models")
        yield batch.category, feature_rows(h, features), feature_rows(hp, features)
