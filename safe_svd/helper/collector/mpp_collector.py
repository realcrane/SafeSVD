"""MPP batch preparation and paired-input route."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from .collector import CaptureContext, CapturedPairs, capture_input, feature_rows

def collect_mpp(context: CaptureContext, layer_name: str) -> CapturedPairs:
    """Mirror MySVD/mysvd.py: per-file shuffle, labels, and time-first input.

    calibration_data is MPP's training dataset; options requires params and
    accepts batches_per_dset (16) and calibration_seed (42).
    """
    params = context.options.get("params")
    if params is None:
        raise ValueError("MPP route_options requires params")
    limit = context.options.get("batches_per_dset", 16)
    if limit is not None and limit < 1:
        raise ValueError("batches_per_dset must be positive or None")
    seed = context.options.get("calibration_seed", 42)
    dataset = context.calibration_data
    features = context.teacher.get_submodule(layer_name).in_features
    for group in dataset.sub_dsets:
        for subset in group.get_per_file_dsets():
            loader = DataLoader(
                subset, batch_size=params.batch_size, num_workers=params.num_data_workers,
                shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True,
            )
            for index, batch in enumerate(loader):
                if limit is not None and index >= limit:
                    break
                inp_raw, bcs, target = (value.to(context.device) for value in batch)
                category = subset.get_name()
                label_values = dataset.subset_dict.get(category, [-1])
                labels = torch.tensor(label_values, device=context.device).unsqueeze(0).expand(target.shape[0], -1)
                if inp_raw.ndim != 5:
                    raise ValueError("MPP input must have shape (batch, time, channels, height, width)")
                inp = inp_raw.permute(1, 0, 2, 3, 4)
                h = capture_input(context.teacher, layer_name, lambda: context.teacher(inp, labels, bcs))
                hp = capture_input(context.student, layer_name, lambda: context.student(inp, labels, bcs))
                if h is None or hp is None:
                    raise RuntimeError(f"MPP target {layer_name} was not reached by both models")
                yield category, feature_rows(h, features), feature_rows(hp, features)


