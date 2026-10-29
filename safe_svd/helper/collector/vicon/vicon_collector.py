"""VICON batch preparation and paired-input route."""

from __future__ import annotations

from tqdm.auto import tqdm

from ..collector import CalibrationBatch, CaptureContext, CapturedPairs, capture_input, feature_rows, tree_to
from ..progress import calibration_batch_count


def iter_vicon_batches(loaders, options):
    limit = options.get("max_calib_batches")
    if limit is not None and limit < 1:
        raise ValueError("max_calib_batches must be positive or None")
    for category, loader in loaders.items():
        for index, batch in enumerate(loader):
            if limit is not None and index >= limit:
                break
            yield CalibrationBatch(category, tree_to(batch, "cpu"))


def prepare_vicon_batch(trainer, batch, device):
    _, pairs, *_rest = batch
    pairs = trainer._move_to_device(tree_to(pairs, device))
    return trainer._data_preprocess(pairs)

def collect_vicon(context: CaptureContext, layer_name: str) -> CapturedPairs:
    """Mirror comb_compress.py: CPU raw-batch cache and trainer preprocessing.

    calibration_data maps dataset types to DataLoaders. options requires trainer
    and accepts max_calib_batches (None), applied separately to each loader.
    The model's attention must already have been converted to explicit Linears.
    """
    trainer = context.options.get("trainer")
    if trainer is None:
        raise ValueError("VICON route_options requires trainer")
    progress = context.options.get("progress", True)
    if "vicon_batches" not in context.runtime:
        batches = iter_vicon_batches(context.calibration_data, context.options)
        total = calibration_batch_count(context.calibration_data, "vicon", context.options) if progress else None
        with tqdm(batches, total=total, desc="Calibration prefetch (vicon)", unit="batch",
                  position=1, leave=False, dynamic_ncols=True, disable=not progress) as bar:
            cached_batches = []
            for batch in bar:
                if progress:
                    bar.set_postfix(dataset=batch.category)
                cached_batches.append((batch.category, batch.payload))
            context.runtime["vicon_batches"] = cached_batches
    features = context.teacher.get_submodule(layer_name).in_features
    with tqdm(context.runtime["vicon_batches"], desc=f"Input capture (vicon): {layer_name}", unit="batch",
              position=1, leave=False, dynamic_ncols=True, disable=not progress) as bar:
        for category, batch in bar:
            if progress:
                bar.set_postfix(dataset=category)
            old_model = trainer.model
            try:
                trainer.model = context.teacher
                data, _, _ = prepare_vicon_batch(trainer, batch, context.device)
                h = capture_input(context.teacher, layer_name, lambda: trainer._model_forward(data))
                trainer.model = context.student
                hp = capture_input(context.student, layer_name, lambda: trainer._model_forward(data))
            finally:
                trainer.model = old_model
            if h is None and hp is None:
                continue  # Optional branch absent for this dataset, as in MySVD.
            if h is None or hp is None:
                raise RuntimeError(f"VICON teacher/student reachability differs for {layer_name}")
            yield category, feature_rows(h, features), feature_rows(hp, features)


