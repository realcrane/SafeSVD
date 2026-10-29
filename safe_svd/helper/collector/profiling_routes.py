"""Local model runners and losses for the two shared calibration collectors."""

from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Mapping
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .collector import CalibrationBatch, trainer_model, tree_to
from .mpp.mpp_collector import iter_mpp_batches
from .vicon.vicon_collector import iter_vicon_batches, prepare_vicon_batch
from .mpp.mpp_fisher_loss import DSET_SOBOLEV_CONFIG, sobolev_loss_mpp
from .vicon.vicon_fisher_loss import sobolev_loss_for_fisher
from .vicon.vicon_fisher_forward import _patched_forward
from .poseidon.fisher_loss import sobolev_loss_calc_denorm
from .poseidon.input_collector import run_rollout


MODEL_TYPES = ("mpp", "vicon", "poseidon")
VICON_SOBOLEV_ORDERS = {"NS2D": 2, "COMPRESSIBLE2D": 1, "EULER2D": 1}


def validate_route(model_type, options):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")
    required = "params" if model_type == "mpp" else "trainer"
    if options.get(required) is None:
        raise ValueError(f"{model_type} route_options requires {required}")


def iter_calibration_batches(data, model_type, device, options):
    if model_type == "mpp":
        yield from iter_mpp_batches(data, device, options)
    elif model_type == "vicon":
        yield from iter_vicon_batches(data, options)
    else:
        loader = data if isinstance(data, DataLoader) else options.get("fisher_loader")
        if loader is None:
            loader = DataLoader(data, batch_size=options.get("fisher_batch_size", 16),
                                shuffle=False, num_workers=options.get("num_workers", 0))
        limit = options.get("max_calib_batches")
        if limit is not None and limit < 1:
            raise ValueError("max_calib_batches must be positive or None")
        for index, batch in enumerate(loader):
            if limit is not None and index >= limit:
                break
            yield CalibrationBatch(options.get("dataset_name", "poseidon"), batch)


def calibration_batch_count(data, model_type, options):
    """Count sized calibration batches without reading any examples.

    Stream datasets have no exact length; tqdm then displays the batch count
    and elapsed time without inventing a total.
    """
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


def run_input_profile(model, data, model_type, device, options, set_category):
    """Use full forward / full rollout, rather than compression's early stop."""
    if model_type == "poseidon":
        dataset = data.dataset if isinstance(data, DataLoader) else data
        set_category(options.get("dataset_name", "poseidon"))
        with trainer_model(options["trainer"], model):
            run_rollout(options["trainer"], dataset, options.get("rollout_fn"))
    else:
        if model_type == "vicon":
            seed = options.get("train_seed", getattr(options.get("cfg"), "train_seed", None))
            if seed is not None:
                # Original profile_svd_vicon calls utils.set_seed(cfg.train_seed).
                import random
                import numpy as np
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        for tagged in iter_calibration_batches(data, model_type, device, options):
            set_category(tagged.category)
            if model_type == "mpp":
                inp, labels, bcs, _target = tagged.payload
                model(inp, labels, bcs)
            else:
                trainer = options["trainer"]
                with trainer_model(trainer, model):
                    prepared, _, _ = prepare_vicon_batch(trainer, tagged.payload, device)
                    trainer._model_forward(prepared)


def fisher_context(model, model_type, options):
    return (_patched_forward(model, use_gradient_checkpointing=options.get("use_gradient_checkpointing", True))
            if model_type == "vicon" else nullcontext())


def loss_function(options, default):
    name = options.get("gradient_loss", options.get("loss_type", default))
    functions = {"l1": F.l1_loss, "mse": F.mse_loss, "smooth_l1": F.smooth_l1_loss}
    if name not in functions:
        raise ValueError(f"Unsupported gradient_loss: {name}")
    return functions[name]


def mpp_fisher_loss(model, tagged, device, options):
    inp, labels, bcs, target = tagged.payload
    output = model(inp, labels, bcs)
    spatial_dims = tuple(range(2, output.ndim))
    base = (((output - target).square().mean(spatial_dims, keepdim=True)) /
            (1e-7 + target.square().mean(spatial_dims, keepdim=True))).mean()
    order = options.get("sobolev_orders", {}).get(tagged.category,
             DSET_SOBOLEV_CONFIG.get(tagged.category, {}).get("order", 0))
    if order <= 0:
        return base
    return base + sobolev_loss_mpp(
        tagged.category, output, target, inp[-1], loss_function(options, "l1"),
        order_control=order, balance_ratio=options.get("balance_ratio", 0.5),
        decay_alpha=options.get("decay_alpha", 1.0), base_loss_val=base.detach().item(),
    )


def vicon_fisher_loss(model, tagged, device, options):
    trainer = options["trainer"]
    order = options.get("sobolev_orders", VICON_SOBOLEV_ORDERS).get(tagged.category, 1)
    with trainer_model(trainer, model):
        data, _, _ = prepare_vicon_batch(trainer, tagged.payload, device)
        return sobolev_loss_for_fisher(trainer, data, order=order,
                                      balance_ratio=options.get("balance_ratio", 10.0),
                                      loss_func=loss_function(options, "l1"))


def _output_field(output, name):
    return output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)


def poseidon_fisher_loss(model, tagged, device, options):
    batch = tagged.payload
    inputs = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
    if "pixel_values" in inputs:
        values = inputs["pixel_values"]
        if values.ndim == 3:
            inputs["pixel_values"] = values.unsqueeze(0)
        elif values.ndim == 5:
            inputs["pixel_values"] = values[:, 0]
    with trainer_model(options["trainer"], model):
        output = model(**inputs)
    base = _output_field(output, "loss")
    if base is None and isinstance(output, (tuple, list)):
        base = output[0]
    prediction = _output_field(output, "output")
    if base is None or prediction is None:
        raise ValueError("Poseidon Fisher requires model outputs.loss and outputs.output")
    return base + sobolev_loss_calc_denorm(
        options.get("dataset_name", tagged.category), prediction, inputs["labels"],
        options.get("order_control", 0), base.detach().item(), loss_function(options, "mse"), device,
        constants=options.get("constants"), transpose=options.get("transpose", False),
        balance_ratio=options.get("balance_ratio", 0.5), decay_alpha=options.get("decay_alpha", 1.0),
    )


FISHER_LOSS_ROUTES = {"mpp": mpp_fisher_loss, "vicon": vicon_fisher_loss, "poseidon": poseidon_fisher_loss}


def fisher_options(data, model_type, options):
    resolved = dict(options)
    if model_type == "poseidon":
        dataset = data.dataset if isinstance(data, DataLoader) else data
        resolved.setdefault("constants", getattr(dataset, "constants", None))
        resolved.setdefault("transpose", getattr(dataset, "transpose", False))
    return resolved
