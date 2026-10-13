"""Poseidon rollout capture, copied from SAFE_SVD/helper/input_collector.py.

Only capture-related code is included. Model inference remains a model-runtime
operation: pass rollout_fn, or use the official scOT runtime.
No original SAFE_SVD module is imported.
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import torch
from torch import nn

from ..collector import feature_rows, tree_to


class _StopForward(Exception):
    """Stop the current rollout batch when the target input is reached."""


@contextmanager
def quiet_everything():
    old_env = {
        key: os.environ.get(key)
        for key in ("TQDM_DISABLE", "HF_HUB_DISABLE_PROGRESS_BARS")
    }
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from transformers.utils import logging as hf_logging
    except ModuleNotFoundError:
        hf_logging = None
    old_hf_level = hf_logging.get_verbosity() if hf_logging is not None else None
    if hf_logging is not None:
        hf_logging.set_verbosity_error()
    loggers = [
        logging.getLogger(),
        logging.getLogger("transformers"),
        logging.getLogger("scOT"),
    ]
    old_levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            yield
        finally:
            if hf_logging is not None:
                hf_logging.set_verbosity(old_hf_level)
            for logger, level in zip(loggers, old_levels):
                logger.setLevel(level)
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _concat_if_list(value):
    if isinstance(value, list):
        try:
            return torch.cat(value, dim=0)
        except (RuntimeError, TypeError):
            return value
    return value


def get_module_by_name(model: nn.Module, dotted_name: str) -> nn.Module:
    module = model
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


@contextmanager
def disable_ar_steps(trainer):
    old_steps = getattr(trainer, "ar_steps", None)
    old_output_all_steps = getattr(trainer, "output_all_steps", False)
    if hasattr(trainer, "set_ar_steps"):
        trainer.set_ar_steps(None, output_all_steps=False)
    try:
        yield
    finally:
        if hasattr(trainer, "set_ar_steps"):
            trainer.set_ar_steps(old_steps, output_all_steps=old_output_all_steps)


@contextmanager
def swallow_stop_and_mute_metrics(trainer):
    def model_device(model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    old_metrics = getattr(trainer, "compute_metrics", None)
    trainer.compute_metrics = None
    original_prediction_step = getattr(trainer, "prediction_step", None)
    original_compute_loss = getattr(trainer, "compute_loss", None)

    if original_prediction_step is not None:
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            try:
                return original_prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys
                )
            except _StopForward:
                return (torch.tensor(0.0, device=model_device(model)), None, None)

        trainer.prediction_step = prediction_step.__get__(trainer, type(trainer))

    if original_compute_loss is not None:
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            try:
                return original_compute_loss(
                    model, inputs, return_outputs=return_outputs, **kwargs
                )
            except _StopForward:
                zero = torch.tensor(0.0, device=model_device(model))
                return (zero, {}) if return_outputs else zero

        trainer.compute_loss = compute_loss.__get__(trainer, type(trainer))

    try:
        yield
    finally:
        trainer.compute_metrics = old_metrics
        if original_prediction_step is not None:
            trainer.prediction_step = original_prediction_step
        if original_compute_loss is not None:
            trainer.compute_loss = original_compute_loss



def run_rollout(trainer, dataset, rollout_fn=None):
    """Invoke model inference; import it lazily only for the Poseidon route."""
    if rollout_fn is None:
        from scOT.inference import rollout as rollout_fn
    return rollout_fn(
        trainer, dataset,
        # model_path=getattr(getattr(trainer.model, "config", None), "name_or_path", ""),
    )


@torch.no_grad()
def capture_rollout(trainer, model, module_name, dataset, device, *, transform=None, rollout_fn=None,
                    capture_device=None):
    """Shared pre-hook/rollout for ordinary inputs and cached block arguments.

    transform receives the full argument tuple. Hooks and trainer state are
    restored before returning, including when rollout or transformation fails.
    capture_device can offload arguments inside the hook, before the next batch
    or the student's forward, while the model itself stays on device.
    """
    old_model = trainer.model
    model.to(device).eval()
    captured = []
    def pre(_module, args):
        value = transform(args) if transform is not None else tree_to(args, capture_device or device)
        captured.append(value)
        raise _StopForward
    handle = get_module_by_name(model, module_name).register_forward_pre_hook(pre)
    try:
        trainer.model = model
        with disable_ar_steps(trainer), swallow_stop_and_mute_metrics(trainer), quiet_everything():
            run_rollout(trainer, dataset, rollout_fn)
    finally:
        handle.remove()
        trainer.model = old_model
    if not captured:
        raise RuntimeError(f"No inputs captured for {module_name}.")
    return captured


@torch.no_grad()
def input_catch(trainer, model, module_name, calibration_dataset, device, aggregate="concat", *, rollout_fn=None):
    """Copy of the original ordinary-input capture using shared rollout cleanup."""
    def transform(args):
        value = args[0]
        if isinstance(value, (list, tuple)):
            value = value[0]
        return _concat_if_list(value).detach().cpu()
    captured = capture_rollout(
        trainer, model, module_name, calibration_dataset, device,
        transform=transform, rollout_fn=rollout_fn,
    )
    return captured if aggregate == "list" else torch.cat(captured, dim=0)


to_2d_features = feature_rows
