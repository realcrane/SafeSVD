"""Minimal activation-capture helpers used by SAFE-SVD compression."""

from __future__ import annotations

import io
import logging
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
from transformers.utils import logging as hf_logging
from tqdm import tqdm
from poseidon_main.scOT.inference import rollout
from SAFE_SVD.helper.whitener import *


class _StopForward(Exception):
    """Stop the current calibration example after the target layer is reached."""

@torch.no_grad()
def profile_inputs(
    trainer,
    calibration_dataset,
    device_name: str,
    whitening_mode="origin",
    whitening_eps=1e-6,
    whitening_eps_mode="absolute",
    return_sample_counts=False,
):
    """Return input whiteners, optionally together with per-layer sample counts.

    ``origin`` uses Fisher's trace-normalized Cholesky/eigen construction;
    ``new`` uses the spectral input whitener directly.
    """
    # torch.device(...) -> torch.device，例如 "cuda" -> device(type="cuda")。
    device = torch.device(device_name)
    # model.to(...).eval() -> 同一个 nn.Module，已移动到目标设备并切换为 eval 模式。
    model = trainer.model.to(device).eval()
    layers: list[tuple[str, nn.Linear]] = []
    handles = []
    count_dict: dict[str, int] = {}

    def collect(module, inputs, _output):
        # reshape(...) -> [token_or_pixel_count, module.in_features] 的二维输入。
        features = inputs[0].detach().float().reshape(-1, module.in_features)
        module.safe_input_gram.addmm_(features.T, features)
        count_dict[module.safe_input_name] += features.shape[0]

    # named_modules() -> (模块名, 模块) 迭代器，例如 ("encoder...query", nn.Linear)。
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # torch.zeros(...) -> [in_features, in_features] 的零 Gram 矩阵。
            module.safe_input_gram = torch.zeros(
                module.in_features,
                module.in_features,
                device=device,
                dtype=torch.float32,
            )
            module.safe_input_name = name
            count_dict[name] = 0
            # register_forward_hook(...) -> RemovableHandle，用于结束后移除 hook。
            handles.append(module.register_forward_hook(collect))
            layers.append((name, module))

    print("==== Collecting input Gram matrices ====")
    try:
        # rollout(...) -> 推理评估结果；这里忽略返回值，只利用 hooks 累积输入 Gram。
        rollout(
            trainer,
            calibration_dataset,
            model_path=trainer.model.config.name_or_path,
        )
    finally:
        for handle in handles:
            handle.remove()

    profiling = {}
    # tqdm(...) -> 包装后的层迭代器，只增加进度显示。
    for name, module in tqdm(layers, desc="Whitening input Grams"):
        # Each flattened activation vector is one sample of X.  SAFE-SVD
        # defines Sigma_XX = E[XX.T], so form an empirical mean rather than
        # using the accumulated Gram sum.
        gram_mean = module.safe_input_gram.double() / count_dict[name]
        if whitening_mode == "origin":
            left_factor, left_inverse = fisher_cholesky_whitener(
                module.safe_input_gram.double(), count_dict[name]
            )
            # Fisher returns a left factor with Ll.T @ Ll = Gram.  Input
            # whitening needs a right factor with Lr @ Lr.T = Gram.
            profiling[name] = (left_factor.T.cpu(), left_inverse.cpu())
            del left_factor, left_inverse
            torch.cuda.empty_cache()
        elif whitening_mode == "new":
            # svd_whitener(...) -> (Lr, Lr_invT)，两者形状均为 [in_features, in_features]。
            resolved_eps = resolve_whitening_eps(
                gram_mean,
                whitening_eps,
                whitening_eps_mode,
                sample_count=count_dict[name],
            )
            right, right_inverse_transpose = svd_whitener(
                gram_mean, eps=resolved_eps,
            )
            profiling[name] = (
                right.cpu().float(),
                right_inverse_transpose.cpu().float(),
            )
        else:
            raise ValueError(
                f"Unknown whitening_mode={whitening_mode!r}; expected 'origin' or 'new'."
            )
        del module.safe_input_gram
        del module.safe_input_name
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if return_sample_counts:
        return profiling, count_dict
    return profiling


@contextmanager
def quiet_everything():
    old_env = {
        key: os.environ.get(key)
        for key in ("TQDM_DISABLE", "HF_HUB_DISABLE_PROGRESS_BARS")
    }
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    old_hf_level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    loggers = [
        logging.getLogger(),
        logging.getLogger("transformers"),
        logging.getLogger("poseidon_main"),
    ]
    old_levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            yield
        finally:
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


def set_module_by_name(
    model: nn.Module, dotted_name: str, replacement: nn.Module
) -> None:
    if "." not in dotted_name:
        setattr(model, dotted_name, replacement)
        return
    parent_name, attribute = dotted_name.rsplit(".", 1)
    setattr(get_module_by_name(model, parent_name), attribute, replacement)


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


@torch.no_grad()
def input_catch(
    trainer,
    model: nn.Module,
    module_name: str,
    calibration_dataset,
    device: torch.device,
    aggregate: str = "concat",
):
    """Collect inputs to one module along the same rollout path used at inference."""
    trainer.model = model
    model.to(device).eval()
    target = get_module_by_name(model, module_name)
    inputs = []

    def capture(_module, args):
        value = args[0]
        if isinstance(value, (list, tuple)):
            value = value[0]
        inputs.append(_concat_if_list(value).detach().cpu())
        raise _StopForward

    handle = target.register_forward_pre_hook(capture)
    try:
        with disable_ar_steps(trainer), swallow_stop_and_mute_metrics(trainer), quiet_everything():
            rollout(
                trainer,
                calibration_dataset,
                model_path=getattr(trainer.model.config, "name_or_path", ""),
            )
    finally:
        handle.remove()

    if not inputs:
        raise RuntimeError(f"No inputs captured for {module_name}.")
    if aggregate == "list":
        return inputs
    return torch.cat(inputs, dim=0)


def to_2d_features(value: torch.Tensor, in_features: int) -> torch.Tensor:
    if value.ndim == 2 and value.shape[1] == in_features:
        return value
    if value.shape[-1] == in_features:
        return value.reshape(-1, in_features)
    if value.ndim == 2 and value.shape[0] == in_features:
        return value.T
    raise ValueError(
        f"in_features={in_features}, input shape={tuple(value.shape)} is incompatible"
    )
