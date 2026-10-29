"""Capture VICON's compression entry stream for replay and checkpoint resume.

The research wrapper starts compression in a separate process. Statistics
cache hits never restore RNGs; only compression checkpoints replay the state
before prefetch. Snapshots are weights_only-compatible, without dataset objects.
"""

import copy
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Subset


def _dataset(loader):
    dataset = getattr(loader, "dataset", None)
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def capture(loaders):
    state = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), *state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "datasets": {},
    }
    for name, loader in loaders.items():
        if getattr(loader, "persistent_workers", False):
            raise ValueError("VICON compression replay requires persistent_workers=False")
        dataset = _dataset(loader)
        fields = {}
        fields["loader_settings"] = {key: getattr(loader, key, None)
                                     for key in ("batch_size", "num_workers", "drop_last")}
        for field in ("dp", "worker_id", "epoch"):
            if hasattr(dataset, field):
                fields[field] = copy.deepcopy(getattr(dataset, field))
        rng = getattr(dataset, "rng", None)
        tc_rng = getattr(dataset, "tc_rng", None)
        fields["rng"] = copy.deepcopy(rng.bit_generator.state) if rng is not None else None
        fields["tc_rng"] = tc_rng.get_state() if tc_rng is not None else None
        generator = getattr(loader, "generator", None)
        fields["loader_generator"] = generator.get_state() if generator is not None else None
        result["datasets"][name] = fields
    return result


def restore(loaders, state):
    if state is None:
        raise ValueError("VICON checkpoint lacks compression RNG state; use a new save_path")
    if list(loaders) != list(state["datasets"]):
        raise ValueError("VICON checkpoint datasets/order do not match the current calibration datasets")
    for name, loader in loaders.items():
        settings = {key: getattr(loader, key, None) for key in ("batch_size", "num_workers", "drop_last")}
        if settings != state["datasets"][name]["loader_settings"]:
            raise ValueError("VICON checkpoint batch size/worker settings changed; use a new save_path")
        if getattr(loader, "persistent_workers", False):
            raise ValueError("VICON compression replay requires persistent_workers=False")
        paths = state["datasets"][name].get("dp")
        if paths is not None and Counter(paths) != Counter(getattr(_dataset(loader), "dp", [])):
            raise ValueError("VICON checkpoint data files differ from current calibration data; use a new save_path")
    random.setstate(state["python"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("VICON cache CUDA device count changed; recompute the cache")
        torch.cuda.set_rng_state_all(state["cuda"])
    for name, loader in loaders.items():
        dataset = _dataset(loader)
        fields = state["datasets"][name]
        for field in ("dp", "worker_id", "epoch"):
            if field in fields:
                setattr(dataset, field, copy.deepcopy(fields[field]))
        if hasattr(dataset, "rng"):
            dataset.rng = None
            if fields["rng"] is not None:
                dataset.rng = np.random.default_rng()
                dataset.rng.bit_generator.state = copy.deepcopy(fields["rng"])
        if hasattr(dataset, "tc_rng"):
            dataset.tc_rng = None
            if fields["tc_rng"] is not None:
                dataset.tc_rng = torch.Generator().set_state(fields["tc_rng"])
        if fields["loader_generator"] is not None:
            if loader.generator is None:
                loader.generator = torch.Generator()
            loader.generator.set_state(fields["loader_generator"])
