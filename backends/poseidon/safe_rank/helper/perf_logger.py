#coding:utf8
import csv
import os
import sys
import time
import uuid
import threading
import resource
from contextlib import contextmanager
from datetime import datetime

import torch

try:
    import psutil
except Exception:
    psutil = None


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _bytes_to_mb(value):
    if value is None:
        return ""
    return float(value) / (1024.0 ** 2)


def _fallback_rss_bytes():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage
    return usage * 1024


def _current_rss_bytes():
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss
    return _fallback_rss_bytes()


def _safe_cuda_device(device):
    if device is None or not torch.cuda.is_available():
        return None
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    return dev if dev.index is not None else torch.device("cuda", torch.cuda.current_device())


def _cuda_mb(cuda_device, fn_name):
    if cuda_device is None:
        return ""
    fn = getattr(torch.cuda, fn_name, None)
    if fn is None:
        return ""
    return _bytes_to_mb(fn(cuda_device))


def _max_mb(current, candidate):
    if candidate == "":
        return current
    if current == "":
        return candidate
    return max(current, candidate)


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


class PerfCSVLogger:
    _cuda_depth = 0
    _cuda_depth_lock = threading.Lock()

    FIELDNAMES = [
        "run_id",
        "stage",
        "status",
        "start_time",
        "end_time",
        "elapsed_sec",
        "cpu_rss_start_mb",
        "cpu_rss_end_mb",
        "cpu_rss_peak_mb",
        "gpu_alloc_start_mb",
        "gpu_alloc_end_mb",
        "gpu_alloc_peak_mb",
        "gpu_reserved_start_mb",
        "gpu_reserved_end_mb",
        "gpu_reserved_peak_mb",
        "device",
        "cuda_device_name",
        "note",
        "phase",
        "method",
        "model_name",
        "dataset",
        "rank_ratio",
        "traj_num",
        "order_control",
        "gradient_loss",
        "alpha",
        "uniform_rank",
        "use_fisher",
        "final_time",
        "rank_path",
        "save_path",
    ]

    def __init__(self, csv_path, device=None, run_id=None, sample_interval=0.1, **metadata):
        self.csv_path = csv_path
        self.device = str(device) if device is not None else ""
        self.cuda_device = _safe_cuda_device(device)
        self.cuda_device_name = (
            torch.cuda.get_device_name(self.cuda_device) if self.cuda_device is not None else ""
        )
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.sample_interval = sample_interval
        self.metadata = {k: "" if v is None else str(v) for k, v in metadata.items()}
        self._ensure_header()

    def _ensure_header(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        needs_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        if needs_header:
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def _write_row(self, row):
        final_row = {name: "" for name in self.FIELDNAMES}
        for key, value in self.metadata.items():
            if key in final_row:
                final_row[key] = value
        for key, value in row.items():
            if key in final_row:
                final_row[key] = value
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(final_row)

    @contextmanager
    def stage(self, stage, note=""):
        did_reset_cuda_peak = False
        if self.cuda_device is not None:
            torch.cuda.synchronize(self.cuda_device)
            with PerfCSVLogger._cuda_depth_lock:
                if PerfCSVLogger._cuda_depth == 0:
                    reset_fn = getattr(torch.cuda, "reset_peak_memory_stats", None)
                    if reset_fn is not None:
                        reset_fn(self.cuda_device)
                        did_reset_cuda_peak = True
                PerfCSVLogger._cuda_depth += 1

        stop_event = threading.Event()
        cpu_start = _current_rss_bytes()
        cpu_peak = {"value": cpu_start}

        start_time = _now_iso()
        start_perf = time.perf_counter()
        gpu_alloc_start = _cuda_mb(self.cuda_device, "memory_allocated")
        gpu_reserved_start = _cuda_mb(self.cuda_device, "memory_reserved")
        gpu_alloc_peak = {"value": gpu_alloc_start}
        gpu_reserved_peak = {"value": gpu_reserved_start}
        status = "ok"

        def sample_memory():
            while not stop_event.is_set():
                try:
                    cpu_peak["value"] = max(cpu_peak["value"], _current_rss_bytes())
                    gpu_alloc_peak["value"] = _max_mb(
                        gpu_alloc_peak["value"],
                        _cuda_mb(self.cuda_device, "memory_allocated"),
                    )
                    gpu_reserved_peak["value"] = _max_mb(
                        gpu_reserved_peak["value"],
                        _cuda_mb(self.cuda_device, "memory_reserved"),
                    )
                except Exception:
                    pass
                stop_event.wait(self.sample_interval)

        thread = threading.Thread(target=sample_memory, daemon=True)
        thread.start()

        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            if self.cuda_device is not None:
                torch.cuda.synchronize(self.cuda_device)
            elapsed = time.perf_counter() - start_perf
            stop_event.set()
            thread.join(timeout=self.sample_interval * 2)
            try:
                cpu_peak["value"] = max(cpu_peak["value"], _current_rss_bytes())
            except Exception:
                pass
            gpu_alloc_end = _cuda_mb(self.cuda_device, "memory_allocated")
            gpu_reserved_end = _cuda_mb(self.cuda_device, "memory_reserved")
            gpu_alloc_peak["value"] = _max_mb(gpu_alloc_peak["value"], gpu_alloc_end)
            gpu_reserved_peak["value"] = _max_mb(gpu_reserved_peak["value"], gpu_reserved_end)
            if did_reset_cuda_peak:
                gpu_alloc_peak["value"] = _max_mb(
                    gpu_alloc_peak["value"],
                    _cuda_mb(self.cuda_device, "max_memory_allocated"),
                )
                gpu_reserved_peak["value"] = _max_mb(
                    gpu_reserved_peak["value"],
                    _cuda_mb(self.cuda_device, "max_memory_reserved"),
                )

            if self.cuda_device is not None:
                with PerfCSVLogger._cuda_depth_lock:
                    PerfCSVLogger._cuda_depth = max(PerfCSVLogger._cuda_depth - 1, 0)

            self._write_row({
                "run_id": self.run_id,
                "stage": stage,
                "status": status,
                "start_time": start_time,
                "end_time": _now_iso(),
                "elapsed_sec": f"{elapsed:.6f}",
                "cpu_rss_start_mb": f"{_bytes_to_mb(cpu_start):.3f}",
                "cpu_rss_end_mb": f"{_bytes_to_mb(_current_rss_bytes()):.3f}",
                "cpu_rss_peak_mb": f"{_bytes_to_mb(cpu_peak['value']):.3f}",
                "gpu_alloc_start_mb": _fmt(gpu_alloc_start),
                "gpu_alloc_end_mb": _fmt(gpu_alloc_end),
                "gpu_alloc_peak_mb": _fmt(gpu_alloc_peak["value"]),
                "gpu_reserved_start_mb": _fmt(gpu_reserved_start),
                "gpu_reserved_end_mb": _fmt(gpu_reserved_end),
                "gpu_reserved_peak_mb": _fmt(gpu_reserved_peak["value"]),
                "device": self.device,
                "cuda_device_name": self.cuda_device_name,
                "note": note,
            })


class NullPerfLogger:
    @contextmanager
    def stage(self, stage, note=""):
        yield


def make_perf_logger(csv_path, device=None, **metadata):
    if not csv_path:
        return NullPerfLogger()
    return PerfCSVLogger(csv_path, device=device, **metadata)
