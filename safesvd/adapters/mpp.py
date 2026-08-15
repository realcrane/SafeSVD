from __future__ import annotations

from pathlib import Path

from safesvd.adapters.base import Adapter, RunRequest
from safesvd.assets import require_existing


class MPPAdapter(Adapter):
    name = "mpp"

    def validate(self, request: RunRequest) -> None:
        require_existing(request.assets.resolved_model_path, "MPP checkpoint")
        require_existing(request.assets.resolved_data_root, "MPP data root")

    def working_directory(self, request: RunRequest) -> Path:
        return request.repo_root / "backends" / "mpp"

    def command(self, request: RunRequest) -> list[str]:
        root = self.working_directory(request)
        script = root / "paper" / "safe_svd.py"
        return [
            *self.python(script),
            "--ckpt_path", str(request.assets.resolved_model_path),
            "--data_root", str(request.assets.resolved_data_root),
            "--save_path", str(request.run_dir / "compressed"),
            "--profile_cache", str(request.run_dir / "cache" / "profile.pt"),
            "--fisher_cache", str(request.run_dir / "cache" / "fisher.pt"),
            "--rank_cache", str(request.run_dir / "ranks" / "rank.json"),
            "--DEV", request.device,
            "--rank_ratio", str(request.config.rank_ratio),
            "--calibration_num", str(request.config.calibration_size),
            "--calibration_seed", str(request.config.calibration_seed),
            "--alpha", str(request.config.alpha),
            *request.config.extra,
        ]
