from __future__ import annotations

from pathlib import Path

from safesvd.adapters.base import Adapter, RunRequest
from safesvd.assets import require_existing


class VICONAdapter(Adapter):
    name = "vicon"

    def validate(self, request: RunRequest) -> None:
        require_existing(request.assets.resolved_model_path, "VICON checkpoint")
        require_existing(request.assets.resolved_data_root, "VICON data root")

    def working_directory(self, request: RunRequest) -> Path:
        return request.repo_root / "backends" / "vicon"

    def command(self, request: RunRequest) -> list[str]:
        script = self.working_directory(request) / "paper" / "safe_svd.py"
        return [
            *self.python(script),
            "--ckpt_path", str(request.assets.resolved_model_path),
            "--save_path", str(request.run_dir / "compressed"),
            "--cache_dir", str(request.run_dir / "cache"),
            "--rank_root", str(request.run_dir / "ranks"),
            "--rank_path", str(request.run_dir / "ranks" / "rank.json"),
            "--data_root", str(request.assets.resolved_data_root),
            "--DEV", request.device,
            "--rank_ratio", str(request.config.rank_ratio),
            "--calibration_num", str(request.config.calibration_size),
            "--calibration_seed", str(request.config.calibration_seed),
            "--alpha", str(request.config.alpha),
            *request.config.extra,
        ]
