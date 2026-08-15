from __future__ import annotations

from pathlib import Path

from safesvd.adapters.base import Adapter, RunRequest
from safesvd.assets import require_existing


class PoseidonAdapter(Adapter):
    name = "poseidon"

    def validate(self, request: RunRequest) -> None:
        require_existing(request.assets.resolved_model_path, "Poseidon model path")
        require_existing(request.assets.resolved_data_root, "Poseidon data root")

    def working_directory(self, request: RunRequest) -> Path:
        return request.repo_root / "backends" / "poseidon"

    def command(self, request: RunRequest) -> list[str]:
        script = self.working_directory(request) / "paper" / "safe_svd.py"
        return [
            *self.python(script),
            "--model_path", str(request.assets.resolved_model_path),
            "--data_path", str(request.assets.resolved_data_root),
            "--rank_path", str(request.run_dir / "ranks" / "rank.json"),
            "--save_path", str(request.run_dir / "compressed"),
            "--DEV", request.device,
            "--rank_ratio", str(request.config.rank_ratio),
            "--traj_num", str(request.config.calibration_size),
            "--calibration_seed", str(request.config.calibration_seed),
            "--alpha", str(request.config.alpha),
            *( ("--overwrite_cache",) if request.config.overwrite else () ),
            *request.config.extra,
        ]
