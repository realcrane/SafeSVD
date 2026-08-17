from __future__ import annotations

import os
from pathlib import Path

from safesvd.adapters.base import Adapter, RunRequest
from safesvd.assets import BackendArtifactLayout, require_existing


class PoseidonAdapter(Adapter):
    name = "poseidon"

    def validate(self, request: RunRequest) -> None:
        require_existing(request.assets.resolved_model_path, "Poseidon model path")
        require_existing(request.assets.resolved_data_root, "Poseidon data root")
        require_existing(request.repo_root / "poseidon", "Poseidon source checkout")

    def working_directory(self, request: RunRequest) -> Path:
        # Run from SAFE-SVD's root so per-backend cache roots remain stable.
        return request.repo_root

    def command(self, request: RunRequest) -> list[str]:
        poseidon_root = request.repo_root / "poseidon"
        artifacts = BackendArtifactLayout(request.repo_root, self.name)
        runtime_python = os.environ.get("SAFE_SVD_RUNTIME_PYTHON", "python3")
        return [
            runtime_python, "-u", "-m", "safesvd.adapters.poseidon_runner",
            "--repo-root", str(request.repo_root),
            "--poseidon-root", str(poseidon_root),
            "--model_path", str(request.assets.resolved_model_path),
            "--data_path", str(request.assets.resolved_data_root),
            "--save_path", str(request.run_dir / "compressed"),
            "--rank-root", str(artifacts.rank_root),
            "--cache-root", str(artifacts.cache_root),
            "--DEV", request.device,
            "--rank_ratio", str(request.config.rank_ratio),
            "--traj_num", str(request.config.calibration_size),
            "--calibration_seed", str(request.config.calibration_seed),
            "--alpha", str(request.config.alpha),
            *( ("--overwrite_cache",) if request.config.overwrite else () ),
            *request.config.extra,
        ]
