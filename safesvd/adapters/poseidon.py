from __future__ import annotations

import sys
from pathlib import Path

from safesvd.adapters.base import Adapter, EvaluationRequest, RunRequest
from safesvd.assets import BackendArtifactLayout, require_existing


class PoseidonAdapter(Adapter):
    name = "poseidon"

    def validate(self, request: RunRequest) -> None:
        require_existing(request.assets.resolved_model_path, "Poseidon model path")
        require_existing(request.assets.resolved_data_root, "Poseidon data root")
        poseidon_root = self.source_root(request)
        require_existing(poseidon_root, "Poseidon source checkout")
        require_existing(poseidon_root / "scOT", "Unmodified Poseidon scOT package")

    def working_directory(self, request: RunRequest) -> Path:
        # Run from SAFE-SVD's root so per-backend cache roots remain stable.
        return request.repo_root

    def command(self, request: RunRequest) -> list[str]:
        poseidon_root = self.source_root(request)
        artifacts = BackendArtifactLayout(request.repo_root, self.name)
        # Use the interpreter that launched ``python -m safesvd.cli``.  This
        # keeps the adapter in the active Poseidon environment without asking
        # users to set a second interpreter path.
        runtime_python = sys.executable
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
            "--rank-selector", request.config.rank_selector,
            "--traj_num", str(request.config.calibration_size),
            "--alpha", str(request.config.alpha),
            *( ("--overwrite_cache",) if request.config.overwrite else () ),
            *request.config.extra,
        ]

    def validate_evaluation(self, request: EvaluationRequest) -> None:
        super().validate_evaluation(request)
        poseidon_root = self.source_root(request)
        require_existing(poseidon_root / "scOT", "Unmodified Poseidon scOT package")
        require_existing(request.assets.resolved_model_path / "config.json", "Poseidon checkpoint config")

    def evaluation_command(self, request: EvaluationRequest) -> list[str]:
        command = [
            sys.executable,
            "-u",
            str(request.repo_root / "evaluation" / "poseidon" / "evaluate.py"),
            "--poseidon-root", str(self.source_root(request)),
            "--data-root", str(request.assets.resolved_data_root),
            "--checkpoint", str(request.assets.resolved_model_path),
            "--output", str(request.output_path),
            "--device", request.device,
        ]
        if request.compressed_model is not None:
            command.extend(("--model", str(request.compressed_model)))
        if request.config.batch_size is not None:
            command.extend(("--batch-size", str(request.config.batch_size)))
        return [*command, *request.config.extra]

    def source_root(self, request: RunRequest | EvaluationRequest) -> Path:
        source = request.config.source_root
        return (source or request.repo_root / "poseidon").expanduser().resolve()
