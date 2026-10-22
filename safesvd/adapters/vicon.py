from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from safesvd.adapters.base import Adapter, EvaluationRequest, RunRequest
from safesvd.assets import BackendArtifactLayout, require_existing


class VICONAdapter(Adapter):
    name = "vicon"

    def validate(self, request: RunRequest) -> None:
        require_existing(request.assets.resolved_model_path, "VICON checkpoint")
        require_existing(request.assets.resolved_data_root, "VICON data root")
        if not request.assets.resolved_model_path.is_file():
            raise FileNotFoundError(f"VICON checkpoint must be a file: {request.assets.resolved_model_path}")
        package = self._package_root(request)
        require_existing(package / "src", "VICON source package")
        require_existing(package / "configs", "VICON configuration directory")

    def working_directory(self, request: RunRequest) -> Path:
        return request.repo_root

    def command(self, request: RunRequest) -> list[str]:
        cache_dir, rank_dir = self._artifact_dirs(request)
        return [
            sys.executable, "-u", "-m", "safesvd.adapters.vicon_runner", "compress",
            "--vicon-root", str(self.source_root(request)),
            "--model-path", str(request.assets.resolved_model_path),
            "--data-path", str(request.assets.resolved_data_root),
            "--save-path", str(request.run_dir / "compressed"),
            "--profile-path", str(cache_dir / "profile.pt"),
            "--fisher-path", str(cache_dir / "fisher.pt"),
            "--rank-path", str(rank_dir / "ranks.json"),
            "--device", request.device,
            "--rank-ratio", str(request.config.rank_ratio),
            "--rank-selector", request.config.rank_selector,
            "--calibration-size", str(request.config.calibration_size),
            "--calibration-seed", str(request.config.calibration_seed),
            "--alpha", str(request.config.alpha),
            *( ("--overwrite",) if request.config.overwrite else () ),
            *request.config.extra,
        ]

    def environment(self, request: RunRequest) -> dict[str, str]:
        env = super().environment(request)
        env["SAFE_SVD_VICON_ROOT"] = str(self.source_root(request))
        return env

    def validate_evaluation(self, request: EvaluationRequest) -> None:
        super().validate_evaluation(request)
        if not request.assets.resolved_model_path.is_file():
            raise FileNotFoundError(f"VICON checkpoint must be a file: {request.assets.resolved_model_path}")
        package = self._package_root(request)
        require_existing(package / "src", "VICON source package")
        require_existing(package / "configs", "VICON configuration directory")

    def evaluation_command(self, request: EvaluationRequest) -> list[str]:
        command = [
            sys.executable, "-u", "-m", "safesvd.adapters.vicon_runner", "evaluate",
            "--vicon-root", str(self.source_root(request)),
            "--model-path", str(request.assets.resolved_model_path),
            "--data-path", str(request.assets.resolved_data_root),
            "--output-path", str(request.output_path),
            "--device", request.device,
        ]
        if request.compressed_model is not None:
            command.extend(("--compressed-path", str(request.compressed_model)))
        if request.config.batch_size is not None:
            command.extend(("--batch-size", str(request.config.batch_size)))
        if request.config.max_batches is not None:
            command.extend(("--max-batches", str(request.config.max_batches)))
        return [*command, *request.config.extra]

    def evaluation_environment(self, request: EvaluationRequest) -> dict[str, str]:
        env = super().evaluation_environment(request)
        env["SAFE_SVD_VICON_ROOT"] = str(self.source_root(request))
        return env

    def source_root(self, request: RunRequest | EvaluationRequest) -> Path:
        return (request.config.source_root or Path(os.environ.get("SAFE_SVD_VICON_ROOT", request.repo_root / "VICON"))).expanduser().resolve()

    def _package_root(self, request: RunRequest | EvaluationRequest) -> Path:
        source = self.source_root(request)
        return source / "VICON" if (source / "VICON").exists() else source

    def _artifact_dirs(self, request: RunRequest) -> tuple[Path, Path]:
        key = _cache_key(request, self.source_root(request))
        artifacts = BackendArtifactLayout(request.repo_root, self.name)
        return artifacts.cache_root / key, artifacts.rank_root / request.config.rank_selector / key


def _cache_key(request: RunRequest, source_root: Path) -> str:
    checkpoint = request.assets.resolved_model_path
    stat = checkpoint.stat() if checkpoint.exists() else None
    fields = (
        "vicon_research_v2",
        str(checkpoint), str(stat.st_size if stat else ""), str(stat.st_mtime_ns if stat else ""),
        str(request.assets.resolved_data_root), str(source_root), str(request.config.calibration_size),
        str(request.config.calibration_seed), str(request.config.rank_ratio), str(request.config.alpha),
        *request.config.extra,
    )
    return hashlib.sha256("\0".join(fields).encode()).hexdigest()[:16]
