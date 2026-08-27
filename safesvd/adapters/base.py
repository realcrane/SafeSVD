"""Adapters from the public SAFE-SVD contract to paper backends."""

from __future__ import annotations

import abc
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from safesvd.assets import AssetLayout
from safesvd.config import SafeSVDConfig, SafeSVDEvaluationConfig


@dataclass(frozen=True)
class RunRequest:
    repo_root: Path
    assets: AssetLayout
    run_dir: Path
    device: str
    config: SafeSVDConfig


@dataclass(frozen=True)
class EvaluationRequest:
    repo_root: Path
    assets: AssetLayout
    compressed_model: Path | None
    output_path: Path
    device: str
    config: SafeSVDEvaluationConfig


class Adapter(abc.ABC):
    """Build, validate, and execute one model's paper-compatible backend."""

    name: str

    @abc.abstractmethod
    def validate(self, request: RunRequest) -> None:
        """Fail before a costly run if required user assets are absent."""

    @abc.abstractmethod
    def command(self, request: RunRequest) -> list[str]:
        """Translate the common SAFE-SVD config into a backend command."""

    @abc.abstractmethod
    def working_directory(self, request: RunRequest) -> Path:
        """Return the legacy implementation's required working directory."""

    def environment(self, request: RunRequest) -> dict[str, str]:
        """Expose paths to future model/data loaders without hard-coding them."""
        env = os.environ.copy()
        env["SAFE_SVD_ASSETS_DIR"] = str(request.assets.assets_dir)
        env["SAFE_SVD_MODEL_PATH"] = str(request.assets.resolved_model_path)
        env["SAFE_SVD_DATA_ROOT"] = str(request.assets.resolved_data_root)
        env["SAFE_SVD_RUN_DIR"] = str(request.run_dir)
        return env

    def execute(self, request: RunRequest) -> int:
        return subprocess.run(
            self.command(request),
            cwd=self.working_directory(request),
            env=self.environment(request),
            check=False,
        ).returncode

    def evaluation_command(self, request: EvaluationRequest) -> list[str]:
        """Translate the public evaluator config, if this backend supports it."""
        raise NotImplementedError(f"{self.name} does not expose an evaluator")

    def validate_evaluation(self, request: EvaluationRequest) -> None:
        """Default validation common to all evaluator routes."""
        require = request.assets.resolved_model_path
        if not require.exists():
            raise FileNotFoundError(f"{self.name} checkpoint was not found at {require}")
        if not request.assets.resolved_data_root.exists():
            raise FileNotFoundError(f"{self.name} data root was not found at {request.assets.resolved_data_root}")
        if request.compressed_model is not None and not request.compressed_model.is_file():
            raise FileNotFoundError(f"compressed model was not found at {request.compressed_model}")

    def execute_evaluation(self, request: EvaluationRequest) -> int:
        return subprocess.run(
            self.evaluation_command(request),
            cwd=self.working_directory_for_evaluation(request),
            env=self.evaluation_environment(request),
            check=False,
        ).returncode

    def working_directory_for_evaluation(self, request: EvaluationRequest) -> Path:
        return request.repo_root

    def evaluation_environment(self, request: EvaluationRequest) -> dict[str, str]:
        env = os.environ.copy()
        env["SAFE_SVD_ASSETS_DIR"] = str(request.assets.assets_dir)
        env["SAFE_SVD_MODEL_PATH"] = str(request.assets.resolved_model_path)
        env["SAFE_SVD_DATA_ROOT"] = str(request.assets.resolved_data_root)
        return env

    @staticmethod
    def python(script: Path) -> list[str]:
        return [sys.executable, "-u", str(script)]
