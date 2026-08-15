"""Adapters from the public SAFE-SVD contract to paper backends."""

from __future__ import annotations

import abc
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from safesvd.assets import AssetLayout
from safesvd.config import SafeSVDConfig


@dataclass(frozen=True)
class RunRequest:
    repo_root: Path
    assets: AssetLayout
    run_dir: Path
    device: str
    config: SafeSVDConfig


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

    @staticmethod
    def python(script: Path) -> list[str]:
        return [sys.executable, "-u", str(script)]
