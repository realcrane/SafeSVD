"""Lightweight MPP checkout discovery for the standalone evaluation CLI."""

from pathlib import Path


def default_mpp_package(repo_root: Path) -> Path:
    """Prefer the documented nested checkout, accepting a flat checkout too.

    Check for actual model code: an empty directory left by a failed clone
    must not shadow a usable checkout in the other location.
    """
    candidates = (repo_root / "mpp/multiple_physics_pretraining",
                  repo_root / "multiple_physics_pretraining")
    return next((path for path in candidates if (path / "models/avit.py").is_file()), candidates[0])
