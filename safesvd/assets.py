"""Location rules for downloaded models, datasets, and generated artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetLayout:
    """User-controlled storage layout.

    SAFE-SVD never writes downloaded assets into its source tree.  A caller may
    either give one ``assets_dir`` (using the conventional layout below) or
    override model/data locations independently.

    ``assets_dir/``
    ``├── models/{poseidon,vicon,mpp}/``
    ``└── data/{poseidon,vicon,mpp}/``
    """

    assets_dir: Path
    model: str
    model_path: Path | None = None
    data_root: Path | None = None

    @property
    def resolved_model_path(self) -> Path:
        return self.model_path or self.assets_dir / "models" / self.model

    @property
    def resolved_data_root(self) -> Path:
        return self.data_root or self.assets_dir / "data" / self.model


def require_existing(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} was not found at {path}. Download or prepare it first, "
            "or pass an explicit path. See SAFE-SVD-v0/docs/models.md and "
            "SAFE-SVD-v0/docs/datasets.md."
        )
