"""Public, model-agnostic SAFE-SVD configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SafeSVDConfig:
    """Inputs shared by every SAFE-SVD compression route.

    Adapter-specific options deliberately live in ``extra``.  This keeps the
    public contract stable while each foundation model retains its own loader
    and calibration semantics.
    """

    model: str
    checkpoint: Path
    data_root: Path
    output_dir: Path
    rank_ratio: float
    calibration_size: int
    calibration_seed: int
    source_root: Path | None = None
    device: str = "cuda"
    alpha: float = 0.7
    preset: str = "paper"
    overwrite: bool = False
    extra: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.model not in {"poseidon", "vicon", "mpp"}:
            raise ValueError("model must be one of: poseidon, vicon, mpp")
        if self.preset != "paper":
            raise ValueError("v0 currently exposes only the paper preset")
        if not 0 < self.rank_ratio <= 1:
            raise ValueError("rank_ratio must be in (0, 1]")
        if self.calibration_size <= 0:
            raise ValueError("calibration_size must be positive")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")


@dataclass(frozen=True)
class SafeSVDEvaluationConfig:
    """Public inputs for evaluating an original and compressed SAFE-SVD model."""

    model: str
    checkpoint: Path
    data_root: Path
    compressed_model: Path | None
    output_path: Path
    source_root: Path | None = None
    device: str = "cuda"
    batch_size: int | None = None
    max_batches: int | None = None
    extra: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.model not in {"poseidon", "vicon", "mpp"}:
            raise ValueError("model must be one of: poseidon, vicon, mpp")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("max_batches must be positive")
