"""SAFE-SVD public API."""

__version__ = "0.0.0"

from .config import SafeSVDConfig, SafeSVDEvaluationConfig
from .pipeline import CompressionResult, EvaluationResult, compress, evaluate

__all__ = [
    "CompressionResult",
    "EvaluationResult",
    "SafeSVDConfig",
    "SafeSVDEvaluationConfig",
    "compress",
    "evaluate",
]
