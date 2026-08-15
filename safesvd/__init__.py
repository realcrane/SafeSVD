"""SAFE-SVD public API."""

__version__ = "0.0.0"

from .config import SafeSVDConfig
from .pipeline import CompressionResult, compress

__all__ = ["CompressionResult", "SafeSVDConfig", "compress"]
