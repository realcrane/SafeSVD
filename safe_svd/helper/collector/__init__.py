"""Shared capture contract used by the three model collectors."""

from .collector import (
    CaptureContext,
    CaptureFunction,
    CapturedPairs,
    cache_pair,
    capture_input,
    collect_statistics,
    feature_rows,
    paired_values,
    tree_to,
)

__all__ = [
    "CaptureContext", "CaptureFunction", "CapturedPairs", "cache_pair",
    "capture_input", "collect_statistics", "feature_rows", "paired_values", "tree_to",
]
