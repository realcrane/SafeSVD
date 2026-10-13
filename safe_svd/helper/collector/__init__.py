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
from .input_collector import collect_inputs
from .fisher_collector import collect_fisher

__all__ = [
    "CaptureContext", "CaptureFunction", "CapturedPairs", "cache_pair",
    "capture_input", "collect_statistics", "feature_rows", "paired_values", "tree_to",
    "collect_inputs", "collect_fisher",
]
