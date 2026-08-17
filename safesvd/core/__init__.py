"""Backend-neutral building blocks for SAFE-SVD compression."""

from .compression import CompressionCallbacks, compress_clean_input, compress_sequentially
from .factorization import LowRankFactors, clean_input_factors, combined_input_factors, make_low_rank_linear
from .ranks import allocate_greedy_ranks
from .whitening import fisher_cholesky_whitener, svd_whitener

__all__ = [
    "CompressionCallbacks",
    "LowRankFactors",
    "allocate_greedy_ranks",
    "clean_input_factors",
    "compress_clean_input",
    "combined_input_factors",
    "compress_sequentially",
    "fisher_cholesky_whitener",
    "make_low_rank_linear",
    "svd_whitener",
]
