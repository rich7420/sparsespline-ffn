"""SparseSpline-FFN public API.

The package keeps the slow, auditable PyTorch reference implementation as the
default path.  Fused kernels can be added behind the same API after the quality
gate passes.
"""
from __future__ import annotations

import torch

from sparsespline_ffn.fullmix_tucker import FullMixTuckerConfig, FullMixTuckerFFN
from sparsespline_ffn.schedules import (
    MLPFFN,
    build_ffn,
    build_fullmix_tucker_ffn,
    should_replace_layer,
)
from sparsespline_ffn.tucker_init import hosvd_warmstart_from_dense

__version__ = "0.1.0"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

__all__ = [
    "FullMixTuckerConfig",
    "FullMixTuckerFFN",
    "MLPFFN",
    "__version__",
    "build_ffn",
    "build_fullmix_tucker_ffn",
    "hosvd_warmstart_from_dense",
    "should_replace_layer",
]
