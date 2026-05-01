"""SparseSpline-FFN public API.

The package keeps the slow, auditable PyTorch reference implementation as the
default path.  Fused kernels can be added behind the same API after the quality
gate passes.
"""
from __future__ import annotations

import torch

from sparsespline_ffn.fullmix_tucker import FullMixTuckerConfig, FullMixTuckerFFN
from sparsespline_ffn.kernels import HAS_TRITON
from sparsespline_ffn.schedules import (
    MLPFFN,
    build_ffn,
    build_fullmix_tucker_ffn,
    should_replace_layer,
)
from sparsespline_ffn.tucker_init import hosvd_warmstart_from_dense

if HAS_TRITON:
    from sparsespline_ffn.kernels import B1Lookup, b1_lookup  # noqa: F401

__version__ = "0.1.0"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def is_kernel_available() -> bool:
    """Return True iff the optional Triton kernel can run on this machine.

    Both conditions must hold:

      - PyTorch reports CUDA available (``torch.cuda.is_available()``);
      - the ``triton`` package is importable (``HAS_TRITON``).

    Use this at the top of training scripts to decide whether to pass
    ``use_kernel=True`` to ``build_ffn`` / ``FullMixTuckerConfig``.  When
    this returns False, the form-B reference path is the only option and
    the call is identical except slower.

    ``HAS_TRITON`` reports just the import; ``is_kernel_available`` adds
    the CUDA check, which is the actual runtime requirement.
    """
    return bool(HAS_TRITON and torch.cuda.is_available())


__all__ = [
    "FullMixTuckerConfig",
    "FullMixTuckerFFN",
    "HAS_TRITON",
    "MLPFFN",
    "__version__",
    "build_ffn",
    "build_fullmix_tucker_ffn",
    "hosvd_warmstart_from_dense",
    "is_kernel_available",
    "should_replace_layer",
]
if HAS_TRITON:
    __all__ += ["B1Lookup", "b1_lookup"]
