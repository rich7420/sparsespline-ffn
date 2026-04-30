"""Optional Triton kernels for SparseSpline-FFN.

The form-B PyTorch reference (fullmix_tucker.py) is the permanent oracle.
These kernels exist solely to close the dQ scatter bottleneck identified by
benchmarks/profile_backward.py — autograd through Q[bin_idx] decomposes into
``aten::_index_put_impl_`` which owns ~97% of the form-B backward cost.

Public surface:
  b1_backward_dq(bin_idx, t, dbeta, *, L) -> dQ      # the Triton kernel wrapper
  HAS_TRITON: bool                                   # cheap availability check
"""
from __future__ import annotations

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:  # pragma: no cover - exercised on CPU-only installs
    HAS_TRITON = False

if HAS_TRITON:
    from .triton_b1 import b1_backward_dq

    __all__ = ["HAS_TRITON", "b1_backward_dq"]
else:  # pragma: no cover
    __all__ = ["HAS_TRITON"]
