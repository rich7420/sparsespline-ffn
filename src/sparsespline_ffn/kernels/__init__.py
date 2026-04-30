"""Optional Triton kernels for SparseSpline-FFN.

The form-B PyTorch reference (fullmix_tucker.py) is the permanent oracle.
These kernels exist solely to close the dQ scatter bottleneck identified by
benchmarks/profile_backward.py — autograd through Q[bin_idx] decomposes into
``aten::_index_put_impl_`` which owned ~97% of form-B backward cost.

Public surface:
  HAS_TRITON          : True if Triton is importable (else only the form-B
                        path is available; everything still works on CPU).
  B1Lookup            : torch.autograd.Function for the B1 spline lookup.
                        FullMixTuckerFFN(use_kernel=True) uses this.
  b1_lookup           : functional convenience wrapper (B1Lookup.apply).
  b1_forward          : raw Triton fwd kernel (Q, bin_idx, t) -> beta.
  b1_backward_dq      : raw Triton bwd kernel for dQ only -- kept as a
                        narrow oracle that the cross-test in test_kernels.py
                        uses to independently validate b1_backward_dq_dt.
  b1_backward_dq_dt   : production fused bwd kernel: (Q, bin_idx, t, dbeta)
                        -> (dQ, dt).  Used by B1Lookup.backward.
"""
from __future__ import annotations

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:  # pragma: no cover - exercised on CPU-only installs
    HAS_TRITON = False

if HAS_TRITON:
    from .b1_autograd import B1Lookup, b1_lookup
    from .triton_b1 import b1_backward_dq, b1_backward_dq_dt, b1_forward

    __all__ = [
        "B1Lookup",
        "HAS_TRITON",
        "b1_backward_dq",
        "b1_backward_dq_dt",
        "b1_forward",
        "b1_lookup",
    ]
else:  # pragma: no cover
    __all__ = ["HAS_TRITON"]
