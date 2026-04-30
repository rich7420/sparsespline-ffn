"""B1Lookup custom autograd Function.

Wraps the B1 spline lookup so PyTorch's autograd never builds an indexing
backward graph through ``Q[bin_idx]``.  All gradient flow is owned by
``backward()`` which calls the fused Triton dQ+dt kernel.

This is the integration point of the Triton kernels into FullMixTuckerFFN.

Math contract:
  forward : beta[n,j,c] = (1-t[n,j])*Q[bin[n,j], c] + t[n,j]*Q[bin[n,j]+1, c]
  backward:
    dQ[bin,   c]  += (1-t)*dbeta[:,:,c]      (Triton kernel, atomic)
    dQ[bin+1, c]  += t    *dbeta[:,:,c]      (Triton kernel, atomic)
    dt[n,j]       = sum_c (Q[bin+1,c] - Q[bin,c]) * dbeta[n,j,c]   (Triton)
    dbin          = None  (integer, no gradient)

Tier 3 implementation: forward and backward are both Triton kernels.  We do
NOT save Q0/Q1 in autograd state; instead we save Q itself (by reference)
and the bwd kernel re-loads Q[bin], Q[bin+1] from global memory.  This saves
~2 * N * m * R_b bytes of activation memory per layer (~384 MB at nanochat
scale, bf16) and folds the ``((Q1-Q0)*dbeta).sum(-1)`` op chain into the
same launch as dQ.
"""
from __future__ import annotations

import torch

from sparsespline_ffn.kernels.triton_b1 import (
    b1_backward_dq_dt,
    b1_forward,
)


class B1Lookup(torch.autograd.Function):
    """Custom autograd primitive for B1 spline lookup."""

    @staticmethod
    def forward(ctx, Q: torch.Tensor, bin_idx: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """Q: (L, R_b)  bin_idx: (..., m) int64  t: (..., m) float

        Returns beta: (..., m, R_b) in t's dtype.
        """
        # Use detached Q for forward read; backward will use Q again from
        # the saved tensors.  PyTorch's tensor versioning ensures Q is not
        # mutated between fwd and bwd of a single training iter.
        Q_detached = Q.detach()
        beta = b1_forward(Q_detached, bin_idx, t)

        # Tier 3: save Q (not Q0/Q1) to amortize activation memory.
        ctx.save_for_backward(Q_detached, bin_idx, t)
        return beta

    @staticmethod
    def backward(ctx, dbeta: torch.Tensor):
        Q, bin_idx, t = ctx.saved_tensors

        # Single fused kernel: dQ (atomic scatter-add) + dt (per-token sum).
        dQ_fp32, dt_fp32 = b1_backward_dq_dt(Q, bin_idx, t, dbeta)

        # Cast to match parameter dtypes.  Adam will internally promote
        # to fp32 anyway; bf16 grads are fine.
        dQ = dQ_fp32.to(Q.dtype)
        dt = dt_fp32.to(t.dtype)

        return dQ, None, dt


def b1_lookup(Q: torch.Tensor, bin_idx: torch.Tensor,
              t: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper: ``B1Lookup.apply(Q, bin_idx, t)``."""
    return B1Lookup.apply(Q, bin_idx, t)


__all__ = ["B1Lookup", "b1_lookup"]
