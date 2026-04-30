"""B1Lookup custom autograd primitive.

Wraps the B1 spline lookup so PyTorch's autograd never builds an indexing
backward graph through ``Q[bin_idx]``.  The forward uses Triton ``b1_forward``
to do the gather + lerp in one kernel; the backward uses Triton
``b1_backward_dq_dt`` to produce both ``dQ`` (atomic scatter-add) and ``dt``
(per-token sum) in a single kernel.

Math contract:
    forward : beta[n,j,c] = (1-t[n,j])*Q[bin[n,j], c] + t[n,j]*Q[bin[n,j]+1, c]
    backward:
        dQ[bin,   c]  += (1-t)*dbeta[:,:,c]      (Triton, fp32 accumulation)
        dQ[bin+1, c]  += t    *dbeta[:,:,c]
        dt[n,j]       = sum_c (Q[bin+1,c] - Q[bin,c]) * dbeta[n,j,c]
        dbin          = None  (integer, no gradient)

Activation memory: only (bin_idx, t, Q-reference) are saved -- not Q0/Q1.
At nanochat scale this saves ~384 MB per layer in bf16.
"""
from __future__ import annotations

import torch

from sparsespline_ffn.kernels.triton_b1 import b1_backward_dq_dt, b1_forward


class B1Lookup(torch.autograd.Function):
    """Custom autograd primitive for B1 spline lookup."""

    @staticmethod
    def forward(ctx, Q: torch.Tensor, bin_idx: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """Q: (L, R_b)  bin_idx: (..., m) int64  t: (..., m) float."""
        # PyTorch tensor versioning ensures Q is not mutated between fwd
        # and bwd of a single training iter, so saving by reference is safe.
        Q_detached = Q.detach()
        beta = b1_forward(Q_detached, bin_idx, t)
        ctx.save_for_backward(Q_detached, bin_idx, t)
        return beta

    @staticmethod
    def backward(ctx, dbeta: torch.Tensor):
        Q, bin_idx, t = ctx.saved_tensors
        dQ_fp32, dt_fp32 = b1_backward_dq_dt(Q, bin_idx, t, dbeta)
        return dQ_fp32.to(Q.dtype), None, dt_fp32.to(t.dtype)


def b1_lookup(Q: torch.Tensor, bin_idx: torch.Tensor,
              t: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper: ``B1Lookup.apply(Q, bin_idx, t)``."""
    return B1Lookup.apply(Q, bin_idx, t)


__all__ = ["B1Lookup", "b1_lookup"]
