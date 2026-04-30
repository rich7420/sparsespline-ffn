"""B1Lookup custom autograd Function.

Wraps the B1 spline lookup so PyTorch's autograd never builds an indexing
backward graph through ``Q[bin_idx]``.  All gradient flow is owned by
``backward()`` which calls the Triton dQ kernel and computes dt manually.

This is the integration point of the Triton kernel into FullMixTuckerFFN.

Math contract:
  forward : beta[n,j,c] = (1-t[n,j])*Q[bin[n,j], c] + t[n,j]*Q[bin[n,j]+1, c]
  backward:
    dQ[bin,   c]  += (1-t)*dbeta[:,:,c]      (Triton kernel)
    dQ[bin+1, c]  += t    *dbeta[:,:,c]      (Triton kernel)
    dt[n,j]       = sum_c (Q[bin+1,c] - Q[bin,c]) * dbeta[n,j,c]
    dbin          = None  (integer, no gradient)
"""
from __future__ import annotations

import torch

from sparsespline_ffn.kernels.triton_b1 import b1_backward_dq


class B1Lookup(torch.autograd.Function):
    """Custom autograd primitive for B1 spline lookup."""

    @staticmethod
    def forward(ctx, Q: torch.Tensor, bin_idx: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """Q: (L, R_b)  bin_idx: (..., m) int64  t: (..., m) float

        Returns beta: (..., m, R_b) in t's dtype.
        """
        # Read Q with .detach() so PyTorch never wires an indexing-backward
        # graph through these reads.  We drive Q's gradient ourselves in
        # backward() via the Triton kernel.
        Q_detached = Q.detach()
        Q0 = Q_detached.index_select(0, bin_idx.reshape(-1))
        Q1 = Q_detached.index_select(0, (bin_idx + 1).reshape(-1))
        # Restore (..., m, R_b) shape
        out_shape = (*bin_idx.shape, Q.shape[-1])
        Q0 = Q0.view(out_shape)
        Q1 = Q1.view(out_shape)
        beta = torch.lerp(Q0, Q1, t.unsqueeze(-1))

        # Save for backward.  We save Q0, Q1 (not Q) because dt only needs
        # the difference, which is shape-matching the output.  This avoids
        # holding a reference to Q (which would block in-place updates).
        ctx.save_for_backward(bin_idx, t, Q0, Q1)
        ctx.Q_shape = tuple(Q.shape)
        return beta

    @staticmethod
    def backward(ctx, dbeta: torch.Tensor):
        bin_idx, t, Q0, Q1 = ctx.saved_tensors
        L, R_b = ctx.Q_shape

        # dQ via Triton kernel.  Always returns fp32; cast back if Q dtype
        # differs.  Keeping fp32 grads is fine — Adam will mix anyway.
        dQ_fp32 = b1_backward_dq(bin_idx, t, dbeta, L=L)

        # dt = sum_c (Q1 - Q0) * dbeta   in fp32 for precision; cast back.
        # Cast to fp32 inside this op so bf16 inputs do not lose accuracy.
        dt = ((Q1.float() - Q0.float()) * dbeta.float()).sum(dim=-1).to(t.dtype)

        # Cast dQ to Q's dtype if needed.  Q dtype matches Q0 dtype (we built
        # Q0 from Q.detach().index_select).  Adam state is fp32 internally
        # via promotion; nanochat's optimizer handles this transparently.
        dQ = dQ_fp32.to(Q0.dtype)

        return dQ, None, dt


def b1_lookup(Q: torch.Tensor, bin_idx: torch.Tensor,
              t: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper: ``B1Lookup.apply(Q, bin_idx, t)``."""
    return B1Lookup.apply(Q, bin_idx, t)


__all__ = ["B1Lookup", "b1_lookup"]
