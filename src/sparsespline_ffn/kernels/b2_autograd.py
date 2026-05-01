"""B2 spline activation autograd Function.

Wraps the Triton B2 forward / backward kernels so that the per-channel
spline activation participates in PyTorch's autograd without falling back
to slow indexing-backward paths.
"""
from __future__ import annotations

import torch

from sparsespline_ffn.kernels.triton_b2 import b2_backward_dq_dz, b2_forward


class B2SplineActivation(torch.autograd.Function):
    """Per-channel quadratic B-spline activation.

    Forward:  y = spline(z, Q)  where Q is (H, L=G+2)
    Backward: dQ via Triton scatter-add; dz via direct kernel computation.

    z and dy participate in autograd; Q is treated as a parameter (gradient
    accumulated into Q.grad).  grid_lo/grid_hi/G are non-tensor metadata.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, Q: torch.Tensor,
                grid_lo: float, grid_hi: float, G: int) -> torch.Tensor:
        Q_detached = Q.detach()
        y = b2_forward(z, Q_detached, grid_lo, grid_hi, G)
        ctx.save_for_backward(z, Q_detached)
        ctx.grid_lo = float(grid_lo)
        ctx.grid_hi = float(grid_hi)
        ctx.G = int(G)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        z, Q = ctx.saved_tensors
        dQ_fp32, dz = b2_backward_dq_dz(
            z, Q, dy, ctx.grid_lo, ctx.grid_hi, ctx.G,
        )
        return dz, dQ_fp32.to(Q.dtype), None, None, None


def b2_spline_activation(
    z: torch.Tensor, Q: torch.Tensor,
    grid_lo: float = -3.0, grid_hi: float = 3.0, G: int = 20,
) -> torch.Tensor:
    """Convenience wrapper around ``B2SplineActivation.apply``."""
    return B2SplineActivation.apply(z, Q, grid_lo, grid_hi, G)


__all__ = ["B2SplineActivation", "b2_spline_activation"]
