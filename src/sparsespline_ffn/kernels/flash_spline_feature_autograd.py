"""FlashSplineFeature autograd skeleton (v7 Phase B2.3).

Until the backward Triton kernel (B2.4) lands, the autograd path uses
**reference recomputation** for backward — slow but provably correct,
exactly the gradient the PyTorch reference computes.

Forward path:
  use_kernel=True (and CUDA): Triton kernel from triton_flash_spline_feature.py
  otherwise:                  PyTorch reference from rl_spline_kv_reference.py

Backward path (always):
  rerun the PyTorch reference forward with autograd traced, then
  torch.autograd.grad through it.  This is the "checkpointed forward"
  pattern.

Once B2.4 ships, ``backward`` will switch to the Triton bwd kernel and
this autograd Function becomes a pure pass-through.

Note that we do NOT integrate this into a nanochat adapter yet — that
is gated on the kernel performance microbench (Task B2.2).
"""
from __future__ import annotations

from typing import Literal

import torch


class FlashSplineFeature(torch.autograd.Function):
    """Autograd-aware wrapper for the FlashSplineFeature forward.

    Parameters mirror the kernel/reference signature.
    """

    @staticmethod
    def forward(
        ctx,
        z: torch.Tensor,             # [N, h]
        C: torch.Tensor,             # [h, L, r]
        grid_lo: float,
        grid_hi: float,
        G: int,
        activation: str,
        lambda_scale: float,
        use_kernel: bool,
        bwd_kernel: str = "triton",   # "triton" | "hopper_cuda" | "wgmma_cuda"
    ) -> torch.Tensor:                # [N, h+r]
        # Detached inputs for the kernel/reference forward
        z_d = z.detach()
        C_d = C.detach()

        if use_kernel and z.is_cuda:
            from sparsespline_ffn.kernels.triton_flash_spline_feature import (
                flash_spline_feature_forward as kernel_fwd,
            )
            # v4 (h-split + atomic_add) is the fastest variant on H100/3080.
            f = kernel_fwd(z_d, C_d, grid_lo, grid_hi, G,
                           activation=activation, lambda_scale=lambda_scale,
                           version="v4")
        else:
            from sparsespline_ffn.rl_spline_kv_reference import (
                flash_spline_feature_reference as ref_fwd,
            )
            f = ref_fwd(z_d, C_d, grid_lo, grid_hi, G,
                        activation=activation, lambda_scale=lambda_scale)

        # Save tensors / hyperparams for backward
        ctx.save_for_backward(z, C)
        ctx.grid_lo = float(grid_lo)
        ctx.grid_hi = float(grid_hi)
        ctx.G = int(G)
        ctx.activation = activation
        ctx.lambda_scale = float(lambda_scale)
        ctx.bwd_kernel = bwd_kernel
        return f

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, C = ctx.saved_tensors
        h = z.shape[-1]
        r = C.shape[-1]
        # Split grad_output into (g_a, g_delta_pre).  g_delta_pre is the
        # gradient w.r.t. the lambda*delta concatenated half; g_delta = lambda * g_delta_pre
        # (chain rule through the lambda scale, v7 §R.3.1).
        g_a = grad_output[..., :h].contiguous()
        g_delta = (grad_output[..., h:] * ctx.lambda_scale).contiguous()

        if z.is_cuda:
            # Backward kernel selection (default: Triton v3).
            bk = getattr(ctx, "bwd_kernel", "triton")
            if bk == "hopper_cuda":
                from sparsespline_ffn.cuda_ext import spline_kv_bwd_hopper_cuda as _cuda_bwd
                dC, dz_spline = _cuda_bwd(
                    z.detach(), C.detach(), g_delta,
                    ctx.grid_lo, ctx.grid_hi, ctx.G,
                )
            elif bk == "wgmma_cuda":
                from sparsespline_ffn.cuda_ext import spline_kv_bwd_wgmma_cuda as _cuda_bwd
                dC, dz_spline = _cuda_bwd(
                    z.detach(), C.detach(), g_delta,
                    ctx.grid_lo, ctx.grid_hi, ctx.G,
                )
            else:
                # Default Triton v3 backward (tl.dot → wgmma codegen on Hopper)
                from sparsespline_ffn.kernels.triton_flash_spline_feature import (
                    flash_spline_delta_backward_v3 as kernel_bwd,
                )
                dC, dz_spline = kernel_bwd(
                    z.detach(), C.detach(), g_delta,
                    ctx.grid_lo, ctx.grid_hi, ctx.G,
                )
            # Add base-path contribution: dz_a = g_a * phi'(z)
            if ctx.activation == "relu_sq":
                phi_prime = (2.0 * z) * (z > 0).to(z.dtype)
            elif ctx.activation == "gelu":
                # PyTorch's GELU has known autograd; use functional grad
                z_t = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    a = torch.nn.functional.gelu(z_t)
                    a.sum().backward()
                phi_prime = z_t.grad
            elif ctx.activation == "identity":
                phi_prime = torch.ones_like(z)
            else:
                phi_prime = torch.ones_like(z)
            dz_base = g_a * phi_prime
            dz = (dz_base + dz_spline).to(z.dtype)
            dC = dC.to(C.dtype)
            return (dz if z.requires_grad else None,
                    dC if C.requires_grad else None,
                    None, None, None, None, None, None, None)

        # ----- CPU fallback: reference recomputation via autograd -----
        from sparsespline_ffn.rl_spline_kv_reference import (
            flash_spline_feature_reference as ref_fwd,
        )
        z_t = z.detach().requires_grad_(z.requires_grad)
        C_t = C.detach().requires_grad_(C.requires_grad)
        with torch.enable_grad():
            f = ref_fwd(
                z_t, C_t, ctx.grid_lo, ctx.grid_hi, ctx.G,
                activation=ctx.activation, lambda_scale=ctx.lambda_scale,
            )
            grads = torch.autograd.grad(
                f, [t for t, need in [(z_t, z.requires_grad),
                                       (C_t, C.requires_grad)] if need],
                grad_outputs=grad_output, allow_unused=True,
            )
        gi = iter(grads)
        dz = next(gi) if z.requires_grad else None
        dC = next(gi) if C.requires_grad else None
        return dz, dC, None, None, None, None, None, None, None


def flash_spline_feature(
    z: torch.Tensor,
    C: torch.Tensor,
    grid_lo: float = -3.0,
    grid_hi: float = 3.0,
    G: int = 20,
    activation: str = "relu_sq",
    lambda_scale: float = 1.0,
    use_kernel: bool = True,
    bwd_kernel: str = "triton",
) -> torch.Tensor:
    """Convenience functional wrapper around the autograd Function."""
    return FlashSplineFeature.apply(
        z, C, grid_lo, grid_hi, G, activation, lambda_scale, use_kernel,
        bwd_kernel,
    )


__all__ = ["FlashSplineFeature", "flash_spline_feature"]
