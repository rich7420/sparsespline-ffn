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
import os

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
        fwd_kernel: str = "auto",      # "auto" | "triton" | "wgmma_cuda"
    ) -> torch.Tensor:                # [N, h+r]
        # Detached inputs for the kernel/reference forward
        z_d = z.detach()
        C_d = C.detach()

        # ---- Forward dispatch -----------------------------------------------
        # auto: native CUDA fwd when CUDA + bf16 + r==32 + supported activation;
        #       otherwise fall back to Triton (which itself falls back to
        #       PyTorch reference for B1).
        # Override via env: SPARSE_SPLINE_FWD_KERNEL = auto|triton|wgmma_cuda
        env_fwd = os.environ.get("SPARSE_SPLINE_FWD_KERNEL", "")
        chosen_fwd = env_fwd or fwd_kernel
        cuda_fwd_eligible = (
            use_kernel
            and z.is_cuda
            and z.dtype == torch.bfloat16
            and C.dtype == torch.bfloat16
            and activation in {"relu_sq", "identity"}
            and z.shape[-1] == C.shape[0]
            and C.shape[-1] in (32, 64)           # 3.A.4: r in {32, 64}
        )

        if chosen_fwd in ("auto", "wgmma_cuda") and cuda_fwd_eligible:
            from sparsespline_ffn.cuda_ext import spline_kv_fwd_cuda as _fwd
            f = _fwd(z_d, C_d, float(grid_lo), float(grid_hi), int(G),
                     activation=activation, lambda_scale=float(lambda_scale))
        elif chosen_fwd == "wgmma_cuda" and not cuda_fwd_eligible:
            # Explicit user request couldn't be honored — surface the reason
            # rather than silently falling back to Triton.
            raise RuntimeError(
                f"fwd_kernel='wgmma_cuda' but inputs are not eligible: "
                f"z.dtype={z.dtype}, C.dtype={C.dtype}, "
                f"activation={activation}, "
                f"z.shape={tuple(z.shape)}, C.shape={tuple(C.shape)}. "
                f"Set fwd_kernel='auto' to allow Triton fallback."
            )
        elif use_kernel and z.is_cuda:
            from sparsespline_ffn.kernels.triton_flash_spline_feature import (
                flash_spline_feature_forward as kernel_fwd,
            )
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

        def _capture_safe_dC_return(dC: torch.Tensor) -> torch.Tensor | None:
            """Route C gradients through preallocated C.grad during graph capture.

            PyTorch's CUDA Graph replay does not re-run this Python autograd
            Function.  Returning a freshly allocated dC tensor to AccumulateGrad
            is correct in eager mode, but the dC -> C.grad edge has proven
            unreliable after graph replay for this custom backward.  During
            capture, CudaGraphTrainStep has already preallocated C.grad, so we
            capture an explicit in-place add into that buffer and suppress the
            normal returned C gradient to avoid double accumulation.
            """
            if (
                C.requires_grad
                and C.grad is not None
                and torch.cuda.is_current_stream_capturing()
            ):
                C.grad.add_(dC.to(C.grad.dtype))
                return None
            return dC if C.requires_grad else None

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
                # 3.A.3: fused_post is ON by default — saves the Python-side
                # fp32→bf16 cast + dz_base adder (one less round-trip through
                # autograd, ~37 MB transient peak savings).  Disable via
                # SPARSE_SPLINE_FUSED_WGMMA_POST=0 if needed for debugging.
                use_fused_wgmma = (
                    os.environ.get("SPARSE_SPLINE_FUSED_WGMMA_POST", "1") != "0"
                )
                if (use_fused_wgmma
                        and z.dtype == torch.bfloat16 and C.dtype == torch.bfloat16
                        and ctx.activation in {"relu_sq", "identity"}):
                    dC, dz = _cuda_bwd(
                        z.detach(), C.detach(), g_delta,
                        ctx.grid_lo, ctx.grid_hi, ctx.G,
                        g_a=g_a, activation=ctx.activation, fused_post=True,
                    )
                    return (dz if z.requires_grad else None,
                            _capture_safe_dC_return(dC),
                            None, None, None, None, None, None, None, None)
                else:
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
                    _capture_safe_dC_return(dC),
                    None, None, None, None, None, None, None, None)

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
        return dz, dC, None, None, None, None, None, None, None, None


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
    fwd_kernel: str = "auto",
) -> torch.Tensor:
    """Convenience functional wrapper around the autograd Function.

    fwd_kernel:
      "auto"       — use native CUDA wgmma fwd when eligible, else Triton
      "triton"     — force Triton fwd
      "wgmma_cuda" — force native CUDA fwd (errors if not eligible)
    Override at runtime via env var SPARSE_SPLINE_FWD_KERNEL.
    """
    return FlashSplineFeature.apply(
        z, C, grid_lo, grid_hi, G, activation, lambda_scale, use_kernel,
        bwd_kernel, fwd_kernel,
    )


class FlashSplineDelta(torch.autograd.Function):
    """Pure-spline forward (no ReLU² base path) for Plan A no_base cells.

    Forward returns delta = lambda * spline(z, C)  shape [N, r] bf16.
    No activation is computed; no f tensor is materialized.

    Backward: only the spline gradient path runs (no dz_base contribution).
    """

    @staticmethod
    def forward(
        ctx,
        z: torch.Tensor,             # [N, h] bf16
        C: torch.Tensor,             # [h, L, r] bf16
        grid_lo: float,
        grid_hi: float,
        G: int,
        lambda_scale: float,
        bwd_kernel: str,             # "triton" | "hopper_cuda" | "wgmma_cuda"
    ) -> torch.Tensor:                # [N, r] bf16  (= lambda * delta)
        z_d = z.detach()
        C_d = C.detach()
        if z.is_cuda:
            # Round 1 reverted: Triton pack kernel had different rounding
            # behavior than PyTorch's .to(bf16) (val degraded ~0.04 nat).
            # Stick with PyTorch ops — the launch savings weren't worth the
            # slight numerical drift.
            from sparsespline_ffn.kernels.triton_flash_spline_feature import (
                flash_spline_delta_forward_v4 as _triton_delta,
            )
            delta_fp32 = _triton_delta(z_d, C_d, float(grid_lo), float(grid_hi), int(G))
            delta = (delta_fp32 * float(lambda_scale)).to(z.dtype)
        else:
            from sparsespline_ffn.rl_spline_kv_reference import (
                flash_spline_feature_reference as ref_fwd,
            )
            f = ref_fwd(z_d, C_d, grid_lo, grid_hi, G,
                        activation="identity", lambda_scale=1.0)
            delta_fp32 = f[:, z.shape[-1]:].float()
            delta = (delta_fp32 * float(lambda_scale)).to(z.dtype)

        ctx.save_for_backward(z, C)
        ctx.grid_lo = float(grid_lo)
        ctx.grid_hi = float(grid_hi)
        ctx.G = int(G)
        ctx.lambda_scale = float(lambda_scale)
        ctx.bwd_kernel = bwd_kernel
        return delta

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, C = ctx.saved_tensors
        # grad_output is dL/d(lambda*delta).  Apply lambda chain rule for the
        # spline backward kernel (which expects dL/d(unscaled delta * lambda)).
        g_delta = (grad_output * ctx.lambda_scale).contiguous()

        def _capture_safe_dC_return(dC: torch.Tensor) -> torch.Tensor | None:
            if (
                C.requires_grad
                and C.grad is not None
                and torch.cuda.is_current_stream_capturing()
            ):
                C.grad.add_(dC.to(C.grad.dtype))
                return None
            return dC if C.requires_grad else None

        if z.is_cuda:
            bk = getattr(ctx, "bwd_kernel", "triton")
            if bk == "wgmma_cuda":
                # Plan A Fix 3 — route through fused_post to eliminate
                # the 2 extra Python `.to(bf16)` cast launches.
                #   activation="identity" → phi'(z) = 1
                #   g_a = zeros          → dz_base = g_a * 1 = 0
                #   dz_out = bf16(dz_base + dz_spline) = bf16(dz_spline)
                # Net effect identical to the cast-after-the-fact path.
                from sparsespline_ffn.cuda_ext import spline_kv_bwd_wgmma_cuda as _cuda_bwd
                if (z.dtype == torch.bfloat16 and C.dtype == torch.bfloat16):
                    g_a_zeros = torch.zeros_like(z)
                    dC, dz = _cuda_bwd(
                        z.detach(), C.detach(), g_delta,
                        ctx.grid_lo, ctx.grid_hi, ctx.G,
                        g_a=g_a_zeros, activation="identity", fused_post=True,
                    )
                    return (dz if z.requires_grad else None,
                            _capture_safe_dC_return(dC),
                            None, None, None, None, None)
                else:
                    dC, dz_spline = _cuda_bwd(
                        z.detach(), C.detach(), g_delta,
                        ctx.grid_lo, ctx.grid_hi, ctx.G,
                    )
            elif bk == "hopper_cuda":
                from sparsespline_ffn.cuda_ext import spline_kv_bwd_hopper_cuda as _cuda_bwd
                dC, dz_spline = _cuda_bwd(
                    z.detach(), C.detach(), g_delta,
                    ctx.grid_lo, ctx.grid_hi, ctx.G,
                )
            else:
                from sparsespline_ffn.kernels.triton_flash_spline_feature import (
                    flash_spline_delta_backward_v3 as _triton_bwd,
                )
                dC, dz_spline = _triton_bwd(
                    z.detach(), C.detach(), g_delta,
                    ctx.grid_lo, ctx.grid_hi, ctx.G,
                )
            # No base-path contribution to dz (no_base = no activation gradient)
            dz = dz_spline.to(z.dtype)
            dC = dC.to(C.dtype)
            return (dz if z.requires_grad else None,
                    _capture_safe_dC_return(dC),
                    None, None, None, None, None)

        # CPU fallback via reference + autograd
        from sparsespline_ffn.rl_spline_kv_reference import (
            flash_spline_feature_reference as ref_fwd,
        )
        z_t = z.detach().requires_grad_(z.requires_grad)
        C_t = C.detach().requires_grad_(C.requires_grad)
        h = z.shape[-1]
        with torch.enable_grad():
            f = ref_fwd(
                z_t, C_t, ctx.grid_lo, ctx.grid_hi, ctx.G,
                activation="identity", lambda_scale=ctx.lambda_scale,
            )
            delta_only = f[:, h:]
            grads = torch.autograd.grad(
                delta_only,
                [t for t, need in [(z_t, z.requires_grad),
                                     (C_t, C.requires_grad)] if need],
                grad_outputs=grad_output, allow_unused=True,
            )
        gi = iter(grads)
        dz = next(gi) if z.requires_grad else None
        dC = next(gi) if C.requires_grad else None
        return dz, dC, None, None, None, None, None


def flash_spline_delta(
    z: torch.Tensor,
    C: torch.Tensor,
    grid_lo: float = -3.0,
    grid_hi: float = 3.0,
    G: int = 20,
    lambda_scale: float = 1.0,
    bwd_kernel: str = "wgmma_cuda",
) -> torch.Tensor:
    """Pure-spline forward (Plan A): returns lambda * delta(z, C)  [N, r]."""
    return FlashSplineDelta.apply(
        z, C, grid_lo, grid_hi, G, lambda_scale, bwd_kernel,
    )


__all__ = [
    "FlashSplineFeature", "flash_spline_feature",
    "FlashSplineDelta", "flash_spline_delta",
]
