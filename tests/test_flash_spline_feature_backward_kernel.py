"""Tests for the B2.4 Triton backward kernel.

Validates ``flash_spline_delta_backward`` against the analytical
reference (``flash_spline_feature_backward_ref.py``), which itself was
validated against autograd in ``test_flash_spline_feature_backward_ref.py``.

So this is the gold→silver→bronze chain:
  gold (autograd / fp64) → silver (analytical fp32) → bronze (kernel fp32/bf16)
"""
from __future__ import annotations

import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA-only kernel test")


@cuda_only
def test_bwd_kernel_matches_reference_fp32_uniform():
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward,
    )
    from sparsespline_ffn.kernels.flash_spline_feature_backward_ref import (
        flash_spline_delta_backward_ref,
    )
    torch.manual_seed(0)
    N, h, r, G = 64, 256, 32, 16
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.float32)
    C = (torch.randn(h, L, r, device="cuda", dtype=torch.float32) * 0.1)
    g = torch.randn(N, r, device="cuda", dtype=torch.float32)

    dC_ref, dz_ref = flash_spline_delta_backward_ref(z, C, g, -3.0, 3.0, G)
    dC_k,   dz_k   = flash_spline_delta_backward(z, C, g, -3.0, 3.0, G)

    dC_diff = (dC_k - dC_ref).abs().max().item()
    dz_diff = (dz_k - dz_ref).abs().max().item()
    rel_dC = ((dC_k - dC_ref).pow(2).mean().sqrt()
              / dC_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    rel_dz = ((dz_k - dz_ref).pow(2).mean().sqrt()
              / dz_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    # fp32 atomic-add reduction non-determinism + summation order
    assert rel_dC < 1e-4, f"dC rel rms = {rel_dC:.3e}, max abs = {dC_diff:.3e}"
    assert rel_dz < 1e-4, f"dz rel rms = {rel_dz:.3e}, max abs = {dz_diff:.3e}"


@cuda_only
def test_bwd_kernel_matches_reference_realistic_shape():
    """h=768, r=64, L=22 — production shape."""
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward,
    )
    from sparsespline_ffn.kernels.flash_spline_feature_backward_ref import (
        flash_spline_delta_backward_ref,
    )
    torch.manual_seed(0)
    N, h, r, G = 256, 768, 64, 20
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.float32)
    C = (torch.randn(h, L, r, device="cuda", dtype=torch.float32) * 0.1)
    g = torch.randn(N, r, device="cuda", dtype=torch.float32)

    dC_ref, dz_ref = flash_spline_delta_backward_ref(z, C, g, -3.0, 3.0, G)
    dC_k,   dz_k   = flash_spline_delta_backward(z, C, g, -3.0, 3.0, G)

    rel_dC = ((dC_k - dC_ref).pow(2).mean().sqrt()
              / dC_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    rel_dz = ((dz_k - dz_ref).pow(2).mean().sqrt()
              / dz_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    # production-scale atomic contention amplifies fp32 noise slightly
    assert rel_dC < 5e-4, f"dC rel rms = {rel_dC:.3e}"
    assert rel_dz < 5e-4, f"dz rel rms = {rel_dz:.3e}"


@cuda_only
def test_bwd_kernel_C_zero_dC_nonzero_dz_zero():
    """v7 §R.5: at C=0, dC must be nonzero (so C learns) but dz_spline=0."""
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward,
    )
    torch.manual_seed(0)
    N, h, r, G = 32, 128, 16, 12
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.float32)
    C = torch.zeros(h, L, r, device="cuda", dtype=torch.float32)
    g = torch.randn(N, r, device="cuda", dtype=torch.float32)
    dC, dz = flash_spline_delta_backward(z, C, g, -3.0, 3.0, G)
    assert dC.abs().sum().item() > 0, "dC must be nonzero at C=0 cold start"
    # dz comes from inner = sum_c C[*]*g, with C=0 → dz=0
    assert dz.abs().max().item() == 0.0, "dz must be 0 at C=0"


@cuda_only
def test_bwd_kernel_g_delta_zero_gives_zero_grads():
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward,
    )
    torch.manual_seed(0)
    N, h, r, G = 32, 128, 16, 12
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.float32)
    C = (torch.randn(h, L, r, device="cuda", dtype=torch.float32) * 0.1)
    g = torch.zeros(N, r, device="cuda", dtype=torch.float32)
    dC, dz = flash_spline_delta_backward(z, C, g, -3.0, 3.0, G)
    assert dC.abs().max().item() == 0.0
    assert dz.abs().max().item() == 0.0


@cuda_only
def test_bwd_kernel_collapsed_bins_high_contention():
    """Adversarial: nearly all tokens collapse into 1 bin → maximum atomic
    contention.  Per v7 §R.6.9 kernel must handle this without producing
    incorrect gradients."""
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward,
    )
    from sparsespline_ffn.kernels.flash_spline_feature_backward_ref import (
        flash_spline_delta_backward_ref,
    )
    torch.manual_seed(0)
    N, h, r, G = 128, 256, 32, 16
    L = G + 2
    # All z near 0 → all hit the same bin
    z = (torch.randn(N, h, device="cuda", dtype=torch.float32) * 0.05)
    C = (torch.randn(h, L, r, device="cuda", dtype=torch.float32) * 0.1)
    g = torch.randn(N, r, device="cuda", dtype=torch.float32)

    dC_ref, dz_ref = flash_spline_delta_backward_ref(z, C, g, -3.0, 3.0, G)
    dC_k,   dz_k   = flash_spline_delta_backward(z, C, g, -3.0, 3.0, G)
    rel_dC = ((dC_k - dC_ref).pow(2).mean().sqrt()
              / dC_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    rel_dz = ((dz_k - dz_ref).pow(2).mean().sqrt()
              / dz_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    # Higher tolerance under heavy atomic contention (more fp32 round-off)
    assert rel_dC < 1e-3, f"collapsed dC rel rms = {rel_dC:.3e}"
    assert rel_dz < 1e-3, f"collapsed dz rel rms = {rel_dz:.3e}"


@cuda_only
def test_bwd_kernel_skewed_in_range():
    """Skewed z near grid_hi triggers the clamp-gradient mask (u > G-1).
    Tests that dz uses the right mask vs autograd's clamp behavior."""
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward,
    )
    from sparsespline_ffn.kernels.flash_spline_feature_backward_ref import (
        flash_spline_delta_backward_ref,
    )
    torch.manual_seed(0)
    N, h, r, G = 64, 256, 32, 16
    L = G + 2
    z = (torch.randn(N, h, device="cuda", dtype=torch.float32) * 0.5 + 2.0)
    C = (torch.randn(h, L, r, device="cuda", dtype=torch.float32) * 0.1)
    g = torch.randn(N, r, device="cuda", dtype=torch.float32)

    dC_ref, dz_ref = flash_spline_delta_backward_ref(z, C, g, -3.0, 3.0, G)
    dC_k,   dz_k   = flash_spline_delta_backward(z, C, g, -3.0, 3.0, G)
    rel_dC = ((dC_k - dC_ref).pow(2).mean().sqrt()
              / dC_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    rel_dz = ((dz_k - dz_ref).pow(2).mean().sqrt()
              / dz_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    assert rel_dC < 5e-4, f"skewed dC rel rms = {rel_dC:.3e}"
    assert rel_dz < 5e-4, f"skewed dz rel rms = {rel_dz:.3e}"
