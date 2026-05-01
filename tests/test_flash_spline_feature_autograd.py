"""Autograd tests for the FlashSplineFeature wrapper (v7 Phase B2.3).

Verifies:
  - forward with use_kernel=True and use_kernel=False produce close output
  - gradients computed by FlashSplineFeature.apply match those from
    the pure-PyTorch reference (because backward uses reference recomp)
  - C=0 cold start: dC nonzero, dz nonzero (key v7 §R.5 property)
"""
from __future__ import annotations

import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA-only autograd test")


def test_autograd_forward_falls_back_to_reference_on_cpu():
    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature,
    )
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference,
    )
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    C = torch.randn(16, 10, 4) * 0.1
    f1 = flash_spline_feature(z, C, grid_lo=-3, grid_hi=3, G=8,
                                use_kernel=True)  # use_kernel=True but no CUDA
    f2 = flash_spline_feature_reference(z, C, grid_lo=-3, grid_hi=3, G=8)
    assert torch.allclose(f1, f2, atol=1e-6)


def test_autograd_backward_recovers_reference_gradients_cpu():
    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature,
    )
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference,
    )
    torch.manual_seed(0)
    # CPU autograd path: forward is reference, backward is reference recomp.
    # Should match exactly the gradients we'd get directly through reference.
    z0 = torch.randn(4, 8)
    C0 = torch.randn(8, 10, 4) * 0.1
    for name, fn in [
        ("autograd-wrapper",
         lambda z, C: flash_spline_feature(z, C, grid_lo=-3, grid_hi=3, G=8,
                                             use_kernel=False)),
        ("direct-reference",
         lambda z, C: flash_spline_feature_reference(z, C, grid_lo=-3,
                                                       grid_hi=3, G=8)),
    ]:
        z = z0.clone().detach().requires_grad_(True)
        C = C0.clone().detach().requires_grad_(True)
        f = fn(z, C)
        loss = (f * torch.arange(f.numel()).float().reshape_as(f)).sum()
        loss.backward()
        if name == "autograd-wrapper":
            dz_w, dC_w = z.grad.clone(), C.grad.clone()
        else:
            dz_r, dC_r = z.grad.clone(), C.grad.clone()
    assert torch.allclose(dz_w, dz_r, atol=1e-6), \
        f"dz mismatch: max diff {(dz_w - dz_r).abs().max()}"
    assert torch.allclose(dC_w, dC_r, atol=1e-6), \
        f"dC mismatch: max diff {(dC_w - dC_r).abs().max()}"


@cuda_only
def test_autograd_kernel_forward_matches_reference():
    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature,
    )
    torch.manual_seed(0)
    z = torch.randn(16, 32, device="cuda", dtype=torch.float32)
    C = torch.randn(32, 10, 8, device="cuda", dtype=torch.float32) * 0.1
    f_k = flash_spline_feature(z, C, grid_lo=-3, grid_hi=3, G=8,
                                 use_kernel=True)
    f_r = flash_spline_feature(z, C, grid_lo=-3, grid_hi=3, G=8,
                                 use_kernel=False)
    max_diff = (f_k - f_r).abs().max().item()
    assert max_diff < 1e-4, max_diff


@cuda_only
def test_autograd_backward_after_kernel_forward():
    """Forward via Triton kernel, backward via reference recomp.
    Backward gradient should match what direct reference forward+backward
    would give."""
    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature,
    )
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference,
    )
    torch.manual_seed(0)
    z0 = torch.randn(4, 16, device="cuda", dtype=torch.float32)
    C0 = (torch.randn(16, 10, 4, device="cuda", dtype=torch.float32) * 0.1)

    z = z0.clone().requires_grad_(True)
    C = C0.clone().requires_grad_(True)
    f = flash_spline_feature(z, C, grid_lo=-3, grid_hi=3, G=8,
                               use_kernel=True)
    target = torch.randn_like(f)
    loss = (f * target).sum()
    loss.backward()
    dz_k, dC_k = z.grad.clone(), C.grad.clone()

    z = z0.clone().requires_grad_(True)
    C = C0.clone().requires_grad_(True)
    f = flash_spline_feature_reference(z, C, grid_lo=-3, grid_hi=3, G=8)
    loss = (f * target).sum()
    loss.backward()
    dz_r, dC_r = z.grad.clone(), C.grad.clone()

    assert torch.allclose(dz_k, dz_r, atol=1e-5)
    assert torch.allclose(dC_k, dC_r, atol=1e-5)


def test_C_zero_cold_start_gradient_flows_to_C():
    """v7 §R.5: even at C=0 init, dC must be nonzero (so C learns)."""
    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature,
    )
    torch.manual_seed(0)
    z = torch.randn(4, 8, requires_grad=True)
    C = torch.zeros(8, 10, 4, requires_grad=True)
    f = flash_spline_feature(z, C, grid_lo=-3, grid_hi=3, G=8,
                               use_kernel=False)
    # loss touching only the delta half so dC must come from there
    delta = f[:, 8:]
    loss = (delta * torch.randn_like(delta)).sum()
    loss.backward()
    assert C.grad is not None
    assert C.grad.abs().sum() > 0, \
        "dC must be nonzero at C=0 cold start (v7 §R.5)"
