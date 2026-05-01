"""Tests for the analytic backward reference (Task 5).

Verifies the explicit dC/dz formulas in
``flash_spline_feature_backward_ref.py`` match autograd through the
PyTorch reference forward.  Includes adversarial collapsed/skewed
distributions per v7 §R.6.9.
"""
from __future__ import annotations

import torch

from sparsespline_ffn.rl_spline_kv_reference import (
    flash_spline_feature_reference,
)
from sparsespline_ffn.kernels.flash_spline_feature_backward_ref import (
    flash_spline_delta_backward_ref,
)


def _autograd_dC_dz(z, C, g_delta, grid_lo, grid_hi, G):
    """Run ref forward + autograd to get reference dC and dz."""
    z_t = z.detach().clone().requires_grad_(True)
    C_t = C.detach().clone().requires_grad_(True)
    f = flash_spline_feature_reference(
        z_t, C_t, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        activation="identity",   # so phi-half does not contribute to dz
        lambda_scale=1.0,
    )
    h = z.shape[1]
    delta = f[:, h:]  # [N, r]
    loss = (delta * g_delta.detach()).sum()
    loss.backward()
    return C_t.grad.detach().clone(), z_t.grad.detach().clone()


def _generate_workload(name: str, N: int, h: int, dtype, device,
                        grid_lo: float = -3.0, grid_hi: float = 3.0):
    if name == "uniform":
        return torch.randn(N, h, dtype=dtype, device=device)
    if name == "skewed":
        return torch.randn(N, h, dtype=dtype, device=device) * 0.5 + 2.0
    if name == "collapsed":
        return torch.randn(N, h, dtype=dtype, device=device) * 0.05
    raise ValueError(name)


def test_dC_uniform_matches_autograd():
    torch.manual_seed(0)
    N, h, r, G = 16, 32, 8, 10
    L = G + 2
    z = _generate_workload("uniform", N, h, torch.float64, "cpu")
    C = torch.randn(h, L, r, dtype=torch.float64) * 0.1
    g = torch.randn(N, r, dtype=torch.float64)

    dC_ag, dz_ag = _autograd_dC_dz(z, C, g, -3.0, 3.0, G)
    dC_ref, dz_ref = flash_spline_delta_backward_ref(z, C, g, -3.0, 3.0, G)
    # Backward ref runs in fp32 while autograd ref runs in fp64; allow
    # fp32-noise-level tolerance in the comparison.
    assert torch.allclose(dC_ref.to(torch.float64), dC_ag, atol=1e-5, rtol=1e-4), \
        f"dC max diff = {(dC_ref.to(torch.float64) - dC_ag).abs().max()}"
    assert torch.allclose(dz_ref.to(torch.float64), dz_ag, atol=1e-5, rtol=1e-4), \
        f"dz max diff = {(dz_ref.to(torch.float64) - dz_ag).abs().max()}"


def test_dC_skewed_matches_autograd():
    torch.manual_seed(0)
    N, h, r, G = 16, 32, 8, 10
    L = G + 2
    z = _generate_workload("skewed", N, h, torch.float64, "cpu")
    C = torch.randn(h, L, r, dtype=torch.float64) * 0.1
    g = torch.randn(N, r, dtype=torch.float64)

    dC_ag, dz_ag = _autograd_dC_dz(z, C, g, -3.0, 3.0, G)
    dC_ref, dz_ref = flash_spline_delta_backward_ref(z, C, g, -3.0, 3.0, G)
    # Backward ref runs in fp32 while autograd ref runs in fp64; allow
    # fp32-noise-level tolerance in the comparison.
    assert torch.allclose(dC_ref.to(torch.float64), dC_ag, atol=1e-5, rtol=1e-4), \
        f"dC max diff = {(dC_ref.to(torch.float64) - dC_ag).abs().max()}"
    assert torch.allclose(dz_ref.to(torch.float64), dz_ag, atol=1e-5, rtol=1e-4), \
        f"dz max diff = {(dz_ref.to(torch.float64) - dz_ag).abs().max()}"


def test_dC_collapsed_matches_autograd():
    """Adversarial: nearly all tokens land in the same bin → maximum
    atomic contention if implemented as scatter-add.
    Per v7 §R.6.9 the kernel must handle this; here we just verify the
    formula reference still computes correctly under high contention."""
    torch.manual_seed(0)
    N, h, r, G = 32, 32, 8, 10
    L = G + 2
    z = _generate_workload("collapsed", N, h, torch.float64, "cpu")
    C = torch.randn(h, L, r, dtype=torch.float64) * 0.1
    g = torch.randn(N, r, dtype=torch.float64)

    dC_ag, dz_ag = _autograd_dC_dz(z, C, g, -3.0, 3.0, G)
    dC_ref, dz_ref = flash_spline_delta_backward_ref(z, C, g, -3.0, 3.0, G)
    # All tokens hit the same bin → high-contention atomic accumulation.
    # Looser tolerance because (a) bwd ref runs in fp32 vs autograd's fp64,
    # and (b) summing many fp32 terms into a single dC slot amplifies noise.
    assert torch.allclose(dC_ref.to(torch.float64), dC_ag, atol=1e-4, rtol=1e-3), \
        f"dC max diff (collapsed) = {(dC_ref.to(torch.float64) - dC_ag).abs().max()}"
    assert torch.allclose(dz_ref.to(torch.float64), dz_ag, atol=1e-4, rtol=1e-3), \
        f"dz max diff (collapsed) = {(dz_ref.to(torch.float64) - dz_ag).abs().max()}"


def test_dC_zero_when_g_delta_zero():
    torch.manual_seed(0)
    N, h, r, G = 8, 16, 4, 8
    L = G + 2
    z = torch.randn(N, h, dtype=torch.float64)
    C = torch.randn(h, L, r, dtype=torch.float64) * 0.1
    g = torch.zeros(N, r, dtype=torch.float64)
    dC, dz = flash_spline_delta_backward_ref(z, C, g, -3.0, 3.0, G)
    assert dC.abs().max() == 0.0
    assert dz.abs().max() == 0.0


def test_dz_partition_of_unity_zero():
    """If C is constant in b, dz_spline = 0 by partition-of-unity (v7 §R.3.0)."""
    torch.manual_seed(0)
    N, h, r, G = 16, 32, 8, 10
    L = G + 2
    z = torch.randn(N, h, dtype=torch.float64)
    # constant in b: C[j, b, c] = phi[j, c] for all b
    phi = torch.randn(h, 1, r, dtype=torch.float64) * 0.1
    C = phi.expand(h, L, r).contiguous()
    g = torch.randn(N, r, dtype=torch.float64)
    _, dz = flash_spline_delta_backward_ref(z, C, g, -3.0, 3.0, G)
    # dz should be nearly 0 for in-range tokens (clamped tokens have B=0
    # already so contribution = 0, but partition-of-unity ensures the
    # in-range contribution sums to zero too).
    # bwd ref runs in fp32 internally, so partition-of-unity gives dz=0
    # only up to fp32 noise, not exact double precision.
    max_dz = float(dz.abs().max().item())
    assert max_dz < 1e-6, f"partition-of-unity should give dz=0, got {max_dz}"


def test_dz_zero_when_C_zero():
    """v7 §R.5: at C=0, dz from spline path is zero (because the basis
    derivatives multiply C, which is zero)."""
    torch.manual_seed(0)
    N, h, r, G = 8, 16, 4, 8
    L = G + 2
    z = torch.randn(N, h, dtype=torch.float64)
    C = torch.zeros(h, L, r, dtype=torch.float64)
    g = torch.randn(N, r, dtype=torch.float64)
    dC, dz = flash_spline_delta_backward_ref(z, C, g, -3.0, 3.0, G)
    assert dz.abs().max() == 0.0
    # dC is non-zero (driven by B_b(z_j) * g_delta) — confirms cold-start
    assert dC.abs().sum() > 0
