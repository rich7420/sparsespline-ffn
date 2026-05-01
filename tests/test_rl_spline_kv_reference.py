"""Tests for the Phase B1 PyTorch reference of RL-Spline-KV.

These are the gold standard against which the eventual FlashSplineFeature
Triton kernel will be tested.  Each test maps to a property in
``docs/THEORY_v7_RL_SPLINE_KV.md``.
"""
from __future__ import annotations

import torch

from sparsespline_ffn.rl_spline_kv_reference import (
    RLSplineKVConfig, RLSplineKVReference,
    flash_spline_feature_reference,
)


# Small dims for fast tests.  Real shapes (h=768, r=64, L=22) are
# exercised in the microbench, not the unit tests.
def _small_setup(seed: int = 0):
    torch.manual_seed(seed)
    return dict(N=8, h=16, r=4, G=10)


# -----------------------------------------------------------------------
# Shape and dtype invariants


def test_reference_returns_correct_shape():
    s = _small_setup()
    z = torch.randn(s["N"], s["h"])
    C = torch.randn(s["h"], s["G"] + 2, s["r"])
    f = flash_spline_feature_reference(z, C, grid_lo=-3.0, grid_hi=3.0,
                                        G=s["G"])
    assert f.shape == (s["N"], s["h"] + s["r"]), f.shape
    assert f.dtype == z.dtype


def test_reference_module_forward():
    cfg = RLSplineKVConfig(d=32, h_ratio=1.0, r=8, G=10)
    m = RLSplineKVReference(cfg)
    x = torch.randn(2, 4, 32)
    y = m(x)
    assert y.shape == x.shape, (y.shape, x.shape)


# -----------------------------------------------------------------------
# v7 §R.5: C=0 cold start — forward equals just [phi(z); 0]


def test_C_zero_cold_start_forward():
    s = _small_setup()
    z = torch.randn(s["N"], s["h"])
    C = torch.zeros(s["h"], s["G"] + 2, s["r"])
    f = flash_spline_feature_reference(z, C, grid_lo=-3.0, grid_hi=3.0,
                                        G=s["G"])
    a, delta = f[:, :s["h"]], f[:, s["h"]:]
    expected_a = torch.where(z > 0, z * z, torch.zeros_like(z))
    assert torch.allclose(a, expected_a, atol=1e-6), \
        f"a half should equal relu²(z), max diff {(a-expected_a).abs().max()}"
    assert torch.all(delta == 0.0), \
        f"delta half should be zero when C=0, max abs {delta.abs().max()}"


def test_C_zero_grad_to_C_is_nonzero():
    """Critical test for v7 §R.5: even with C=0, dC must be nonzero so C
    starts to learn from step 0.  Otherwise the spline branch is dead.
    """
    s = _small_setup()
    z = torch.randn(s["N"], s["h"])
    C = torch.zeros(s["h"], s["G"] + 2, s["r"], requires_grad=True)
    f = flash_spline_feature_reference(z, C, grid_lo=-3.0, grid_hi=3.0,
                                        G=s["G"])
    # Loss that uses the delta half (index >= h) so gradient flows into C
    delta = f[:, s["h"]:]
    loss = (delta * torch.randn_like(delta)).sum()
    loss.backward()
    assert C.grad is not None
    assert C.grad.abs().sum() > 0, "dC should be nonzero with random g_delta even when C=0"


# -----------------------------------------------------------------------
# v7 §R.3.0: partition-of-unity gradient identity
# If C[j, b, c] is constant in b for all (j, c), then dz_spline = 0.


def test_partition_of_unity_gives_zero_dz():
    s = _small_setup()
    z = torch.randn(s["N"], s["h"], requires_grad=True)
    # Constant C across the b dimension
    L = s["G"] + 2
    C_per_jc = torch.randn(s["h"], 1, s["r"])
    C = C_per_jc.expand(s["h"], L, s["r"]).contiguous()
    f = flash_spline_feature_reference(z, C, grid_lo=-3.0, grid_hi=3.0,
                                        G=s["G"])
    # Take loss only on the delta half
    delta = f[:, s["h"]:]
    loss = (delta * torch.randn_like(delta)).sum()
    loss.backward()
    # dz contribution from the delta path should be ~0 because the
    # interpolation result is a constant function of z (B0+B1+B2=1
    # times the same constant).
    # However z.grad also includes the path through `a = relu_sq(z)`.
    # Since loss does NOT use a here (we sliced delta only), the
    # phi gradient is 0, and dz should equal exactly the spline
    # contribution — which by partition-of-unity is 0.
    assert z.grad is not None
    max_g = z.grad.abs().max().item()
    assert max_g < 1e-5, \
        f"partition-of-unity should give dz=0 from delta path, got max |dz|={max_g}"


# -----------------------------------------------------------------------
# Linearity in C: doubling C doubles delta (forward sanity)


def test_delta_linear_in_C():
    s = _small_setup()
    z = torch.randn(s["N"], s["h"])
    C = torch.randn(s["h"], s["G"] + 2, s["r"])
    f1 = flash_spline_feature_reference(z, C, grid_lo=-3.0, grid_hi=3.0,
                                         G=s["G"])
    f2 = flash_spline_feature_reference(z, 2.0 * C, grid_lo=-3.0,
                                         grid_hi=3.0, G=s["G"])
    delta1 = f1[:, s["h"]:]
    delta2 = f2[:, s["h"]:]
    assert torch.allclose(delta2, 2.0 * delta1, atol=1e-5), \
        f"delta should be linear in C: max diff {(delta2 - 2*delta1).abs().max()}"
    # And a half is unchanged
    a1 = f1[:, :s["h"]]; a2 = f2[:, :s["h"]]
    assert torch.allclose(a1, a2, atol=1e-7)


# -----------------------------------------------------------------------
# Out-of-range z (clamp regime): partition is masked, delta only from
# in-range tokens.


def test_out_of_range_z_does_not_explode():
    s = _small_setup()
    # Half tokens way out of grid range
    z = torch.cat([torch.randn(4, s["h"]), torch.randn(4, s["h"]) * 100.0],
                  dim=0)
    C = torch.randn(s["h"], s["G"] + 2, s["r"])
    f = flash_spline_feature_reference(z, C, grid_lo=-3.0, grid_hi=3.0,
                                        G=s["G"])
    assert torch.isfinite(f).all(), "out-of-range z should not blow up forward"


# -----------------------------------------------------------------------
# Numerical gradient check (small dims only)


def test_numerical_grad_check_C():
    """Autograd vs finite difference for dL/dC."""
    torch.manual_seed(0)
    z = torch.randn(3, 4, dtype=torch.float64)
    C = torch.randn(4, 7, 2, dtype=torch.float64, requires_grad=True)
    G = 5  # L = G+2 = 7

    def loss_fn(C):
        f = flash_spline_feature_reference(z, C, grid_lo=-2.0,
                                            grid_hi=2.0, G=G)
        return f.sum()

    assert torch.autograd.gradcheck(loss_fn, (C,), eps=1e-6, atol=1e-4,
                                     rtol=1e-3, fast_mode=True), \
        "gradcheck failed for dL/dC"


def test_numerical_grad_check_z():
    torch.manual_seed(0)
    z = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    C = torch.randn(4, 7, 2, dtype=torch.float64)
    G = 5

    def loss_fn(z):
        f = flash_spline_feature_reference(z, C, grid_lo=-2.0,
                                            grid_hi=2.0, G=G)
        return f.sum()

    assert torch.autograd.gradcheck(loss_fn, (z,), eps=1e-6, atol=1e-4,
                                     rtol=1e-3, fast_mode=True), \
        "gradcheck failed for dL/dz"


# -----------------------------------------------------------------------
# Module-level: C=0 init means initial output equals a narrow MLP


def test_module_C_zero_init_initial_forward():
    """At init with init_C_zero=True, the module should output exactly
    W_out[:, :h] @ relu_sq(K x).  (Spline branch contributes nothing.)
    """
    cfg = RLSplineKVConfig(d=16, h_ratio=1.0, r=4, G=8, init_C_zero=True)
    m = RLSplineKVReference(cfg)
    assert torch.all(m.C == 0.0), "init_C_zero should leave C all zero"

    x = torch.randn(2, 16)
    z = m.K(x)
    a = torch.where(z > 0, z * z, torch.zeros_like(z))  # [2, h]
    expected_y = torch.nn.functional.linear(
        torch.cat([a, torch.zeros(2, cfg.r)], dim=-1),
        m.W_out.weight, m.W_out.bias,
    )
    actual_y = m(x)
    assert torch.allclose(actual_y, expected_y, atol=1e-5), \
        f"C=0 init should make module ≡ narrow MLP path, max diff " \
        f"{(actual_y - expected_y).abs().max()}"
