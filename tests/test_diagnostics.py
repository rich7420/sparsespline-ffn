"""Tests for diagnostics helpers (v7 §R.1.4, §R.6.10)."""
from __future__ import annotations

import math

import torch

from sparsespline_ffn.diagnostics import (
    rms, rho_delta_ratio, rho_delta_from_module,
    bin_occupancy, bin_entropy, dead_bin_fraction,
    code_norms, grad_norm, snapshot_rl_spline_kv,
)
from sparsespline_ffn.rl_spline_kv_reference import (
    RLSplineKVConfig, RLSplineKVReference,
)


def test_rms_basic():
    t = torch.tensor([3.0, 4.0])
    # rms = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.5355
    assert abs(rms(t) - math.sqrt(12.5)) < 1e-5


def test_rho_delta_zero_when_spline_zero():
    """If spline branch outputs zero, rho_delta = 0."""
    base = torch.randn(8, 16)
    spline = torch.zeros(8, 16)
    assert rho_delta_ratio(base, spline) == 0.0


def test_rho_delta_one_when_equal_rms():
    base = torch.ones(8, 16)
    spline = torch.ones(8, 16)
    assert abs(rho_delta_ratio(base, spline) - 1.0) < 1e-6


def test_rho_delta_C_zero_init_gives_zero_ratio():
    """v7 §R.5: at C=0 init, the spline path output is exactly 0,
    so rho_delta=0.  After training, this should grow."""
    cfg = RLSplineKVConfig(d=32, h_ratio=1.0, r=8, G=8, init_C_zero=True)
    m = RLSplineKVReference(cfg)
    x = torch.randn(2, 32)
    rd = rho_delta_from_module(m, x)
    assert rd["rho_delta"] == 0.0
    assert rd["rms_spline"] == 0.0
    assert rd["rms_base"] > 0.0


def test_rho_delta_nonzero_when_C_nonzero():
    cfg = RLSplineKVConfig(d=32, h_ratio=1.0, r=8, G=8, init_C_zero=False)
    m = RLSplineKVReference(cfg)
    x = torch.randn(2, 32)
    rd = rho_delta_from_module(m, x)
    assert rd["rho_delta"] > 0.0
    assert rd["rms_spline"] > 0.0


def test_bin_occupancy_uniform():
    """Uniform z over grid range should give roughly uniform occupancy
    (excluding the last bin since the kernel clamps to G-1)."""
    G = 10
    L = G + 2
    torch.manual_seed(0)
    # Uniform in [grid_lo, grid_hi]
    z = torch.empty(2000, 4).uniform_(-3.0, 3.0)
    occ = bin_occupancy(z, -3.0, 3.0, G)
    assert occ.shape == (L,), occ.shape
    # First G bins should each have roughly 2000*4/G = 800 ± slack
    for b in range(G):
        # allow large tolerance — random uniform has variance
        assert 600 < occ[b].item() < 1100, f"bin {b}: {occ[b].item()}"
    # The last 2 bins (extension for B2 cushion) should be 0 (no token
    # has its primary bin land there because we clamp to G-1).
    assert occ[L - 1] == 0
    assert occ[L - 2] == 0


def test_bin_entropy_uniform_near_max():
    """Uniform distribution gives entropy ≈ log(num_active_bins)."""
    occ = torch.tensor([100, 100, 100, 100, 100, 0, 0])  # 5 active bins
    ent = bin_entropy(occ)
    assert abs(ent - math.log(5)) < 1e-6


def test_bin_entropy_collapsed_zero():
    """All-in-one-bin gives entropy = 0."""
    occ = torch.tensor([1000, 0, 0, 0])
    ent = bin_entropy(occ)
    assert ent == 0.0


def test_dead_bin_fraction_basic():
    occ = torch.tensor([1000, 1000, 1, 0])  # 2 of 4 are "dead" at 1e-3
    df = dead_bin_fraction(occ, threshold_frac=1e-3)
    # Total = 2001, threshold = 2.001, bins with < 2.001 = bins 2 and 3 → 2/4 = 0.5
    assert abs(df - 0.5) < 1e-6


def test_code_norms_zero_C():
    C = torch.zeros(4, 6, 8)
    n = code_norms(C)
    assert n["frobenius"] == 0.0
    assert n["mean_abs"] == 0.0
    assert n["max_abs"] == 0.0


def test_grad_norm_no_grad_returns_zero():
    p = torch.nn.Parameter(torch.randn(4, 4))
    assert grad_norm(p) == 0.0


def test_grad_norm_after_backward():
    p = torch.nn.Parameter(torch.randn(4, 4))
    loss = (p * 2.0).sum()
    loss.backward()
    # grad of (p*2).sum() w.r.t. p is all 2s; norm = sqrt(16*4) = 8
    assert abs(grad_norm(p) - 8.0) < 1e-5


def test_full_snapshot_after_step():
    """End-to-end: build module, do one fwd+bwd, take snapshot."""
    cfg = RLSplineKVConfig(d=16, h_ratio=1.0, r=4, G=8, init_C_zero=False)
    m = RLSplineKVReference(cfg)
    x = torch.randn(4, 16)
    target = torch.randn_like(x)
    loss = (m(x) - target).pow(2).sum()
    loss.backward()
    snap = snapshot_rl_spline_kv(m, x)
    # Sanity: all fields populated, no NaN, gradients nonzero
    assert snap.rho_delta > 0
    assert snap.bin_entropy_nats > 0
    assert snap.bin_entropy_max_nats > snap.bin_entropy_nats - 1e-6
    assert snap.C_grad_norm > 0
    assert snap.W_delta_grad_norm > 0
    assert snap.K_grad_norm > 0
    assert math.isfinite(snap.dead_bin_fraction)
