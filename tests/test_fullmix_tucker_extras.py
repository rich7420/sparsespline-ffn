"""Extra invariant tests beyond ``test_fullmix_tucker.py``.

These cover theory claims and code paths that the original test file does
not exercise:

  A. HOSVD warm-start re-injection — load the helper's factors back into a
     ``FullMixTuckerFFN`` and verify forward agreement with the original
     dense W.  (L.4 "train dense -> SVD-init -> switch" flow.)
  B. ``torch.utils.checkpoint`` round-trip — the recommended K.0.3 fix for
     the beta-tensor VRAM cost must produce identical output and valid
     gradients.
  C. ``gamma`` learnability — the per-layer scalar gain must accumulate
     gradient and move under SGD.
  D. T_direct (``use_mixer=False``) end-to-end — init variance and SGD
     learnability, not just forward equivalence.
  E. ``bias_in_mixer=True`` path — the optional bias on A is exercised.
  F. ``output_subspace_dim`` diagnostic returns the documented bound.
  G. ``extra_repr`` reports the configured fields.
  H. ``variance_preserving_spline_coef_init`` honors ``target_output_var``.
  I. F.4.a param formula at multiple scales (parametrized).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.utils.checkpoint as ckpt
from conftest import make_small_ffn as _make_small

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN
from sparsespline_ffn.tucker_init import (
    hosvd_warmstart_from_dense,
    variance_preserving_spline_coef_init,
)

# ---- A. HOSVD warm-start re-injection -----------------------------------


def test_hosvd_warmstart_reinjection_matches_dense_W():
    """Take a layer's dense W, factor it via HOSVD, write factors back into a
    fresh layer, and verify forward(x) matches the original.

    This is the L.4 "dense pretrain -> SVD-init Tucker -> switch" flow:
    factor extraction must be lossless when the original W is already exactly
    Tucker rank-(R_o, R_i, R_b).
    """
    src = _make_small(d=12, m=12, R_o=6, R_i=6, R_b=3, G=8)
    cfg = src.cfg

    with torch.no_grad():
        W_dense = src.reconstruct_dense_W()

    U, V, core, Q = hosvd_warmstart_from_dense(
        W_dense, R_o=cfg.R_o, R_i=cfg.R_i, R_b=cfg.R_b
    )

    dst = _make_small(d=12, m=12, R_o=6, R_i=6, R_b=3, G=8)
    with torch.no_grad():
        dst.U.copy_(U)
        dst.V.copy_(V)
        dst.C.copy_(core)
        dst.Q.copy_(Q)
        dst.gamma.copy_(src.gamma)
        # Mirror the mixer so spline inputs match.
        dst.A.weight.copy_(src.A.weight)
        if src.A.bias is not None:
            dst.A.bias.copy_(src.A.bias)

    x = torch.randn(8, cfg.d)
    with torch.no_grad():
        y_src = src(x)
        y_dst = dst(x)
    rel = (y_src - y_dst).norm() / (y_src.norm() + 1e-9)
    assert rel < 1e-4, (
        f"HOSVD reinjection forward drift {rel.item():.2e} > 1e-4; "
        f"warm-start path is lossy on already-low-rank W"
    )


def test_hosvd_warmstart_returns_orthogonal_U_V():
    """U and V from HOSVD on a random tensor should have near-orthogonal
    columns — this is what makes the L.4 variance derivation hold after a
    warm-start."""
    torch.manual_seed(42)
    W = torch.randn(20, 18, 9)
    U, V, _core, _Q = hosvd_warmstart_from_dense(W, R_o=6, R_i=6, R_b=4)
    eye_U = U.t() @ U
    eye_V = V.t() @ V
    assert torch.allclose(eye_U, torch.eye(6), atol=1e-5)
    assert torch.allclose(eye_V, torch.eye(6), atol=1e-5)


# ---- B. torch.utils.checkpoint round-trip --------------------------------


def test_checkpoint_forward_matches_eager():
    """Wrapping the layer with ``torch.utils.checkpoint`` (K.0.3's recommended
    VRAM fix for Pattern Full) must not change outputs or break autograd."""
    ffn = _make_small()
    torch.manual_seed(11)
    x = torch.randn(4, ffn.cfg.d, requires_grad=True)

    # Eager reference.
    y_eager = ffn(x)
    y_eager.pow(2).sum().backward()
    grad_eager = x.grad.detach().clone()
    ffn.zero_grad(set_to_none=True)
    x.grad = None

    # Checkpointed.  use_reentrant=False is the modern path.
    y_ckp = ckpt.checkpoint(ffn, x, use_reentrant=False)
    y_ckp.pow(2).sum().backward()

    assert torch.allclose(y_eager, y_ckp, atol=1e-6)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.allclose(grad_eager, x.grad, atol=1e-5)


def test_checkpoint_in_residual_stack_K6():
    """Mimic Pattern A+ usage: 6 residual layers, each checkpointed.  Verify
    forward is finite and backward populates parameter grads."""
    K = 6
    cfg = FullMixTuckerConfig(d=24, m=24, R_o=8, R_i=8, R_b=4, G=8)
    layers = torch.nn.ModuleList([FullMixTuckerFFN(cfg) for _ in range(K)])
    x = torch.randn(8, cfg.d, requires_grad=True)
    h = x
    for layer in layers:
        h = h + ckpt.checkpoint(layer, h, use_reentrant=False)
    assert torch.isfinite(h).all()
    h.pow(2).sum().backward()
    for layer in layers:
        for p in layer.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()


# ---- C. Gamma learnability -----------------------------------------------


def test_gamma_accumulates_gradient_and_moves():
    """``gamma`` is the per-layer learnable scalar gain (J.1.b).  After SGD
    on a regression target it must (1) get nonzero grad and (2) move from 1.0.
    """
    ffn = _make_small()
    torch.manual_seed(20)
    x = torch.randn(32, ffn.cfg.d)
    target = 2.0 * ffn(x).detach()  # force gamma to want to grow

    initial_gamma = ffn.gamma.item()
    opt = torch.optim.SGD([ffn.gamma], lr=0.05)
    for _ in range(30):
        opt.zero_grad()
        y = ffn(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        assert ffn.gamma.grad is not None
        assert torch.isfinite(ffn.gamma.grad)
        opt.step()

    final_gamma = ffn.gamma.item()
    assert abs(final_gamma - initial_gamma) > 0.05, (
        f"gamma did not move: {initial_gamma} -> {final_gamma}"
    )
    assert final_gamma > initial_gamma, (
        "gamma should grow toward 2x target, but it shrank/stayed put"
    )


# ---- D. T_direct (use_mixer=False) end-to-end ----------------------------


def test_t_direct_init_variance_reasonable():
    """T_direct (no mixer) — output std should still be in the [0.4, 2.5]
    band at init.  Confirms the variance-preserving init is not specific to
    the mixer path."""
    cfg = FullMixTuckerConfig(d=64, m=64, R_o=32, R_i=32, R_b=8,
                              G=16, use_mixer=False)
    torch.manual_seed(30)
    ffn = FullMixTuckerFFN(cfg)
    x = torch.randn(256, cfg.d)
    with torch.no_grad():
        y = ffn(x)
    assert 0.4 < y.std().item() < 2.5, (
        f"T_direct init std {y.std():.3f} outside [0.4, 2.5]"
    )


def test_t_direct_can_learn_via_sgd():
    """T_direct must still be trainable.  Same target shape as the small
    SGD smoke test in the main suite."""
    cfg = FullMixTuckerConfig(d=8, m=8, R_o=8, R_i=8, R_b=4, G=6,
                              use_mixer=False)
    torch.manual_seed(31)
    ffn = FullMixTuckerFFN(cfg)
    x = torch.randn(64, cfg.d)
    target = torch.zeros(64, cfg.d)
    target[:, 0] = torch.sin(x[:, 0])
    target[:, 1] = 0.3 * x[:, 1] ** 2

    opt = torch.optim.SGD(ffn.parameters(), lr=0.05)
    initial = float("inf")
    final = float("inf")
    for step in range(200):
        opt.zero_grad()
        mse = (ffn(x) - target).pow(2).mean()
        if step == 0:
            initial = mse.item()
        if step == 199:
            final = mse.item()
        mse.backward()
        opt.step()
    assert math.isfinite(final)
    assert final < initial / 3.0, (
        f"T_direct failed to learn: {initial:.4f} -> {final:.4f}"
    )


# ---- E. bias_in_mixer=True path ------------------------------------------


def test_bias_in_mixer_creates_bias_param():
    cfg = FullMixTuckerConfig(d=16, m=16, R_o=8, R_i=8, R_b=4, G=6,
                              bias_in_mixer=True)
    torch.manual_seed(40)
    ffn = FullMixTuckerFFN(cfg)
    assert ffn.A is not None
    assert ffn.A.bias is not None
    # init_mixer zeros the bias, so it should start at 0.
    assert torch.allclose(ffn.A.bias, torch.zeros_like(ffn.A.bias))


def test_bias_in_mixer_trains_and_moves():
    cfg = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=4, G=6,
                              bias_in_mixer=True)
    torch.manual_seed(41)
    ffn = FullMixTuckerFFN(cfg)
    initial_bias = ffn.A.bias.detach().clone()

    x = torch.randn(64, cfg.d)
    target = torch.zeros(64, cfg.d)
    target[:, 0] = 0.5 * x[:, 0] + 0.2  # constant offset -> bias should help

    opt = torch.optim.SGD(ffn.parameters(), lr=0.05)
    for _ in range(100):
        opt.zero_grad()
        (ffn(x) - target).pow(2).mean().backward()
        opt.step()

    bias_delta = (ffn.A.bias - initial_bias).abs().max().item()
    assert bias_delta > 1e-4, (
        f"mixer bias never moved (max |delta|={bias_delta:.2e}); "
        f"is the parameter actually trainable?"
    )


# ---- F. output_subspace_dim diagnostic -----------------------------------


def test_output_subspace_dim_returns_R_o_for_random_init():
    """At init, U is orthogonal so its rank should equal R_o."""
    ffn = _make_small()
    assert ffn.output_subspace_dim() == ffn.cfg.R_o


def test_output_subspace_dim_collapses_when_U_zeroed():
    """Sanity: zeroing U makes its rank 0 (catches a future change that
    breaks the diagnostic)."""
    ffn = _make_small()
    with torch.no_grad():
        ffn.U.zero_()
    assert ffn.output_subspace_dim() == 0


# ---- G. extra_repr reports the configured fields -------------------------


def test_extra_repr_contains_key_hyperparams():
    ffn = _make_small(d=32, m=32, R_o=16, R_i=8, R_b=4, G=10)
    rep = ffn.extra_repr()
    assert "d=32" in rep
    assert "m=32" in rep
    assert "R=(16,8,4)" in rep
    assert "G=10" in rep
    assert "L=11" in rep   # G + 1 for B1
    assert "mixer=True" in rep


def test_extra_repr_reflects_t_direct():
    cfg = FullMixTuckerConfig(d=16, m=16, R_o=8, R_i=8, R_b=4, G=6,
                              use_mixer=False)
    ffn = FullMixTuckerFFN(cfg)
    assert "mixer=False" in ffn.extra_repr()


# ---- H. variance_preserving_spline_coef_init target_output_var ------------


@pytest.mark.parametrize("target_var", [0.25, 1.0, 4.0])
def test_variance_init_honors_target_output_var(target_var):
    """sigma_c should scale as sqrt(target_var); doubling target_var sqrt-doubles
    sigma_c, which roughly doubles output var."""
    Q = torch.empty(20, 8)
    variance_preserving_spline_coef_init(
        Q, d=128, R_o=64, target_output_var=target_var
    )
    expected_sigma = math.sqrt(3.0 * 128.0 * target_var / (2.0 * 64.0))
    actual_sigma = Q.std().item()
    # Sample-std on 160 entries: relative error roughly 1/sqrt(2*160)~5%.
    assert abs(actual_sigma - expected_sigma) / expected_sigma < 0.15, (
        f"target_var={target_var}: sigma {actual_sigma:.3f} vs expected "
        f"{expected_sigma:.3f}"
    )


# ---- I. F.4.a param formula at multiple scales ---------------------------


@pytest.mark.parametrize("d,m,R_o,R_i,R_b,G", [
    (64, 64, 32, 32, 8, 12),
    (128, 128, 64, 64, 16, 20),
    (256, 256, 96, 96, 16, 20),
    (256, 256, 256, 96, 16, 20),  # asymmetric, Strategy A flavor
    (768, 768, 96, 96, 16, 20),    # nanochat
])
def test_param_count_F4a_formula_at_scale(d, m, R_o, R_i, R_b, G):
    cfg = FullMixTuckerConfig(d=d, m=m, R_o=R_o, R_i=R_i, R_b=R_b, G=G)
    ffn = FullMixTuckerFFN(cfg)
    L = G + 1
    expected = (
        d * m
        + d * R_o
        + m * R_i
        + L * R_b
        + R_o * R_i * R_b
        + 1   # gamma
    )
    actual = sum(p.numel() for p in ffn.parameters())
    assert actual == expected, (
        f"d={d}, R=({R_o},{R_i},{R_b}): "
        f"actual {actual:,} != F.4.a formula {expected:,}"
    )


# ---- J. 4D and 5D leading-dim shapes -------------------------------------


def test_higher_rank_leading_dims():
    """Leading-dim flatten/reshape must work for 4-D and 5-D inputs."""
    ffn = _make_small()
    d = ffn.cfg.d
    for leading in [(2,), (2, 3), (2, 3, 4), (1, 2, 3, 4)]:
        x = torch.randn(*leading, d)
        y = ffn(x)
        assert y.shape == x.shape, f"failed for leading={leading}"
        assert torch.isfinite(y).all()
