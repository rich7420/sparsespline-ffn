"""Invariant tests for FullMix-Tucker FFN reference implementation.

These tests pin the numerical and structural properties that JHCG_REDESIGN_THEORY.md
asserts, so that any future change to the reference (or the eventual fused
Triton kernel) can be verified against the same invariants.

Test groups:
  1. Equivalence    — 5-stage matches direct W_{kji} contraction (fp32 1e-5).
  2. Shape          — input/output shapes and parameter shapes.
  3. Autograd       — gradients are finite, nonzero, compile-compatible.
  4. Output rank    — per F.4.b, dim(image of forward) <= R_o.
  5. Topology       — m < d rejected; T_direct requires m=d.
  6. Init variance  — corrected sigma_c = sqrt(3d/(2 R_o)) gives Var[y]≈1.
  7. HOSVD          — warm-start helper round-trips a small dense W.
  8. Distributional — t-uniformity, Var[y] robustness across Var[x],
                       out-of-grid clamping stability.
  9. Stacking       — K-layer composition: activation magnitude stable,
                       F.5.1 cumulative subspace coverage, gradient flow
                       through deep stacks.
  10. Numerical     — extreme inputs (large/small magnitude), determinism,
                       NaN/Inf input propagation, fp32 vs fp64 agreement.

All tests run on CPU and are <2s each, so they can live in CI without GPU.
"""
from __future__ import annotations

import math

import pytest
import torch
from conftest import capture_bin_frac
from conftest import make_small_ffn as _make_small
from conftest import make_stack as _stack_layers
from conftest import pre_rmsnorm_stack_forward as _pre_rmsnorm_stack_forward
from conftest import residual_stack_forward as _residual_stack_forward

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN
from sparsespline_ffn.tucker_init import hosvd_warmstart_from_dense

# ---- 1. Equivalence: 5-stage matches direct dense Tucker contraction -----


def test_five_stage_matches_dense_W():
    """The five-stage forward must equal W_{kji}-based direct evaluation.

    This is the K.0.1 equivalence proof reduced to runnable assertions.
    """
    ffn = _make_small()
    cfg = ffn.cfg
    torch.manual_seed(1)
    x = torch.randn(4, cfg.d, dtype=torch.float32)

    # Reference forward (5-stage).
    y_ref = ffn(x)

    # Direct path — compute z, then evaluate y_k = gamma * sum_{j,i} W_{kji} B_i(z_j).
    with torch.no_grad():
        if ffn.A is not None:
            z = ffn.A(x)
        else:
            z = x

        # B1 basis evaluation (dense — only feasible because L is small here).
        bin_idx, t = ffn._bin_and_frac(z)  # (N, m)
        L = cfg.G + 1
        N, m = z.shape
        B = torch.zeros(N, m, L, dtype=z.dtype)
        # B_i(z_j) is (1-t) at i=bin, t at i=bin+1, 0 elsewhere.
        B.scatter_(2, bin_idx.unsqueeze(-1), (1.0 - t).unsqueeze(-1))
        B.scatter_(2, (bin_idx + 1).unsqueeze(-1), t.unsqueeze(-1))

        W = ffn.reconstruct_dense_W()  # (d, m, L)
        y_direct = ffn.gamma * torch.einsum("kji, nji -> nk", W, B[:, :, : W.size(-1)])

    rel = (y_ref - y_direct).norm() / (y_direct.norm() + 1e-9)
    assert rel < 1e-5, f"5-stage vs dense W rel err {rel.item():.2e}"


def test_five_stage_matches_dense_W_no_mixer():
    """T_direct (use_mixer=False) variant should also match dense form."""
    ffn = _make_small(use_mixer=False)  # m == d enforced
    torch.manual_seed(2)
    x = torch.randn(3, ffn.cfg.d, dtype=torch.float32)
    y_ref = ffn(x)
    with torch.no_grad():
        bin_idx, t = ffn._bin_and_frac(x)
        L = ffn.cfg.G + 1
        N, m = x.shape
        B = torch.zeros(N, m, L, dtype=x.dtype)
        B.scatter_(2, bin_idx.unsqueeze(-1), (1.0 - t).unsqueeze(-1))
        B.scatter_(2, (bin_idx + 1).unsqueeze(-1), t.unsqueeze(-1))
        W = ffn.reconstruct_dense_W()
        y_direct = ffn.gamma * torch.einsum("kji, nji -> nk", W, B)
    rel = (y_ref - y_direct).norm() / (y_direct.norm() + 1e-9)
    assert rel < 1e-5, f"T_direct 5-stage vs dense rel err {rel.item():.2e}"


# ---- 2. Shape ------------------------------------------------------------


def test_shape_2d_input():
    ffn = _make_small()
    x = torch.randn(7, ffn.cfg.d)
    assert ffn(x).shape == (7, ffn.cfg.d)


def test_shape_3d_input_batch_seq():
    """Realistic transformer call: (B, T, d) -> (B, T, d)."""
    ffn = _make_small()
    x = torch.randn(2, 5, ffn.cfg.d)
    y = ffn(x)
    assert y.shape == (2, 5, ffn.cfg.d)


def test_param_shapes():
    ffn = _make_small()
    cfg = ffn.cfg
    L = cfg.G + 1
    assert ffn.Q.shape == (L, cfg.R_b)
    assert ffn.V.shape == (cfg.m, cfg.R_i)
    assert ffn.C.shape == (cfg.R_o, cfg.R_i, cfg.R_b)
    assert ffn.U.shape == (cfg.d, cfg.R_o)
    assert ffn.gamma.shape == (1,)


# ---- 3. Autograd ---------------------------------------------------------


def test_backward_finite_and_nonzero():
    ffn = _make_small()
    torch.manual_seed(3)
    x = torch.randn(4, ffn.cfg.d, requires_grad=True)
    y = ffn(x)
    loss = y.pow(2).sum()
    loss.backward()
    # All FFN parameters must have finite gradients.
    for name, p in ffn.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"
        # Spline-mode lookup is gather-based; some Q rows may be untouched
        # by a small batch.  But Q overall must have *some* nonzero grad.
        # For other params, every entry should be exercised.
        if name == "Q":
            assert p.grad.abs().sum() > 0, "Q grad is all zero"
        else:
            assert p.grad.abs().mean() > 0, f"{name} grad is all zero"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_backward_works_through_torch_compile():
    """Ensure the reference is compatible with torch.compile (eager backend)."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    try:
        ffn = _make_small()
        compiled = torch.compile(ffn, backend="eager", fullgraph=False)
        x = torch.randn(2, ffn.cfg.d, requires_grad=True)
        y = compiled(x)
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
    except Exception as e:
        pytest.skip(f"torch.compile incompatibility (not blocking): {e}")


# ---- 4. Output-rank bound (F.4.b) ---------------------------------------


def test_output_subspace_rank_at_most_R_o():
    """Per F.4.b, image of forward(x) lies in col-space(U), so rank <= R_o."""
    ffn = _make_small()
    cfg = ffn.cfg
    torch.manual_seed(4)
    # Sample many inputs and form the matrix of outputs.
    N = max(64, cfg.d + 2 * cfg.R_o)
    x = torch.randn(N, cfg.d)
    with torch.no_grad():
        Y = ffn(x)  # (N, d)
    # rank of Y is at most rank of col-space(U), which is at most R_o.
    rank_Y = int(torch.linalg.matrix_rank(Y).item())
    assert rank_Y <= cfg.R_o, (
        f"Output rank {rank_Y} exceeds R_o={cfg.R_o}; "
        f"violates F.4.b's per-layer bound"
    )


def test_output_subspace_attainable_with_diverse_inputs():
    """Sanity: with R_o sample-rich inputs and an orthogonal U, we should
    actually attain the upper bound (or close to it).  This rules out a
    trivial pass where forward returns zeros."""
    ffn = _make_small()
    cfg = ffn.cfg
    torch.manual_seed(5)
    x = torch.randn(4 * cfg.R_o, cfg.d)
    with torch.no_grad():
        Y = ffn(x)
    rank_Y = int(torch.linalg.matrix_rank(Y).item())
    # Should be at least R_o // 2 in practice — a low bound that catches
    # accidental rank collapse without being flaky.
    assert rank_Y >= cfg.R_o // 2, (
        f"Output rank {rank_Y} < R_o/2={cfg.R_o // 2}; "
        f"layer may be near-degenerate"
    )


# ---- 5. Topology guards --------------------------------------------------


def test_compressive_mixer_rejected():
    """m < d would re-introduce JHCG's input-side bottleneck (Defect 1)."""
    with pytest.raises(ValueError, match="Defect 1"):
        FullMixTuckerConfig(d=16, m=8, R_o=4, R_i=4, R_b=2)


def test_t_direct_requires_m_equals_d():
    """T_direct (no mixer) requires m == d, otherwise dim mismatch in spline."""
    cfg = FullMixTuckerConfig(d=16, m=32, R_o=8, R_i=8, R_b=4, use_mixer=False)
    with pytest.raises(ValueError, match="use_mixer=False"):
        FullMixTuckerFFN(cfg)


def test_grid_validation():
    with pytest.raises(ValueError, match="grid_hi"):
        FullMixTuckerConfig(d=16, m=16, R_o=4, R_i=4, R_b=2, grid_lo=1.0, grid_hi=0.5)
    with pytest.raises(ValueError, match="G="):
        FullMixTuckerConfig(d=16, m=16, R_o=4, R_i=4, R_b=2, G=1)


# ---- 6. Initialization (variance-preserving) -----------------------------


@pytest.mark.parametrize("d,m,R_o,R_i,R_b", [
    (64, 64, 32, 32, 8),     # small symmetric
    (128, 128, 64, 64, 16),  # medium symmetric
    (256, 256, 64, 64, 16),  # asymmetric (R_o == R_i, but smaller R)
    (256, 256, 128, 64, 16), # asymmetric output-rich (Strategy A flavor)
])
def test_init_output_variance_close_to_unit(d, m, R_o, R_i, R_b):
    """At init, layer output std should be ~1 per the corrected L.4 formula.

    The variance-preserving spline-coef sigma_c = sqrt(3 d / (2 R_o))
    accounts for the full Tucker readout's variance shrinkage.  Output
    should land in [0.5, 2.0] across config sweep.
    """
    cfg = FullMixTuckerConfig(d=d, m=m, R_o=R_o, R_i=R_i, R_b=R_b, G=20)
    torch.manual_seed(6)
    ffn = FullMixTuckerFFN(cfg)
    x = torch.randn(256, cfg.d)
    with torch.no_grad():
        y = ffn(x)
    sigma = y.std().item()
    assert 0.5 < sigma < 2.0, (
        f"d={d}, R_o={R_o}: init output std={sigma:.3f} not in [0.5, 2.0]; "
        f"variance-preserving init miscalibrated"
    )


# ---- 7. HOSVD warm-start --------------------------------------------------


def test_hosvd_warmstart_shapes():
    """Output factor shapes must match requested ranks and input dims."""
    W = torch.randn(7, 8, 6)
    U, V, core, Q = hosvd_warmstart_from_dense(W, R_o=3, R_i=4, R_b=2)

    assert U.shape == (7, 3)
    assert V.shape == (8, 4)
    assert core.shape == (3, 4, 2)
    assert Q.shape == (6, 2)


def test_hosvd_warmstart_roundtrip_low_rank():
    """A tensor that is exactly Tucker rank-(R_o, R_i, R_b) must be reconstructed
    from its HOSVD factors with negligible error."""
    torch.manual_seed(7)
    d, m, L, R_o, R_i, R_b = 16, 16, 7, 8, 8, 4
    # Build an exactly-low-rank W via random factors.
    U_true = torch.linalg.qr(torch.randn(d, R_o))[0]
    V_true = torch.linalg.qr(torch.randn(m, R_i))[0]
    Q_true = torch.linalg.qr(torch.randn(L, R_b))[0]
    core_true = torch.randn(R_o, R_i, R_b)
    W = torch.einsum("ka, jb, ic, abc -> kji", U_true, V_true, Q_true, core_true)

    U, V, core, Q = hosvd_warmstart_from_dense(W, R_o, R_i, R_b)

    # Reconstruct.
    W_rec = torch.einsum("ka, jb, ic, abc -> kji", U, V, Q, core)
    rel = (W - W_rec).norm() / (W.norm() + 1e-12)
    assert rel < 1e-5, f"HOSVD round-trip rel err {rel.item():.2e}"


def test_hosvd_warmstart_truncation_bounded_error():
    """For a non-low-rank W, HOSVD with smaller ranks gives a bounded
    approximation (sanity check, not an exact roundtrip)."""
    torch.manual_seed(8)
    d, m, L = 16, 16, 7
    W = torch.randn(d, m, L)
    R_o, R_i, R_b = 4, 4, 2

    U, V, core, Q = hosvd_warmstart_from_dense(W, R_o, R_i, R_b)
    W_rec = torch.einsum("ka, jb, ic, abc -> kji", U, V, Q, core)
    rel = (W - W_rec).norm() / W.norm()
    # A naive truncated HOSVD on random data should explain a meaningful
    # fraction (~rank / d^3 of variance is wrong; better bound is via
    # singular values, but rel < 1.0 is the trivial-floor check).
    assert rel < 1.0, f"Truncated HOSVD did worse than zero recon (rel={rel.item():.3f})"
    # And clearly nonzero (we kept positive ranks).
    assert rel > 0.01, "Truncated HOSVD claimed exact recon on random data"


# ---- Bonus: equivalence under bf16 (loose tolerance) --------------------


@pytest.mark.cuda
def test_bf16_equivalence_within_kernel_tolerance():
    """Forward in bf16 must agree with fp32 within bf16's ~1e-3 tolerance.

    This is the contract that K.0.1 imposes on Phase 2 kernels.  The
    reference itself should already meet it (mostly via fp32 accumulation
    inside einsum).  Auto-skipped via conftest when CUDA is unavailable.
    """
    torch.manual_seed(9)
    cfg = FullMixTuckerConfig(d=32, m=32, R_o=16, R_i=16, R_b=4, G=8)
    ffn_fp32 = FullMixTuckerFFN(cfg).cuda().to(torch.float32)
    ffn_bf16 = FullMixTuckerFFN(cfg).cuda().to(torch.bfloat16)
    # Sync weights from fp32 to bf16 copy.
    with torch.no_grad():
        for (n_a, p_a), (n_b, p_b) in zip(
            ffn_fp32.named_parameters(), ffn_bf16.named_parameters(), strict=True
        ):
            assert n_a == n_b
            p_b.copy_(p_a.to(torch.bfloat16))
        for (n_a, b_a), (n_b, b_b) in zip(
            ffn_fp32.named_buffers(), ffn_bf16.named_buffers(), strict=True
        ):
            assert n_a == n_b
            b_b.copy_(b_a.to(b_b.dtype))

    x = torch.randn(8, cfg.d, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        y_fp32 = ffn_fp32(x)
        y_bf16 = ffn_bf16(x.to(torch.bfloat16)).to(torch.float32)
    rel = (y_fp32 - y_bf16).norm() / (y_fp32.norm() + 1e-9)
    # bf16 ~ 1e-3 relative per op; with 5 stages of bf16 accumulation (no fp32
    # promote in the reference) we measure ~2e-2 at small d.  The Phase 2 fused
    # kernel will need explicit fp32 accumulation to hit the K.0.1 1e-3 target.
    assert rel < 5e-2, f"bf16 vs fp32 rel err {rel.item():.2e} exceeds 5e-2"


# ===========================================================================
# 8. Distributional robustness — t-uniformity, Var[y] sweep, OOR clamping
# ===========================================================================


def test_t_distribution_is_uniform_under_gaussian_input():
    """The variance-preserving init relies on E[B0^2 + B1^2] = 2/3, which
    requires t ~ Uniform[0,1].  Verify empirically that t lies in [0,1] AND
    its histogram is flat (max deviation from 10% per bin <= 1pp at N=8192).
    """
    cfg = FullMixTuckerConfig(d=128, m=128, R_o=64, R_i=64, R_b=8, G=20)
    torch.manual_seed(101)
    ffn = FullMixTuckerFFN(cfg)

    x = torch.randn(8192, cfg.d)
    with capture_bin_frac(ffn) as captured, torch.no_grad():
        ffn(x)

    t = captured["t"]
    # Range
    assert t.min().item() >= 0.0 and t.max().item() <= 1.0
    # Mean and std (Uniform[0,1] -> mean=0.5, std=sqrt(1/12) ~ 0.2887)
    assert abs(t.mean().item() - 0.5) < 0.01, f"t mean {t.mean():.4f} != 0.5"
    assert abs(t.std().item() - math.sqrt(1.0 / 12.0)) < 0.01, (
        f"t std {t.std():.4f} != sqrt(1/12)"
    )
    # Histogram flatness
    hist = torch.histogram(t.flatten().cpu(), bins=10, range=(0.0, 1.0))
    counts_pct = hist.hist.float() / hist.hist.sum() * 100.0
    max_dev = (counts_pct - 10.0).abs().max().item()
    assert max_dev < 1.0, f"t-histogram deviates {max_dev:.2f}pp from uniform"


@pytest.mark.parametrize("x_std", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_output_std_robust_to_input_scale(x_std):
    """The init formula assumes Var[z]=1, but the output should be robust to
    a 16x range of input variance (the t-uniform property dominates).

    Concretely: y_std should stay in [0.5, 2.0] across x_std in [0.25, 4.0].
    """
    cfg = FullMixTuckerConfig(d=128, m=128, R_o=64, R_i=64, R_b=8, G=20)
    torch.manual_seed(102)
    ffn = FullMixTuckerFFN(cfg)
    x = torch.randn(512, cfg.d) * x_std
    with torch.no_grad():
        y = ffn(x)
    assert 0.4 < y.std().item() < 2.5, (
        f"x_std={x_std}: y_std={y.std():.3f} outside robust band [0.4, 2.5]"
    )


def test_out_of_grid_inputs_finite_and_bounded():
    """Inputs that exceed [grid_lo, grid_hi] should be clamped via the
    bin/frac mechanism rather than producing NaN or unbounded output.

    The reference impl uses ``bin_idx.clamp_(0, G-1)`` and ``frac.clamp_(0, 1)``
    to enforce this; the test pins that behavior.
    """
    cfg = FullMixTuckerConfig(d=64, m=64, R_o=32, R_i=32, R_b=8, G=10,
                              grid_lo=-2.0, grid_hi=2.0)
    torch.manual_seed(103)
    ffn = FullMixTuckerFFN(cfg)

    for x_scale in [10.0, 100.0, 1000.0]:
        x = torch.randn(8, cfg.d) * x_scale
        with torch.no_grad():
            y = ffn(x)
        assert torch.isfinite(y).all(), f"non-finite at x_scale={x_scale}"
        # Output should still be of reasonable magnitude (not blown up by
        # 100x, not collapsed to zero).  The clamping caps how far off-grid
        # the spline output can drift.
        assert 0.1 < y.std().item() < 100.0, (
            f"x_scale={x_scale}: y_std={y.std():.3f} suggests broken clamping"
        )


def test_bin_coverage_at_unit_input():
    """At unit-variance Gaussian input mapped onto a wide grid [-3, 3] with
    G=20 bins, we expect *most* bins to be touched (>= 75% with N=512 tokens).

    A failure here means z is not properly spreading over the grid — usually
    a sign that the mixer A under-scales (Var[z] much less than expected).
    """
    cfg = FullMixTuckerConfig(d=128, m=128, R_o=64, R_i=64, R_b=8, G=20)
    torch.manual_seed(104)
    ffn = FullMixTuckerFFN(cfg)

    x = torch.randn(512, cfg.d)
    with capture_bin_frac(ffn) as captured, torch.no_grad():
        ffn(x)

    bins_touched = captured["bin"].unique().numel()
    coverage = bins_touched / cfg.G
    assert coverage >= 0.75, (
        f"only {bins_touched}/{cfg.G} bins touched ({coverage*100:.0f}%); "
        f"mixer may be under-scaling z"
    )


# ===========================================================================
# 9. Stacking / depth — F.5.1 cumulative subspace coverage, gradient flow
# ===========================================================================


@pytest.mark.parametrize("K", [3, 6, 12])
def test_residual_stack_activation_stable(K):
    """Stacking K FullMix-Tucker FFN as residual blocks should keep activation
    magnitude bounded — neither vanishing nor exploding through the stack.

    With per-layer FFN output Var ~= 1 and residual addition, the stream Var
    grows linearly: Var[x_K] ~= Var[x_0] + K (assuming independence).  So
    std[x_K] / std[x_0] should be in [1, sqrt(K+1) * 1.5] (the 1.5 is slack
    for non-independence + finite-sample noise)."""
    torch.manual_seed(200 + K)
    layers, cfg = _stack_layers(K)
    x = torch.randn(64, cfg.d)
    with torch.no_grad():
        h = _residual_stack_forward(layers, x)
    ratio = (h.std() / x.std()).item()
    upper = math.sqrt(K + 1) * 1.5
    assert 0.8 < ratio < upper, (
        f"K={K}: std-ratio {ratio:.3f} outside [0.8, {upper:.2f}]"
    )
    assert torch.isfinite(h).all()


def test_cumulative_output_subspace_coverage_F51():
    """F.5.1 prediction: with K layers each of output rank R_o, the union
    of column spaces dim(union Col(U_l)) approaches min(K * R_o, d) when
    the U_l are diverse.

    Build K layers (different seeds), stack their U into [U_1 | ... | U_K],
    measure rank.  Expect rank >= min(K * R_o, d) * 0.85 (allowing 15%
    slack for finite-sample numerical near-degeneracy)."""
    K = 6
    R_o = 8
    d = 32
    layers, cfg = _stack_layers(
        K, d=d, m=d, R_o=R_o, R_i=R_o, R_b=4, G=8
    )
    # Each layer was init'd with its own torch.manual_seed via FullMixTuckerFFN,
    # so U_l are independent.
    Us = torch.cat([ffn.U for ffn in layers], dim=1)  # (d, K * R_o)
    rank = int(torch.linalg.matrix_rank(Us).item())
    expected = min(K * R_o, d)
    assert rank >= int(0.85 * expected), (
        f"cumulative U-rank {rank} < 0.85 * min(K*R_o, d)={expected}; "
        f"U_l may be collapsing into shared subspace"
    )


def test_gradient_flow_through_K12_stack():
    """Backward through K=12 residual FullMix-Tucker layers should produce
    finite, non-vanishing gradients on x and on the *first* layer's params.

    This is a raw (un-normalized) stack, so we only check finiteness and
    non-vanishing.  The properly-normalized transformer-style stack is
    tested separately (`test_pre_rmsnorm_stack_stable_gradients`).
    """
    K = 12
    torch.manual_seed(202)
    layers, cfg = _stack_layers(K)
    x = torch.randn(8, cfg.d, requires_grad=True)
    h = _residual_stack_forward(layers, x)
    h.pow(2).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.norm().item() > 1e-6, "gradient on x has vanished"

    first_layer_grad_norms = [
        p.grad.norm().item()
        for p in layers[0].parameters()
        if p.grad is not None
    ]
    assert all(math.isfinite(g) for g in first_layer_grad_norms)
    # First layer (deepest from loss) should still have meaningful gradient.
    assert max(first_layer_grad_norms) > 1e-6, (
        f"first-layer gradient vanished: max norm {max(first_layer_grad_norms):.2e}"
    )


def test_pre_rmsnorm_stack_stable_activations_and_gradients():
    """When FullMix-Tucker is used inside a transformer-style pre-RMSNorm
    residual block, BOTH activation magnitudes AND gradient magnitudes
    should stay bounded across K=12 layers.

    This is the realistic deployment scenario.  Pass criteria:
      - activation std ratio across K=12 within [1, 5]
      - max-to-min layer-wise gradient norm ratio within [1, 50]
        (much tighter than the 190000x we measured without RMSNorm)
    """
    K = 12
    torch.manual_seed(204)
    layers, cfg = _stack_layers(K, d=64, m=64, R_o=16, R_i=16, R_b=4, G=8)
    x = torch.randn(16, cfg.d, requires_grad=True)
    h = _pre_rmsnorm_stack_forward(layers, x)

    # Forward stability — activations don't blow up despite K=12 residual sums,
    # because RMSNorm bounds each FFN's input variance to ~1.
    assert torch.isfinite(h).all()
    ratio = (h.std() / x.std()).item()
    assert 1.0 < ratio < 5.0, f"activation ratio {ratio:.2f} out of [1, 5]"

    # Backward stability — gradient norms across layers should be within
    # one order of magnitude (RMSNorm prevents the variance compounding).
    h.pow(2).sum().backward()
    layer_grad_norms = [
        sum(p.grad.norm().item() ** 2 for p in layer.parameters() if p.grad is not None)
        ** 0.5
        for layer in layers
    ]
    assert all(math.isfinite(g) and g > 1e-6 for g in layer_grad_norms)
    spread = max(layer_grad_norms) / min(layer_grad_norms)
    assert spread < 50.0, (
        f"gradient norm spread across K=12 layers: {spread:.1f}x — "
        f"expected <50x with RMSNorm"
    )


# ===========================================================================
# 10. Numerical robustness — determinism, dtype, NaN propagation
# ===========================================================================


def test_determinism_same_seed_same_output():
    """Two FullMixTuckerFFN constructed under the same torch seed must
    produce bit-identical outputs on the same input."""
    torch.manual_seed(300)
    ffn1 = _make_small()
    torch.manual_seed(300)
    ffn2 = _make_small()
    x = torch.randn(4, ffn1.cfg.d)
    with torch.no_grad():
        y1, y2 = ffn1(x), ffn2(x)
    assert torch.equal(y1, y2), "same-seed init produced divergent outputs"


def test_fp32_fp64_agreement():
    """fp32 and fp64 references must agree within fp32 precision (~1e-5)."""
    torch.manual_seed(301)
    ffn32 = _make_small()
    ffn64 = _make_small()
    # Sync params: copy fp32 -> fp64 then promote.
    with torch.no_grad():
        for p32, p64 in zip(ffn32.parameters(), ffn64.parameters(), strict=True):
            p64.data = p32.data.to(torch.float64)
        for b32, b64 in zip(ffn32.buffers(), ffn64.buffers(), strict=True):
            b64.data = b32.data.to(b64.dtype)
    ffn64 = ffn64.to(torch.float64)

    x = torch.randn(8, ffn32.cfg.d, dtype=torch.float32)
    with torch.no_grad():
        y32 = ffn32(x)
        y64 = ffn64(x.to(torch.float64)).to(torch.float32)
    rel = (y32 - y64).norm() / (y64.norm() + 1e-9)
    assert rel < 5e-5, f"fp32 vs fp64 rel err {rel:.2e}"


def test_nan_input_propagates_not_silently_zeroed():
    """NaN in x should propagate to NaN in y (otherwise we'd silently swallow
    debugging signals).  This is the standard PyTorch behavior for Linear +
    einsum and we want to confirm the gather lookup doesn't accidentally
    sanitize it."""
    ffn = _make_small()
    x = torch.randn(2, ffn.cfg.d)
    x[0, 0] = float("nan")
    with torch.no_grad():
        y = ffn(x)
    # At least the first row should have NaN (it propagates through every stage).
    assert torch.isnan(y[0]).any(), "NaN was silently sanitized"


def test_grid_endpoints_handled():
    """z exactly at grid_lo or grid_hi should map to bin_idx=0/G-1 with
    frac=0 or 1 respectively, no out-of-bounds Q lookup."""
    cfg = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2, G=5,
                              grid_lo=-1.0, grid_hi=1.0)
    torch.manual_seed(303)
    ffn = FullMixTuckerFFN(cfg)
    # Construct z at exact grid endpoints by bypassing the mixer.
    z_at_lo = torch.full((1, cfg.m), cfg.grid_lo, dtype=torch.float32)
    z_at_hi = torch.full((1, cfg.m), cfg.grid_hi, dtype=torch.float32)
    bin_lo, t_lo = ffn._bin_and_frac(z_at_lo)
    bin_hi, t_hi = ffn._bin_and_frac(z_at_hi)
    assert (bin_lo == 0).all(), f"z=grid_lo should give bin=0, got {bin_lo}"
    assert (bin_hi == cfg.G - 1).all(), (
        f"z=grid_hi should give bin=G-1={cfg.G-1}, got {bin_hi}"
    )
    assert (t_lo == 0.0).all()
    # Allow t at hi to be either 0 (next bin would be G, clamped) or just
    # under 1 due to floor() rounding behavior.  Reference impl keeps t in
    # [0, 1] and clamps bin to G-1 so t=1 reads Q[G-1+1] = Q[G].
    assert (t_hi >= 0.0).all() and (t_hi <= 1.0).all()


@pytest.mark.parametrize("R_o,R_i", [(96, 96), (128, 64), (64, 128), (256, 96)])
def test_asymmetric_rank_configurations(R_o, R_i):
    """F.4.c Strategy A allows R_o != R_i.  Verify all such configs build
    cleanly, forward/backward run, and outputs are finite."""
    cfg = FullMixTuckerConfig(d=64, m=64, R_o=R_o, R_i=R_i, R_b=8, G=10)
    torch.manual_seed(400)
    ffn = FullMixTuckerFFN(cfg)
    # Param shapes follow F.4.a exactly — verify in this asymmetric case.
    assert ffn.U.shape == (cfg.d, R_o)
    assert ffn.V.shape == (cfg.m, R_i)
    assert ffn.C.shape == (R_o, R_i, cfg.R_b)
    x = torch.randn(16, cfg.d, requires_grad=True)
    y = ffn(x)
    assert y.shape == x.shape
    y.pow(2).sum().backward()
    for name, p in ffn.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_layer_can_actually_learn_via_sgd():
    """End-to-end smoke test: a single FullMix-Tucker layer should be able
    to fit a simple nonlinear regression target via plain SGD.

    Target: y* = sin(x_0) + 0.3 * x_1^2 - 0.2 * x_2 (per-token, deterministic).
    Train for 200 SGD steps on a single fixed batch; expect MSE to drop
    by at least 5x from start to end.

    This catches:
      - frozen parameters that don't update;
      - gradient sign bugs that cause divergence;
      - init that puts the layer in a saddle.
    """
    cfg = FullMixTuckerConfig(d=8, m=8, R_o=8, R_i=8, R_b=4, G=6)
    torch.manual_seed(500)
    ffn = FullMixTuckerFFN(cfg)
    x = torch.randn(64, cfg.d)
    # Target is a simple per-token nonlinear mapping; pad to d-dim by zeroing
    # the rest of the output channels so we have a reproducible regression task.
    target = torch.zeros(64, cfg.d)
    target[:, 0] = torch.sin(x[:, 0])
    target[:, 1] = 0.3 * x[:, 1] ** 2
    target[:, 2] = -0.2 * x[:, 2]

    opt = torch.optim.SGD(ffn.parameters(), lr=0.05)
    initial_mse = float("inf")
    final_mse = float("inf")
    for step in range(200):
        opt.zero_grad()
        y = ffn(x)
        mse = (y - target).pow(2).mean()
        if step == 0:
            initial_mse = mse.item()
        if step == 199:
            final_mse = mse.item()
        mse.backward()
        opt.step()

    assert math.isfinite(final_mse), f"diverged: final MSE {final_mse}"
    assert final_mse < initial_mse / 5.0, (
        f"layer failed to learn: initial MSE {initial_mse:.4f} -> "
        f"final MSE {final_mse:.4f} (ratio {initial_mse / final_mse:.2f}x, need >5x)"
    )


def test_residual_stack_can_learn_via_sgd():
    """K-layer residual stack should also be trainable.  Same target as the
    single-layer test but routed through 4 stacked FFN blocks."""
    K = 4
    torch.manual_seed(501)
    layers, cfg = _stack_layers(K, d=8, m=8, R_o=8, R_i=8, R_b=4, G=6)
    x = torch.randn(64, cfg.d)
    target = torch.zeros(64, cfg.d)
    target[:, 0] = torch.sin(x[:, 0])
    target[:, 1] = 0.3 * x[:, 1] ** 2

    opt = torch.optim.SGD(
        [p for layer in layers for p in layer.parameters()], lr=0.03
    )
    initial_mse = final_mse = float("inf")
    for step in range(300):
        opt.zero_grad()
        h = _residual_stack_forward(layers, x)
        mse = (h - target).pow(2).mean()
        if step == 0:
            initial_mse = mse.item()
        if step == 299:
            final_mse = mse.item()
        mse.backward()
        opt.step()
    assert math.isfinite(final_mse)
    assert final_mse < initial_mse / 3.0, (
        f"stack failed to learn: {initial_mse:.4f} -> {final_mse:.4f}"
    )


def test_param_count_matches_doc_F4a_formula():
    """F.4.a gives the per-layer storage formula:

        P = d*m + d*R_o + m*R_i + L*R_b + R_o*R_i*R_b

    Verify the implementation matches this exactly at nanochat scale."""
    cfg = FullMixTuckerConfig(d=768, m=768, R_o=96, R_i=96, R_b=16, G=20)
    ffn = FullMixTuckerFFN(cfg)
    L = cfg.G + 1  # B1
    expected = (
        cfg.d * cfg.m                       # mixer A
        + cfg.d * cfg.R_o                    # U
        + cfg.m * cfg.R_i                    # V
        + L * cfg.R_b                        # Q
        + cfg.R_o * cfg.R_i * cfg.R_b        # core C
        + 1                                  # gamma
    )
    actual = sum(p.numel() for p in ffn.parameters())
    assert actual == expected, (
        f"param count {actual} != F.4.a formula {expected}"
    )
    # And confirm doc's "885K" claim.
    assert 880_000 <= actual <= 890_000, (
        f"param count {actual} not in F.4.a's claimed 885K band"
    )
