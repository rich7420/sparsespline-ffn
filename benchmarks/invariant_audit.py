"""Production-scale invariant audit for FullMix-Tucker FFN.

Re-runs the most important invariants from JHCG_REDESIGN_THEORY.md at
*nanochat scale* (d=768, m=768, R_o=96), where unit tests are too slow.

Each check prints PASS/FAIL with a short summary.  Exits non-zero on any
failure so CI can use this as a smoke gate.

Checks:
  1. F.4.b   per-layer output rank <= R_o
  2. F.5.1   K=12 cumulative col-space union approaches d
  3. L.4     init output std in [0.5, 2.0]
  4. K.0.1   5-stage matches dense W (small-batch sample, since dense W is
              26MB and the dense-evaluation path is N x d x m x L = blows up
              at full batch)
  5. distrib t ~ Uniform[0, 1] under Gaussian input
  6. F.4.a   param count matches closed-form
  7. autograd full forward+backward finite at production batch size
"""
from __future__ import annotations

import math
import sys
import time

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _check(name: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {name}{('  -- ' + detail) if detail else ''}")
    return ok


def _make_prod() -> FullMixTuckerFFN:
    cfg = FullMixTuckerConfig(d=768, m=768, R_o=96, R_i=96, R_b=16, G=20)
    torch.manual_seed(0)
    return FullMixTuckerFFN(cfg)


def audit_output_rank() -> bool:
    """F.4.b: rank(forward(X)) <= R_o."""
    print("\n[F.4.b] Per-layer output rank bound")
    ffn = _make_prod()
    cfg = ffn.cfg
    torch.manual_seed(1)
    x = torch.randn(4 * cfg.R_o, cfg.d)
    with torch.no_grad():
        Y = ffn(x)
    rank = int(torch.linalg.matrix_rank(Y).item())
    bound_ok = rank <= cfg.R_o
    nontrivial_ok = rank >= cfg.R_o // 2
    return all([
        _check(f"rank(Y) = {rank} <= R_o = {cfg.R_o}", bound_ok),
        _check(f"rank(Y) = {rank} >= R_o/2 = {cfg.R_o // 2} "
               f"(non-degenerate)", nontrivial_ok),
    ])


def audit_cumulative_subspace() -> bool:
    """F.5.1: K=12 stack cumulative U-rank approaches min(K * R_o, d)."""
    print("\n[F.5.1] Cumulative U col-space coverage at K=12")
    K = 12
    d, R_o = 768, 96
    cfg = FullMixTuckerConfig(d=d, m=d, R_o=R_o, R_i=R_o, R_b=16, G=20)
    layers = [FullMixTuckerFFN(cfg) for _ in range(K)]
    Us = torch.cat([layer.U for layer in layers], dim=1)  # (d, K*R_o)
    rank = int(torch.linalg.matrix_rank(Us).item())
    expected = min(K * R_o, d)  # K*R_o = 1152, d = 768 -> bound = 768
    ok = rank >= int(0.95 * expected)  # tighter bound at production scale
    return _check(
        f"cumulative rank = {rank}, target = min(K*R_o, d) = {expected} "
        f"({100 * rank / expected:.1f}% of bound)", ok
    )


def audit_init_variance() -> bool:
    """L.4: at init, output std in [0.5, 2.0]."""
    print("\n[L.4] Variance-preserving init")
    ffn = _make_prod()
    torch.manual_seed(2)
    x = torch.randn(2048, ffn.cfg.d)
    with torch.no_grad():
        y = ffn(x)
    sigma = y.std().item()
    return _check(
        f"output std = {sigma:.3f} in [0.5, 2.0] "
        f"(target ~1.0 from sigma_c = sqrt(3d/(2 R_o)))",
        0.5 <= sigma <= 2.0
    )


def audit_5stage_dense_equivalence() -> bool:
    """K.0.1: 5-stage equals dense W path within fp32 tolerance.

    Use a small batch (N=8) since dense path's einsum allocates
    (N, m, R_b) * (d, m, L) intermediates."""
    print("\n[K.0.1] 5-stage vs dense W reconstruction")
    ffn = _make_prod()
    cfg = ffn.cfg
    torch.manual_seed(3)
    x = torch.randn(8, cfg.d)
    y_ref = ffn(x)
    with torch.no_grad():
        z = ffn.A(x)
        bin_idx, t = ffn._bin_and_frac(z)
        L = cfg.G + 1
        N = z.shape[0]
        B = torch.zeros(N, cfg.m, L)
        B.scatter_(2, bin_idx.unsqueeze(-1), (1.0 - t).unsqueeze(-1))
        B.scatter_(2, (bin_idx + 1).unsqueeze(-1), t.unsqueeze(-1))
        W = ffn.reconstruct_dense_W()
        y_direct = ffn.gamma * torch.einsum("kji, nji -> nk", W, B)
    rel = ((y_ref - y_direct).norm() / (y_direct.norm() + 1e-9)).item()
    return _check(
        f"rel err = {rel:.2e} < 1e-5 (fp32 tolerance)", rel < 1e-5
    )


def audit_t_uniform() -> bool:
    """t-distribution under Gaussian input is Uniform[0,1]."""
    print("\n[distrib] t ~ Uniform[0, 1]")
    ffn = _make_prod()
    captured: dict = {}
    orig = ffn._bin_and_frac

    def spy(z):  # noqa: ANN001
        bin_idx, t = orig(z)
        captured["t"] = t.detach().clone()
        captured["bin"] = bin_idx.detach().clone()
        return bin_idx, t

    ffn._bin_and_frac = spy  # type: ignore[method-assign]
    torch.manual_seed(4)
    x = torch.randn(8192, ffn.cfg.d)
    with torch.no_grad():
        ffn(x)
    t = captured["t"]
    mean_err = abs(t.mean().item() - 0.5)
    std_err = abs(t.std().item() - math.sqrt(1 / 12))
    bin_cov = captured["bin"].unique().numel() / ffn.cfg.G
    return all([
        _check(f"|mean - 0.5| = {mean_err:.4f} < 0.01", mean_err < 0.01),
        _check(f"|std - sqrt(1/12)| = {std_err:.4f} < 0.01", std_err < 0.01),
        _check(f"bin coverage = {100 * bin_cov:.1f}% >= 90%", bin_cov >= 0.9),
    ])


def audit_param_count() -> bool:
    """F.4.a: param count matches the closed-form."""
    print("\n[F.4.a] Param-count formula")
    ffn = _make_prod()
    cfg = ffn.cfg
    L = cfg.G + 1
    expected = (
        cfg.d * cfg.m
        + cfg.d * cfg.R_o
        + cfg.m * cfg.R_i
        + L * cfg.R_b
        + cfg.R_o * cfg.R_i * cfg.R_b
        + 1
    )
    actual = sum(p.numel() for p in ffn.parameters())
    return _check(
        f"actual = {actual:,}, formula = {expected:,} "
        f"({actual / 1e3:.0f}K, F.4.a's stated 885K)", actual == expected
    )


def audit_full_batch_autograd() -> bool:
    """Autograd at production batch size: B=4, T=2048, N=8192 tokens."""
    print("\n[autograd] forward+backward at N=8192 tokens")
    ffn = _make_prod()
    torch.manual_seed(5)
    x = torch.randn(4, 2048, ffn.cfg.d, requires_grad=True)
    t0 = time.perf_counter()
    y = ffn(x)
    fwd_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    y.pow(2).sum().backward()
    bwd_ms = (time.perf_counter() - t0) * 1000
    grads_finite = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in ffn.parameters()
    )
    x_grad_finite = x.grad is not None and torch.isfinite(x.grad).all()
    return all([
        _check(f"forward finite, fwd = {fwd_ms:.1f} ms",
               bool(torch.isfinite(y).all())),
        _check(f"backward finite (params), bwd = {bwd_ms:.1f} ms", grads_finite),
        _check("backward finite (input.grad)", bool(x_grad_finite)),
    ])


def main() -> int:
    print("=" * 78)
    print("FullMix-Tucker FFN — production-scale invariant audit")
    print("d=768, m=768, R_o=96, R_i=96, R_b=16, G=20")
    print("=" * 78)
    results = [
        audit_param_count(),
        audit_init_variance(),
        audit_output_rank(),
        audit_cumulative_subspace(),
        audit_5stage_dense_equivalence(),
        audit_t_uniform(),
        audit_full_batch_autograd(),
    ]
    print("\n" + "=" * 78)
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        print(f"  ALL {n_total} CHECKS PASSED")
        return 0
    print(f"  {n_pass}/{n_total} passed; {n_total - n_pass} failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
