"""Quality benchmark: Jacobian, Lipschitz, and effective-rank analysis.

For each FFN type at d=128 (smaller than nanochat for Jacobian feasibility,
larger than the regression bench), we compute:

  1. Jacobian rank distribution
       Per-input Jacobian J(x) ∈ R^{d × d}.  Rank tells us the local effective
       output dim.  For FullMix at R_o=R_i, we expect rank <= R_o.

  2. Lipschitz constant lower bound (top singular value of J)
       max_i ||J(x_i)||_2  over a sample.  Higher = more sensitive layer;
       very low = oversmoothing.

  3. Effective rank (entropy of singular values, via exp(H))
       Ranks "how much rank is actually used" — robust to numerical noise.

  4. Output covariance rank
       For 1024 random inputs Y = forward(X), rank(Y).  Bounded by R_o for
       FullMix per F.4.b.

We compare FullMix-Tucker (form B) against MLPFFN at matched param budget.
"""
from __future__ import annotations

import statistics
import time

import torch

from sparsespline_ffn import MLPFFN, FullMixTuckerConfig, FullMixTuckerFFN


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build_models(d: int, R_o: int = 32, R_i: int = 32, R_b: int = 8, G: int = 16):
    cfg = FullMixTuckerConfig(d=d, m=d, R_o=R_o, R_i=R_i, R_b=R_b, G=G)
    fm = FullMixTuckerFFN(cfg)
    fm_p = count_params(fm)
    best_r = 1
    for r in [1, 2, 3, 4, 6, 8]:
        if 2 * d * (r * d) <= fm_p:
            best_r = r
    mlp = MLPFFN(d=d, mlp_ratio=best_r)
    mlp_p = count_params(mlp)
    return fm, mlp, fm_p, mlp_p, best_r


def jacobian_singular_values(
    model: torch.nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """Compute SVs of J(x) ∈ R^{d×d} via torch.func.jacrev for one row of x."""
    from torch.func import jacrev

    def f(x_one):
        return model(x_one.unsqueeze(0)).squeeze(0)

    J = jacrev(f)(x)
    return torch.linalg.svdvals(J)


def effective_rank_from_svs(svs: torch.Tensor, eps: float = 1e-12) -> float:
    """Effective rank = exp(entropy of normalized squared SVs)."""
    s2 = svs.pow(2)
    s2 = s2[s2 > eps]
    if s2.numel() == 0:
        return 0.0
    p = s2 / s2.sum()
    H = -(p * p.log()).sum()
    return float(torch.exp(H).item())


def hard_rank_from_svs(svs: torch.Tensor, rel_tol: float = 1e-4) -> int:
    """Rank counting SVs above rel_tol * max(SV).

    rel_tol=1e-4 cuts above fp32 noise floor (Jacobian-of-tucker numerically
    has rank-deficient SVs around 1e-5 to 1e-7 of max in fp32)."""
    if svs.numel() == 0:
        return 0
    cutoff = rel_tol * svs.max().item()
    return int((svs > cutoff).sum().item())


def output_rank(model: torch.nn.Module, X: torch.Tensor, rel_tol: float = 1e-5) -> int:
    with torch.no_grad():
        Y = model(X)
    return int(torch.linalg.matrix_rank(Y, tol=rel_tol * Y.abs().max()).item())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = 128
    seeds = [0, 1, 2]
    n_jacobian_samples = 32
    n_output_samples = 1024

    print("=" * 78)
    print("Jacobian / Lipschitz / Effective-rank analysis")
    print(f"device={device}, d={d}, seeds={seeds}")
    print("=" * 78)

    summary_fm: dict[str, list[float]] = {
        "lip": [], "eff_rank": [], "hard_rank": [], "out_rank": []
    }
    summary_mlp: dict[str, list[float]] = {
        "lip": [], "eff_rank": [], "hard_rank": [], "out_rank": []
    }
    fm_p_seen = mlp_p_seen = best_r_seen = None

    t0 = time.perf_counter()
    for seed in seeds:
        torch.manual_seed(seed)
        fm, mlp, fm_p, mlp_p, best_r = build_models(d)
        fm_p_seen = fm_p
        mlp_p_seen = mlp_p
        best_r_seen = best_r
        fm = fm.to(device).eval()
        mlp = mlp.to(device).eval()

        g = torch.Generator(device=device).manual_seed(seed + 100)
        X = torch.randn(n_output_samples, d, device=device, generator=g)
        x_jac = X[:n_jacobian_samples]

        for _name, model, summary in [("FullMix", fm, summary_fm),
                                      ("MLP", mlp, summary_mlp)]:
            lips: list[float] = []
            eff_ranks: list[float] = []
            hard_ranks: list[int] = []
            for i in range(n_jacobian_samples):
                svs = jacobian_singular_values(model, x_jac[i])
                lips.append(svs.max().item())
                eff_ranks.append(effective_rank_from_svs(svs))
                hard_ranks.append(hard_rank_from_svs(svs))
            o_rank = output_rank(model, X)

            summary["lip"].append(statistics.mean(lips))
            summary["eff_rank"].append(statistics.mean(eff_ranks))
            summary["hard_rank"].append(statistics.mean(hard_ranks))
            summary["out_rank"].append(o_rank)

    print(f"\nBudget: FullMix params = {fm_p_seen:,}, "
          f"MLP params = {mlp_p_seen:,} (mlp_ratio={best_r_seen}, "
          f"d={d})")
    print(f"\n{'metric':<32} {'FullMix':>20} {'MLP':>20}")
    print("-" * 75)
    for key, label in [
        ("lip", "Lipschitz (mean ||J||_2)"),
        ("eff_rank", "Effective rank exp(H(p))"),
        ("hard_rank", f"Hard rank (rel_tol=1e-4, max d={d})"),
        ("out_rank", f"Output rank (1024 inputs, max d={d})"),
    ]:
        fm_vals = summary_fm[key]
        mlp_vals = summary_mlp[key]
        fm_str = (f"{statistics.mean(fm_vals):.3f} ± "
                  f"{statistics.stdev(fm_vals) if len(fm_vals) > 1 else 0:.3f}")
        mlp_str = (f"{statistics.mean(mlp_vals):.3f} ± "
                   f"{statistics.stdev(mlp_vals) if len(mlp_vals) > 1 else 0:.3f}")
        print(f"{label:<32} {fm_str:>20} {mlp_str:>20}")

    elapsed = time.perf_counter() - t0
    print(f"\nWall: {elapsed:.1f}s")
    print("\nInterpretation:")
    print(" - Output rank for FullMix should saturate at R_o (here R_o=32);")
    print("   MLP can in principle reach min(d, hidden) but in practice activation")
    print("   nonlinearity often reduces effective rank.")
    print(" - Lipschitz: lower = smoother layer.  Very large = unstable; near 0 = collapsed.")
    print(" - Effective rank exp(H) measures how 'isotropic' the local Jacobian is;")
    print("   it is robust to small SVs that cumulative rank counting misses.")


if __name__ == "__main__":
    main()
