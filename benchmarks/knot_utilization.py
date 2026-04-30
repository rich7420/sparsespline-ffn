"""Knot utilization diagnostic.

Diagnoses two failure modes that show up as "training looks fine but eval
quality stalls":

  1. Dead knots: rows of Q (the spline-mode lookup table) that never
     receive a gradient because no token's mixed activation z_j ever
     falls into the corresponding bin.  L.4 hints this can happen when
     the grid is too wide for Var[z].
  2. Hot knots: a tiny fraction of Q rows accumulating most of the
     gradient signal — usually a sign the grid is too narrow and tokens
     pile up in a few bins (clamp_-saturation symptom).

We run a short SGD pass on a synthetic target and report:

  - bin coverage (fraction of bins ever touched);
  - per-row Q gradient magnitude distribution (min / median / p90 / max);
  - parameter-wise gradient L2 ratios across (Q, V, C, U, A) so we can
    see if any single tensor is hogging the signal.

Auto-detects CUDA; runs in fp32 for stable per-row gradient stats.
"""
from __future__ import annotations

import time

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def target(x):
    y = torch.zeros_like(x)
    d = x.shape[-1]
    for k in range(min(d, 6)):
        y[..., k] = 0.4 * torch.sin((k + 1) * x[..., k])
    return y


def main():
    device = _device()
    d = 64
    R_o = R_i = 16
    R_b = 8
    G = 20
    steps = 200
    lr = 3e-3

    print("=" * 78)
    print("Knot utilization diagnostic")
    print(f"device={device}, d={d}, R_o={R_o}, R_b={R_b}, G={G}, "
          f"steps={steps}")
    print("=" * 78)

    torch.manual_seed(0)
    cfg = FullMixTuckerConfig(d=d, m=d, R_o=R_o, R_i=R_i, R_b=R_b, G=G,
                              grid_lo=-3.0, grid_hi=3.0)
    fm = FullMixTuckerFFN(cfg).to(device).float()

    # -- Bin coverage (one-pass forward) --
    g = torch.Generator(device=device).manual_seed(7)
    x_probe = torch.randn(2048, d, device=device, generator=g)
    bins_seen = torch.zeros(G, dtype=torch.long, device=device)
    orig = fm._bin_and_frac

    def spy(z):
        bin_idx, t = orig(z)
        flat = bin_idx.reshape(-1)
        bins_seen.index_add_(0, flat, torch.ones_like(flat))
        return bin_idx, t

    fm._bin_and_frac = spy  # type: ignore[method-assign]
    with torch.no_grad():
        fm(x_probe)
    fm._bin_and_frac = orig  # restore

    coverage = (bins_seen > 0).float().mean().item()
    busiest = bins_seen.float() / bins_seen.sum()
    top_share = busiest.max().item()
    bottom_share = busiest.min().item()
    print(f"\nBin coverage at init: {100*coverage:.1f}% of {G} bins seen")
    print(f"  busiest bin gets {100*top_share:.2f}% of all token-dim hits")
    print(f"  emptiest bin gets {100*bottom_share:.2f}%")

    # -- Per-row Q gradient distribution --
    x = torch.randn(1024, d, device=device, generator=g)
    y = target(x)
    opt = torch.optim.Adam(fm.parameters(), lr=lr)

    Q_grad_sum = torch.zeros_like(fm.Q)  # (L, R_b)
    grad_l2 = {n: 0.0 for n, _ in fm.named_parameters()}

    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        (fm(x) - y).pow(2).mean().backward()
        # Accumulate gradient stats BEFORE step (otherwise zero_grad clears).
        Q_grad_sum += fm.Q.grad.detach().abs()
        for n, p in fm.named_parameters():
            if p.grad is not None:
                grad_l2[n] += p.grad.detach().pow(2).sum().sqrt().item()
        opt.step()
    elapsed = time.perf_counter() - t0

    # Per-Q-row magnitude (sum over R_b columns).
    row_mag = Q_grad_sum.sum(dim=1).cpu()
    row_sorted = torch.sort(row_mag).values
    L = row_sorted.numel()
    print(f"\nPer-Q-row |grad| accumulated over {steps} steps "
          f"(L = G+1 = {L}):")
    print(f"  min     : {row_sorted[0].item():.4e}")
    print(f"  p10     : {row_sorted[max(0, L // 10)].item():.4e}")
    print(f"  median  : {row_sorted[L // 2].item():.4e}")
    print(f"  p90     : {row_sorted[min(L - 1, (9 * L) // 10)].item():.4e}")
    print(f"  max     : {row_sorted[-1].item():.4e}")

    dead = int((row_mag == 0).sum().item())
    print(f"  rows with exactly zero accumulated gradient: {dead}/{L}")

    # -- Parameter-wise gradient share --
    print(f"\nGradient L2 share across parameters (sum over {steps} steps):")
    total = sum(grad_l2.values())
    rows = sorted(grad_l2.items(), key=lambda kv: -kv[1])
    for n, v in rows:
        share = 100 * v / max(total, 1e-12)
        print(f"  {n:<20} {v:>12.4e}  ({share:5.1f}%)")

    print(f"\nWall: {elapsed:.1f}s")
    print("\nDiagnostics:")
    if dead > 0:
        print(f"  WARNING: {dead} Q rows received zero gradient — "
              "grid_lo/grid_hi likely too wide for Var[z]; tighten the grid.")
    if top_share > 0.5:
        print("  WARNING: one bin captures >50% of token-dim hits — "
              "grid_lo/grid_hi likely too narrow; widen the grid.")
    if dead == 0 and top_share < 0.4:
        print("  OK: all knots active and coverage is roughly uniform.")


if __name__ == "__main__":
    main()
