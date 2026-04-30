"""Quality benchmark: grid resolution G sweep.

THEORY.md E.2 / I.2: B1 splines on G grid intervals over [grid_lo, grid_hi]
have G+1 sharp transitions per channel.  The frequency cliff for fitting
sin(omega * x) lies near omega ~ G / (grid_hi - grid_lo) cycles per unit.

We sweep G and check how the eval_mse on a fixed-frequency target evolves.
Expected: eval_mse drops sharply once G is large enough to resolve the
target frequency, then plateaus (or worsens slightly from over-fitting on
small training sets).

Auto-detects CUDA; bf16 on CUDA, fp32 on CPU.
"""
from __future__ import annotations

import statistics
import time

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN


def _device_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def make_target(omega: float):
    def f(x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[..., 0] = torch.sin(omega * x[..., 0])
        return y
    return f


def train_grid(G: int, omega: float, seeds: list[int], device, dtype):
    d = 8
    R_o = R_i = 4
    R_b = 4
    steps = 800
    lr = 3e-3
    grid_lo, grid_hi = -2.5, 2.5
    target_fn = make_target(omega)
    evals: list[float] = []
    params = None
    for seed in seeds:
        torch.manual_seed(seed)
        cfg = FullMixTuckerConfig(
            d=d, m=d, R_o=R_o, R_i=R_i, R_b=R_b, G=G,
            grid_lo=grid_lo, grid_hi=grid_hi,
        )
        fm = FullMixTuckerFFN(cfg).to(device=device, dtype=dtype)
        if params is None:
            params = sum(p.numel() for p in fm.parameters())
        g_ = torch.Generator(device=device).manual_seed(seed)
        x = (torch.empty(2048, d, device=device, dtype=dtype)
             .uniform_(grid_lo, grid_hi, generator=g_))
        y = target_fn(x).to(dtype)
        opt = torch.optim.Adam(fm.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            (fm(x)[..., 0] - y[..., 0]).pow(2).mean().backward()
            opt.step()
        with torch.no_grad():
            x_eval = (torch.empty(1024, d, device=device, dtype=dtype)
                      .uniform_(grid_lo, grid_hi, generator=g_))
            y_eval = target_fn(x_eval).to(dtype)
            evals.append(
                (fm(x_eval)[..., 0] - y_eval[..., 0]).pow(2).mean().item()
            )
    return params, evals


def main():
    device, dtype = _device_dtype()
    seeds = [0, 1]

    print("=" * 78)
    print("Quality benchmark: grid-resolution G sweep")
    print(f"device={device}, dtype={dtype}, d=8, grid=[-2.5, 2.5]")
    print("Theory: knot density G/(hi-lo) sets the frequency cliff.  At")
    print("        grid range = 5.0 the cliff for omega is at G/5 cycles/unit.")
    print("=" * 78)

    for omega in [2.0, 4.0, 8.0]:
        cycles_per_unit = omega / (2 * 3.14159265)
        knots_needed = cycles_per_unit * 5.0 * 2  # crude Nyquist x 2
        print(f"\n--- omega = {omega}  "
              f"({cycles_per_unit:.2f} cycles/unit, "
              f"~{knots_needed:.0f} knots needed) ---")
        print(f"{'G':>4} {'params':>10} {'eval_mean':>14} {'eval_std':>12} "
              f"{'wall(s)':>10}")
        print("-" * 60)
        for G in [4, 8, 16, 32, 64]:
            t0 = time.perf_counter()
            params, evals = train_grid(G, omega, seeds, device, dtype)
            wall = time.perf_counter() - t0
            em = statistics.mean(evals)
            es = statistics.stdev(evals) if len(evals) > 1 else 0.0
            print(f"{G:>4} {params:>10,} {em:>14.4e} {es:>12.2e} "
                  f"{wall:>10.2f}")

    print("\n" + "=" * 78)
    print("Headline:")
    print("  - For each omega, the eval_mse should drop sharply when G")
    print("    crosses the frequency-cliff threshold and plateau after.")


if __name__ == "__main__":
    main()
