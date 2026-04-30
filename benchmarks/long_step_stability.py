"""Long-step training stability: late-stage divergence detection.

Most quality benchmarks here run ~500-800 SGD steps.  Some failure modes
(grid drift, knot saturation, output gain runaway, late-layer Var blowup)
only appear after thousands of steps.  This benchmark trains a single
FullMix-Tucker layer for a few thousand steps and tracks:

  - loss vs step (every log_every);
  - Var[y] vs step (output magnitude);
  - ||gamma|| vs step (output gain runaway);
  - max(|U|), max(|Q|) vs step (parameter blowup).

A "stable" run keeps all four within an order of magnitude of their
init values.  Any monotone divergence over the second half is a fail.

Auto-detects CUDA; uses fp32 throughout to isolate dtype-related failures.
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
        y[..., k] = (
            0.4 * torch.sin((k + 1) * x[..., k])
            + 0.2 * torch.cos(x[..., (k + 1) % d])
        )
    return y


def main():
    device = _device()
    d = 32
    R_o = R_i = 16
    R_b = 4
    G = 16
    steps = 3000
    lr = 1e-3
    log_every = 250

    print("=" * 78)
    print("Long-step training stability")
    print(f"device={device}, d={d}, R=({R_o},{R_i},{R_b}), G={G}")
    print(f"steps={steps}, lr={lr}, log_every={log_every}")
    print("=" * 78)

    torch.manual_seed(0)
    cfg = FullMixTuckerConfig(d=d, m=d, R_o=R_o, R_i=R_i, R_b=R_b, G=G)
    fm = FullMixTuckerFFN(cfg).to(device).float()

    g = torch.Generator(device=device).manual_seed(1)
    x = torch.randn(2048, d, device=device, generator=g)
    y = target(x)
    opt = torch.optim.Adam(fm.parameters(), lr=lr)

    print(f"\n{'step':>6} {'loss':>14} {'Var[y]':>12} {'|gamma|':>10} "
          f"{'max|U|':>10} {'max|Q|':>10}")
    print("-" * 70)

    init_metrics = {}
    final_metrics = {}
    log_history: list[dict[str, float]] = []

    t0 = time.perf_counter()
    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                pred = fm(x)
                metrics = {
                    "loss":    (pred - y).pow(2).mean().item(),
                    "var_y":   pred.var().item(),
                    "gamma":   fm.gamma.abs().item(),
                    "max_U":   fm.U.abs().max().item(),
                    "max_Q":   fm.Q.abs().max().item(),
                }
            log_history.append({"step": step, **metrics})
            print(f"{step:>6} {metrics['loss']:>14.4e} "
                  f"{metrics['var_y']:>12.4e} {metrics['gamma']:>10.4f} "
                  f"{metrics['max_U']:>10.4f} {metrics['max_Q']:>10.4f}")
            if step == 0:
                init_metrics = dict(metrics)
            if step == steps:
                final_metrics = dict(metrics)
        if step == steps:
            break
        opt.zero_grad()
        (fm(x) - y).pow(2).mean().backward()
        if not all(torch.isfinite(p).all() for p in fm.parameters()):
            print(f"\nDIVERGED at step {step}: non-finite parameters.")
            return
        opt.step()
    elapsed = time.perf_counter() - t0

    print(f"\nWall: {elapsed:.1f}s")
    print("\nDelta vs init:")
    for k in ["loss", "var_y", "gamma", "max_U", "max_Q"]:
        i = init_metrics[k]
        f = final_metrics[k]
        ratio = f / max(abs(i), 1e-12)
        print(f"  {k:<10}: {i:.4e} -> {f:.4e}  (x{ratio:.2f})")

    # Pass criterion: no metric should grow more than 5x or shrink below 1/5.
    bad = []
    for k in ["var_y", "gamma", "max_U", "max_Q"]:
        ratio = final_metrics[k] / max(abs(init_metrics[k]), 1e-12)
        if ratio > 5.0 or ratio < 0.2:
            bad.append((k, ratio))
    print()
    if not bad:
        print("OK: all stability metrics stayed within [0.2x, 5x] of init.")
    else:
        print("WARNING: parameters drifted outside the [0.2x, 5x] band:")
        for name, r in bad:
            print(f"  {name}: x{r:.2f}")


if __name__ == "__main__":
    main()
