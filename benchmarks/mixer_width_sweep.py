"""Mixer width sweep: J.1.b non-compressive m sweep.

THEORY.md J.1.b says m = d is the lossless default mixer width and m = 2d
is a "quality rescue, not the default" because the dense mixer cost
becomes the dominant arithmetic.  This benchmark sweeps m / d over
{1.0, 1.25, 1.5, 2.0} and tracks:

  - parameter count and forward MACs (analytical),
  - eval MSE on a regression target,
  - wall clock (informally, for the MAC vs latency trade).

The compressive direction (m < d) is not tested here — the config
explicitly forbids it (re-introduces JHCG Defect 1).

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


def target(x: torch.Tensor) -> torch.Tensor:
    """Mixed nonlinear target across several output channels."""
    y = torch.zeros_like(x)
    d = x.shape[-1]
    for k in range(min(d, 6)):
        y[..., k] = (
            0.4 * torch.sin((k + 1) * x[..., k % d])
            + 0.2 * torch.tanh(x[..., (k + 2) % d])
        )
    return y


def train(model, *, d, steps, lr, seed, device, dtype):
    model = model.to(device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(2048, d, device=device, dtype=dtype, generator=g)
    y = target(x).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (model(x) - y).pow(2).mean().backward()
        opt.step()
    with torch.no_grad():
        x_eval = torch.randn(1024, d, device=device, dtype=dtype, generator=g)
        y_eval = target(x_eval).to(dtype)
        return (model(x_eval) - y_eval).pow(2).mean().item()


def fullmix_macs(d: int, m: int, R_o: int, R_i: int, R_b: int) -> int:
    """Analytical MAC count per token, mirroring benchmarks/flops.py."""
    return (
        d * m              # mixer
        + m * R_b          # B1 lookup (rough)
        + m * R_i * R_b    # input contraction
        + R_o * R_i * R_b  # core
        + d * R_o          # readout
    )


def main():
    device, dtype = _device_dtype()
    d = 32
    R_o = R_i = 16
    R_b = 4
    G = 12
    steps = 500
    lr = 3e-3
    seeds = [0, 1, 2]

    print("=" * 78)
    print("Mixer width m sweep: J.1.b non-compressive m / d")
    print(f"device={device}, dtype={dtype}, d={d}, R=({R_o},{R_i},{R_b})")
    print(f"steps={steps}, seeds={seeds}")
    print("=" * 78)
    print(f"\n{'m/d':>6} {'m':>5} {'params':>10} {'MACs/tok':>12} "
          f"{'eval_mean':>14} {'eval_std':>12} {'wall(s)':>10}")
    print("-" * 80)

    for ratio in [1.0, 1.25, 1.5, 2.0]:
        m = int(round(ratio * d))
        if m < d:
            continue
        evals: list[float] = []
        params = None
        t0 = time.perf_counter()
        for seed in seeds:
            torch.manual_seed(seed)
            cfg = FullMixTuckerConfig(d=d, m=m, R_o=R_o, R_i=R_i,
                                      R_b=R_b, G=G)
            fm = FullMixTuckerFFN(cfg)
            if params is None:
                params = sum(p.numel() for p in fm.parameters())
            evals.append(train(fm, d=d, steps=steps, lr=lr, seed=seed,
                               device=device, dtype=dtype))
        wall = time.perf_counter() - t0
        macs = fullmix_macs(d, m, R_o, R_i, R_b)
        em = statistics.mean(evals)
        es = statistics.stdev(evals) if len(evals) > 1 else 0.0
        print(f"{ratio:>6.2f} {m:>5} {params:>10,} {macs:>12,} "
              f"{em:>14.4e} {es:>12.2e} {wall:>10.2f}")

    print("\n" + "=" * 78)
    print("Headline:")
    print("  - Larger m increases param/MACs ~linearly; eval_mse should plateau")
    print("    if the inductive bias is already saturated at m=d.")
    print("  - If m=2d clearly beats m=d, it justifies the J.1.b 'quality rescue'.")


if __name__ == "__main__":
    main()
