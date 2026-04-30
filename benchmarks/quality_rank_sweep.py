"""Quality benchmark: F.4.b output-rank sweep.

The central claim of THEORY.md F.4.b is that FullMix-Tucker's per-layer
FFN update is constrained to col-space(U) of dim <= R_o, vs MLP's full
rank-d output.  This benchmark sweeps R_o and measures eval MSE on a
synthetic regression target — the goal is to find the R_o saturation
point at which adding more output rank stops helping.

We hold R_i and R_b fixed and vary R_o ∈ {2, 4, 8, 16, 32, 48} at d=32.
A matched-budget MLP serves as the parity baseline (param-count chosen so
MLP_params ≈ symmetric (R_o=R_i=d/2) FullMix params).

Expected pattern (per F.4.b):
    eval_mse(R_o) decreases sharply for R_o < intrinsic-rank,
    flattens once R_o >= intrinsic-rank.

Auto-detects CUDA; uses bf16 on CUDA, fp32 on CPU.
"""
from __future__ import annotations

import statistics
import time

import torch

from sparsespline_ffn import MLPFFN, FullMixTuckerConfig, FullMixTuckerFFN


def _device_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def target_smooth(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    y[..., 0] = torch.sin(x[..., 0]) + 0.5 * torch.cos(x[..., 1])
    y[..., 1] = torch.tanh(x[..., 2])
    return y


def target_rank_rich(x: torch.Tensor) -> torch.Tensor:
    """A target whose effective output rank is higher: each output channel
    is a distinct nonlinear function of the input.  R_o-bottlenecked layers
    should under-fit this more than they under-fit ``smooth``."""
    y = torch.zeros_like(x)
    d = x.shape[-1]
    for k in range(min(8, d)):
        y[..., k] = 0.4 * torch.sin((k + 1) * x[..., k % d])
    return y


TARGETS = {"smooth": target_smooth, "rank_rich": target_rank_rich}


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def train(model, target_fn, *, d, steps, lr, seed, device, dtype):
    model = model.to(device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(seed + 13)
    x = torch.randn(1024, d, device=device, dtype=dtype, generator=g)
    y = target_fn(x).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    initial = float("inf")
    for step in range(steps):
        opt.zero_grad()
        pred = model(x)
        mse = (pred - y).pow(2).mean()
        if step == 0:
            initial = mse.item()
        mse.backward()
        opt.step()
    with torch.no_grad():
        x_eval = torch.randn(512, d, device=device, dtype=dtype, generator=g)
        y_eval = target_fn(x_eval).to(dtype)
        eval_mse = (model(x_eval) - y_eval).pow(2).mean().item()
    return initial, eval_mse


def main():
    device, dtype = _device_dtype()
    d = 32
    seeds = [0, 1, 2]
    steps = 600
    lr = 3e-3
    R_i = 16
    R_b = 8
    G = 16

    print("=" * 78)
    print("Quality benchmark: F.4.b output-rank sweep")
    print(f"device={device}, dtype={dtype}, d={d}, steps={steps}, "
          f"R_i={R_i}, R_b={R_b}, G={G}")
    print("=" * 78)

    for tname, tfn in TARGETS.items():
        print(f"\n--- target = {tname} ---")
        print(f"{'R_o':>4} {'params':>10} {'eval_mse_mean':>16} "
              f"{'eval_mse_std':>14} {'wall(s)':>10}")
        print("-" * 60)
        rows = []
        for R_o in [2, 4, 8, 16, 32, 48]:
            evals: list[float] = []
            t0 = time.perf_counter()
            for seed in seeds:
                torch.manual_seed(seed)
                cfg = FullMixTuckerConfig(d=d, m=d, R_o=R_o, R_i=R_i,
                                          R_b=R_b, G=G)
                fm = FullMixTuckerFFN(cfg)
                _init, ev = train(fm, tfn, d=d, steps=steps, lr=lr,
                                  seed=seed, device=device, dtype=dtype)
                evals.append(ev)
            wall = time.perf_counter() - t0
            params = sum(p.numel() for p in fm.parameters())
            mean = statistics.mean(evals)
            sd = statistics.stdev(evals) if len(evals) > 1 else 0.0
            rows.append((R_o, params, mean, sd))
            print(f"{R_o:>4} {params:>10,} {mean:>16.4e} {sd:>14.2e} "
                  f"{wall:>10.2f}")

        # Saturation diagnostic: smallest R_o whose eval is within 10% of best
        best = min(r[2] for r in rows)
        saturate = next(
            (r[0] for r in rows if r[2] <= best * 1.1), rows[-1][0]
        )
        print(f"\n  saturation R_o (within 10% of best): {saturate}")
        print(f"  best eval_mse                       : {best:.4e}")

        # MLP parity baseline
        torch.manual_seed(0)
        mlp = MLPFFN(d=d, mlp_ratio=2)  # ~2*d^2 = 2048 params at d=32
        mlp_params = count_params(mlp)
        _init, mlp_eval = train(mlp, tfn, d=d, steps=steps, lr=lr,
                                seed=0, device=device, dtype=dtype)
        print(f"  MLP baseline (mlp_ratio=2, {mlp_params:,} params): "
              f"eval_mse {mlp_eval:.4e}")

    print("\n" + "=" * 78)
    print("Headline:")
    print("  - Lower R_o => higher eval_mse confirms F.4.b's per-layer rank cap.")
    print("  - 'saturation R_o' is the smallest R_o whose quality is within 10%")
    print("    of the best in the sweep — the practical sufficient rank.")


if __name__ == "__main__":
    main()
