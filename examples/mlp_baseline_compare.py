"""Side-by-side MLP vs FullMix-Tucker training trace.

A concrete walk-through of the parity comparison: at the same parameter
budget, fit a synthetic regression target with both an MLPFFN baseline
and a FullMixTuckerFFN.  Print the final eval MSE side-by-side.

This is the smallest possible "does FullMix-Tucker actually fit" demo —
useful both as documentation and as a sanity check after install.
"""
from __future__ import annotations

import statistics

import torch

from sparsespline_ffn import MLPFFN, FullMixTuckerConfig, FullMixTuckerFFN


def target(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    y[..., 0] = torch.sin(x[..., 0]) + 0.5 * torch.cos(x[..., 1])
    y[..., 1] = torch.tanh(2 * x[..., 2])
    if x.shape[-1] >= 4:
        y[..., 2] = 0.3 * x[..., 3] ** 2
    return y


def train(model: torch.nn.Module, *, d: int, steps: int = 600,
          lr: float = 3e-3, seed: int = 0) -> float:
    torch.manual_seed(seed)
    x = torch.randn(1024, d)
    y = target(x)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (model(x) - y).pow(2).mean().backward()
        opt.step()
    with torch.no_grad():
        x_eval = torch.randn(512, d)
        return (model(x_eval) - target(x_eval)).pow(2).mean().item()


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    d = 32
    seeds = [0, 1, 2]

    # FullMix-Tucker at modest rank.
    fm_template = FullMixTuckerFFN(
        FullMixTuckerConfig(d=d, m=d, R_o=16, R_i=16, R_b=4, G=12)
    )
    fm_params = count_params(fm_template)

    # Pick the largest mlp_ratio whose params <= FullMix params.
    best_r = 1
    for r in [1, 2, 3, 4, 6]:
        if 2 * d * (r * d) <= fm_params:
            best_r = r
    mlp_template = MLPFFN(d=d, mlp_ratio=best_r)
    mlp_params = count_params(mlp_template)

    print(f"d={d}, FullMix params {fm_params:,}, "
          f"matched MLP (mlp_ratio={best_r}) params {mlp_params:,}")
    print(f"Training {len(seeds)} seeds for 600 steps each on a small "
          f"synthetic regression target.\n")

    fm_evals: list[float] = []
    mlp_evals: list[float] = []
    for seed in seeds:
        torch.manual_seed(seed)
        fm = FullMixTuckerFFN(
            FullMixTuckerConfig(d=d, m=d, R_o=16, R_i=16, R_b=4, G=12)
        )
        fm_evals.append(train(fm, d=d, seed=seed))

        torch.manual_seed(seed)
        mlp = MLPFFN(d=d, mlp_ratio=best_r)
        mlp_evals.append(train(mlp, d=d, seed=seed))

    print(f"{'model':<14} {'eval_mse_mean':>16} {'eval_mse_std':>14}")
    print("-" * 48)
    print(f"{'FullMix':<14} {statistics.mean(fm_evals):>16.4e} "
          f"{statistics.stdev(fm_evals):>14.2e}")
    print(f"{'MLP':<14} {statistics.mean(mlp_evals):>16.4e} "
          f"{statistics.stdev(mlp_evals):>14.2e}")

    wins = sum(1 for a, b in zip(fm_evals, mlp_evals, strict=True) if a < b)
    print(f"\nFullMix wins {wins}/{len(seeds)} seeds at matched param "
          f"budget on this target.")


if __name__ == "__main__":
    main()
