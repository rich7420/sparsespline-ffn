"""Init sensitivity benchmark: sigma_c sweep around the L.4 recommendation.

THEORY.md L.4 derives sigma_c = sqrt(3 d / (2 R_o)) for variance-preserving
init.  This benchmark perturbs sigma_c by multiplicative factors in
{0.1, 0.3, 1.0, 3.0, 10.0} and measures:

  1. init output std (should drift away from 1.0)
  2. training stability over 200 SGD steps (does it explode/vanish?)
  3. final eval_mse on a small regression target

Pass criterion (per L.4 diagnostic): off-recipe inits should produce
either init std out of band [0.5, 2.0] or training divergence; the
on-recipe init should land in band and converge.
"""
from __future__ import annotations

import math
import statistics
import time

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def target(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    d = x.shape[-1]
    for k in range(min(d, 4)):
        y[..., k] = 0.4 * torch.sin((k + 1) * x[..., k])
    return y


def build_with_perturbed_Q(cfg, multiplier, seed):
    torch.manual_seed(seed)
    fm = FullMixTuckerFFN(cfg)
    sigma_c_base = math.sqrt(3.0 * cfg.d / (2.0 * cfg.R_o))
    target_std = sigma_c_base * multiplier
    with torch.no_grad():
        fm.Q.normal_(mean=0.0, std=target_std)
    return fm


def measure_init_std(fm, d, device, n: int = 1024) -> float:
    fm.to(device).float()
    g = torch.Generator(device=device).manual_seed(99)
    x = torch.randn(n, d, device=device, generator=g)
    with torch.no_grad():
        return fm(x).std().item()


def train_and_diagnose(fm, *, d, steps, lr, seed, device):
    fm.to(device).float()
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1024, d, device=device, generator=g)
    y = target(x)
    opt = torch.optim.SGD(fm.parameters(), lr=lr)
    initial_loss = float("inf")
    final_loss = float("inf")
    diverged = False
    output_stds = []
    for step in range(steps):
        opt.zero_grad()
        pred = fm(x)
        if not torch.isfinite(pred).all():
            diverged = True
            break
        if step % 50 == 0:
            output_stds.append(pred.std().item())
        mse = (pred - y).pow(2).mean()
        if step == 0:
            initial_loss = mse.item()
        if step == steps - 1:
            final_loss = mse.item()
        mse.backward()
        # Gradient clip to keep wild inits from immediately exploding
        torch.nn.utils.clip_grad_norm_(fm.parameters(), max_norm=10.0)
        opt.step()
    with torch.no_grad():
        eval_x = torch.randn(512, d, device=device, generator=g)
        eval_mse = (fm(eval_x) - target(eval_x)).pow(2).mean().item()
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "eval_mse": eval_mse,
        "diverged": diverged,
        "output_std_history": output_stds,
    }


def main():
    device = _device()
    d = 32
    R_o = R_i = 16
    R_b = 4
    G = 16
    seeds = [0, 1, 2]
    steps = 200
    lr = 0.02

    print("=" * 78)
    print("Init sensitivity: sigma_c sweep around L.4 recommendation")
    sigma_c_base = math.sqrt(3.0 * d / (2.0 * R_o))
    print(f"device={device}, d={d}, R_o={R_o}")
    print(f"L.4 sigma_c = sqrt(3d/(2 R_o)) = {sigma_c_base:.4f}")
    print(f"steps={steps}, lr={lr}, seeds={seeds}")
    print("=" * 78)
    print(f"\n{'mult':>8} {'sigma_c':>10} {'init_std_mean':>14} "
          f"{'final_loss':>12} {'eval_mse':>14} {'div?':>6} {'wall(s)':>10}")
    print("-" * 80)

    cfg = FullMixTuckerConfig(d=d, m=d, R_o=R_o, R_i=R_i, R_b=R_b, G=G)

    for mult in [0.1, 0.3, 1.0, 3.0, 10.0]:
        init_stds = []
        finals = []
        evals = []
        divs = 0
        t0 = time.perf_counter()
        for seed in seeds:
            fm = build_with_perturbed_Q(cfg, mult, seed)
            init_std = measure_init_std(fm, d, device)
            init_stds.append(init_std)

            torch.manual_seed(seed)
            fm = build_with_perturbed_Q(cfg, mult, seed)  # rebuild
            r = train_and_diagnose(fm, d=d, steps=steps, lr=lr,
                                    seed=seed, device=device)
            finals.append(r["final_loss"])
            evals.append(r["eval_mse"])
            if r["diverged"]:
                divs += 1
        wall = time.perf_counter() - t0

        print(f"{mult:>8.1f} {sigma_c_base * mult:>10.4f} "
              f"{statistics.mean(init_stds):>14.4f} "
              f"{statistics.mean([f for f in finals if math.isfinite(f)] or [float('nan')]):>12.4e} "
              f"{statistics.mean([e for e in evals if math.isfinite(e)] or [float('nan')]):>14.4e} "
              f"{divs:>3}/{len(seeds):<2} {wall:>10.2f}")

    print("\n" + "=" * 78)
    print("Diagnostics:")
    print("  - mult=1.0 (the recommended sigma_c) should give init_std in [0.5, 2.0]")
    print("    and the lowest eval_mse with no divergence.")
    print("  - Very small mult (0.1) starves spline updates; very large (10.0)")
    print("    can saturate or diverge depending on grid.")


if __name__ == "__main__":
    main()
