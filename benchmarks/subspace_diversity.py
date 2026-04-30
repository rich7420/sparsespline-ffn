"""Subspace-diversity benchmark: F.5.1 caveat tracking.

THEORY.md F.5.1 claims that the union of column spaces of K stacked
U_l matrices has dimension up to min(K * R_o, d), but only IF the U_l
remain linearly independent across layers.  If during training they
collapse to a shared subspace, the cumulative-rank argument breaks down.

This benchmark trains a K-layer residual stack and tracks
sigma_min / sigma_max / cond([U_1 | ... | U_K]) at fixed step intervals.
A divergent cond or sigma_min -> 0 indicates collapse.

The init U is orthogonal per L.4, so step-0 cond is 1.0.  We watch how
training perturbs that.

Auto-detects CUDA; runs in fp32 because torch.linalg.svdvals is fp32-only
on CPU and we want consistent precision across devices.
"""
from __future__ import annotations

import time

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rmsnorm(x, eps=1e-6):
    return x / x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()


def stack_forward(layers, x):
    h = x
    for ffn in layers:
        h = h + ffn(rmsnorm(h))
    return h


def target(x):
    y = torch.zeros_like(x)
    d = x.shape[-1]
    for k in range(d):
        y[..., k] = (
            0.3 * torch.sin((k + 1) * x[..., k])
            + 0.2 * torch.cos(x[..., (k + 1) % d])
        )
    return y


def stacked_U_diagnostics(layers):
    Us = torch.cat([layer.U.detach().float().cpu() for layer in layers], dim=1)
    sv = torch.linalg.svdvals(Us)
    smin = sv.min().item()
    smax = sv.max().item()
    cond = smax / max(smin, 1e-12)
    return smin, smax, cond


def main():
    device = _device()
    d = 32
    R_o = R_i = 8
    R_b = 4
    G = 12
    K = 6
    steps = 600
    lr = 3e-3
    log_every = 50

    print("=" * 78)
    print("F.5.1 caveat: U-subspace diversity tracking during training")
    print(f"device={device}, d={d}, K={K}, R_o={R_o}, steps={steps}")
    print(f"  per-layer rank covers {R_o}/{d} = {100 * R_o / d:.0f}% of d")
    print(f"  cumulative bound min(K*R_o, d) = {min(K * R_o, d)}")
    print("  init: U has orthonormal columns (cond ~= 1.0 expected at step 0)")
    print("=" * 78)

    torch.manual_seed(0)
    cfg = FullMixTuckerConfig(d=d, m=d, R_o=R_o, R_i=R_i, R_b=R_b, G=G)
    layers = [FullMixTuckerFFN(cfg) for _ in range(K)]
    for layer in layers:
        layer.to(device).float()

    g = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(1024, d, device=device, generator=g)
    y = target(x)
    params = [p for layer in layers for p in layer.parameters()]
    opt = torch.optim.Adam(params, lr=lr)

    print(f"\n{'step':>6} {'loss':>14} {'sigma_min':>12} {'sigma_max':>12} "
          f"{'cond':>10} {'rank':>6}")
    print("-" * 70)

    t0 = time.perf_counter()
    rank_history: list[tuple[int, float, float, float, int]] = []
    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                pred = stack_forward(layers, x)
                loss = (pred - y).pow(2).mean().item()
            smin, smax, cond = stacked_U_diagnostics(layers)
            Us = torch.cat([layer.U.detach().float().cpu() for layer in layers],
                           dim=1)
            rank = int(torch.linalg.matrix_rank(Us).item())
            rank_history.append((step, loss, smin, cond, rank))
            print(f"{step:>6} {loss:>14.4e} {smin:>12.4f} {smax:>12.4f} "
                  f"{cond:>10.2f} {rank:>6}")
        if step == steps:
            break
        opt.zero_grad()
        (stack_forward(layers, x) - y).pow(2).mean().backward()
        opt.step()
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 78)
    init = rank_history[0]
    final = rank_history[-1]
    print(f"Wall: {elapsed:.1f}s")
    print("\nDiagnostics:")
    print(f"  step 0   : sigma_min={init[2]:.4f}, cond={init[3]:.2f}, "
          f"rank={init[4]}/{min(K * R_o, d)}")
    print(f"  step {final[0]:>3}: sigma_min={final[2]:.4f}, cond={final[3]:.2f}, "
          f"rank={final[4]}/{min(K * R_o, d)}")

    if final[2] < 0.05 * init[2]:
        print("\n  WARNING: sigma_min collapsed by >20x — U_l columns are")
        print("  becoming redundant.  F.5.1 cumulative bound is loose here;")
        print("  Pattern Full's rank advantage may not materialize.")
    elif final[2] > 0.5 * init[2]:
        print("\n  OK: sigma_min stayed within 2x of init (training preserved")
        print("  U-diversity).  F.5.1 cumulative bound holds empirically.")
    else:
        print("\n  CAUTION: sigma_min shrank but did not collapse — borderline.")


if __name__ == "__main__":
    main()
