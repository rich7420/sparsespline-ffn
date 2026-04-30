"""Quality benchmark: F.5.1 cumulative output-rank coverage.

THEORY.md F.5.1 predicts that with K replaced layers each of output rank
R_o, the union of column spaces approaches min(K * R_o, d).  Empirically
this should mean: at fixed per-layer R_o below d, MORE replaced layers
(K) should give MORE quality than fewer.

We build a residual MLP-style stack
   h_{l+1} = h_l + FFN_l(RMSNorm(h_l))
with K layers, all FullMix-Tucker, and train on a regression target.
We sweep K ∈ {1, 2, 4, 8} at fixed per-layer R_o = d/4.

The thesis predicts: eval_mse(K=8) <= eval_mse(K=4) <= eval_mse(K=2) <= K=1.
The diversity caveat (F.5.1 caveat) is that U_l columns must remain
linearly independent — we also report sigma_min/sigma_max of stacked U
post-training.

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


def rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / rms


def stack_forward(layers, x):
    h = x
    for ffn in layers:
        h = h + ffn(rmsnorm(h))
    return h


def target(x: torch.Tensor) -> torch.Tensor:
    """Per-channel mixture of nonlinearities.  Effective output rank ~= d."""
    y = torch.zeros_like(x)
    d = x.shape[-1]
    for k in range(d):
        y[..., k] = (
            0.3 * torch.sin((k + 1) * x[..., k])
            + 0.2 * torch.tanh(x[..., (k + 1) % d])
        )
    return y


def train_stack(layers, *, d, steps, lr, seed, device, dtype):
    for layer in layers:
        layer.to(device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(2048, d, device=device, dtype=dtype, generator=g)
    y = target(x).to(dtype)
    params = [p for layer in layers for p in layer.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (stack_forward(layers, x) - y).pow(2).mean().backward()
        opt.step()
    with torch.no_grad():
        x_eval = torch.randn(1024, d, device=device, dtype=dtype, generator=g)
        y_eval = target(x_eval).to(dtype)
        eval_mse = (stack_forward(layers, x_eval) - y_eval).pow(2).mean().item()
    return eval_mse


def stacked_U_sigmas(layers):
    """Return (sigma_min, sigma_max) of [U_1 | ... | U_K] in fp32."""
    Us = torch.cat([layer.U.detach().float().cpu() for layer in layers], dim=1)
    sv = torch.linalg.svdvals(Us)
    return sv.min().item(), sv.max().item()


def main():
    device, dtype = _device_dtype()
    d = 32
    seeds = [0, 1, 2]
    steps = 500
    lr = 3e-3
    R_o = d // 4   # 8
    R_i = R_o
    R_b = 4
    G = 12

    print("=" * 78)
    print("Quality benchmark: F.5.1 placement-K cumulative coverage")
    print(f"device={device}, dtype={dtype}, d={d}, steps={steps}")
    print(f"per-layer rank: R_o={R_o}, R_i={R_i}, R_b={R_b}, G={G}")
    print("Thesis: more layers (K) => more quality; sigma_min(stacked U) > 0")
    print("=" * 78)
    print(f"\n{'K':>3} {'cum_rank_bound':>16} {'eval_mean':>14} {'eval_std':>12} "
          f"{'sigma_min':>12} {'sigma_max':>12} {'cond':>10} {'wall(s)':>10}")
    print("-" * 100)

    rows = []
    for K in [1, 2, 4, 8]:
        evals: list[float] = []
        smins: list[float] = []
        smaxs: list[float] = []
        t0 = time.perf_counter()
        for seed in seeds:
            torch.manual_seed(seed)
            cfg = FullMixTuckerConfig(d=d, m=d, R_o=R_o, R_i=R_i,
                                      R_b=R_b, G=G)
            layers = [FullMixTuckerFFN(cfg) for _ in range(K)]
            ev = train_stack(layers, d=d, steps=steps, lr=lr,
                             seed=seed, device=device, dtype=dtype)
            evals.append(ev)
            smin, smax = stacked_U_sigmas(layers)
            smins.append(smin)
            smaxs.append(smax)
        wall = time.perf_counter() - t0
        cum_bound = min(K * R_o, d)
        em = statistics.mean(evals)
        es = statistics.stdev(evals) if len(evals) > 1 else 0.0
        sm = statistics.mean(smins)
        sx = statistics.mean(smaxs)
        cond = sx / sm if sm > 1e-12 else float("inf")
        rows.append((K, cum_bound, em, es, sm, sx, cond, wall))
        print(f"{K:>3} {cum_bound:>16} {em:>14.4e} {es:>12.2e} "
              f"{sm:>12.4f} {sx:>12.4f} {cond:>10.2f} {wall:>10.2f}")

    print("\n" + "=" * 78)
    print("Diagnostics:")
    last = rows[-1]
    first = rows[0]
    if last[2] < first[2]:
        print(f"  K=8 ({last[2]:.4e}) beats K=1 ({first[2]:.4e})  "
              "=> F.5.1 thesis SUPPORTED")
    else:
        print(f"  K=8 ({last[2]:.4e}) does not beat K=1 ({first[2]:.4e}) — "
              "could be saturated or bottlenecked elsewhere")
    print(f"\n  U-subspace diversity (sigma_min of stacked U at end of training):")
    for K, _b, _em, _es, sm, _sx, cond, _w in rows:
        flag = "ok" if sm > 1e-3 else "COLLAPSE"
        print(f"    K={K:>2}: sigma_min={sm:.4f}, cond={cond:.1f}  [{flag}]")
    print("\n  If sigma_min collapses to ~0, U_l columns are redundant and the")
    print("  cumulative-rank bound min(K*R_o, d) is loose (F.5.1 Caveat).")


if __name__ == "__main__":
    main()
