"""Quality benchmark: F.4.c Strategy A — asymmetric Tucker ranks.

THEORY.md F.4.c proposes that when the per-layer FFN-update is output-rank
limited, the cleanest mitigation is to spend rank on R_o specifically:

   symmetric  (R_o, R_i, R_b) = (96, 96, 16)   — primary FullMix-Tucker
   asymmetric (R_o, R_i, R_b) = (256, 96, 16)  — same storage as r128, 2.7x R_o

This benchmark runs the small-scale analog: at d=32, compare
  - symmetric r4   (R_o = 4,  R_i = 4,  R_b = 4)  -- low-budget primary
  - symmetric r8   (R_o = 8,  R_i = 8,  R_b = 4)  -- symmetric rescue
  - asymmetric A   (R_o = 16, R_i = 4,  R_b = 4)  -- output-rich Strategy A
  - asymmetric I   (R_o = 4,  R_i = 16, R_b = 4)  -- input-rich (anti-strategy)
  - symmetric r16  (R_o = 16, R_i = 16, R_b = 4)  -- expensive ceiling

If F.4.b is correct, asymmetric A should beat symmetric r8 at the same
or smaller param count, and the input-rich anti-strategy should not.
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


def target_rank_rich(x: torch.Tensor) -> torch.Tensor:
    """Each output channel is a distinct nonlinear function — high effective
    output rank.  Asymmetric (R_o > R_i) should help here per F.4.b."""
    y = torch.zeros_like(x)
    d = x.shape[-1]
    for k in range(min(d, 8)):
        y[..., k] = 0.4 * torch.sin((k + 1) * x[..., k % d])
    return y


def train(model, target_fn, *, d, steps, lr, seed, device, dtype):
    model = model.to(device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(2048, d, device=device, dtype=dtype, generator=g)
    y = target_fn(x).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (model(x) - y).pow(2).mean().backward()
        opt.step()
    with torch.no_grad():
        x_eval = torch.randn(1024, d, device=device, dtype=dtype, generator=g)
        y_eval = target_fn(x_eval).to(dtype)
        eval_mse = (model(x_eval) - y_eval).pow(2).mean().item()
    return eval_mse


CONFIGS = [
    ("sym  r4    ", dict(R_o=4,  R_i=4,  R_b=4)),
    ("sym  r8    ", dict(R_o=8,  R_i=8,  R_b=4)),
    ("ASYM out-A ", dict(R_o=16, R_i=4,  R_b=4)),
    ("asym in    ", dict(R_o=4,  R_i=16, R_b=4)),
    ("sym  r16   ", dict(R_o=16, R_i=16, R_b=4)),
]


def main():
    device, dtype = _device_dtype()
    d = 32
    seeds = [0, 1, 2]
    steps = 800
    lr = 3e-3
    G = 16

    print("=" * 78)
    print("Quality benchmark: F.4.c asymmetric Tucker rank")
    print(f"device={device}, dtype={dtype}, d={d}, steps={steps}, G={G}")
    print("target = rank_rich (8 distinct nonlinear output channels)")
    print("=" * 78)
    print(f"\n{'config':<14} {'R_o':>4} {'R_i':>4} {'R_b':>4} "
          f"{'params':>10} {'eval_mean':>14} {'eval_std':>12} {'wall(s)':>10}")
    print("-" * 86)

    rows = []
    for label, cfg_kwargs in CONFIGS:
        evals: list[float] = []
        t0 = time.perf_counter()
        params = None
        for seed in seeds:
            torch.manual_seed(seed)
            cfg = FullMixTuckerConfig(d=d, m=d, G=G, **cfg_kwargs)
            fm = FullMixTuckerFFN(cfg)
            if params is None:
                params = sum(p.numel() for p in fm.parameters())
            ev = train(fm, target_rank_rich, d=d, steps=steps, lr=lr,
                       seed=seed, device=device, dtype=dtype)
            evals.append(ev)
        wall = time.perf_counter() - t0
        mean = statistics.mean(evals)
        sd = statistics.stdev(evals) if len(evals) > 1 else 0.0
        rows.append((label, cfg_kwargs, params, mean, sd))
        print(f"{label:<14} {cfg_kwargs['R_o']:>4} {cfg_kwargs['R_i']:>4} "
              f"{cfg_kwargs['R_b']:>4} {params:>10,} {mean:>14.4e} "
              f"{sd:>12.2e} {wall:>10.2f}")

    print("\n" + "-" * 78)
    by_label = {r[0]: r for r in rows}
    sym_r8 = by_label["sym  r8    "]
    asym_A = by_label["ASYM out-A "]
    asym_I = by_label["asym in    "]
    sym_r16 = by_label["sym  r16   "]

    print("F.4.c thesis read:")
    print(f"  sym r8       : {sym_r8[3]:.4e}  ({sym_r8[2]:,} params)")
    print(f"  ASYM out-A   : {asym_A[3]:.4e}  ({asym_A[2]:,} params)  "
          f"<-- output-rank fix per F.4.c Strategy A")
    print(f"  asym in      : {asym_I[3]:.4e}  ({asym_I[2]:,} params)  "
          f"<-- anti-strategy (input-rich)")
    print(f"  sym r16      : {sym_r16[3]:.4e}  ({sym_r16[2]:,} params)  "
          f"<-- expensive ceiling")
    print()
    if asym_A[3] < sym_r8[3]:
        print("  => ASYM out-A beats symmetric at similar storage  "
              "(F.4.c thesis SUPPORTED at this scale)")
    else:
        print("  => ASYM out-A does NOT beat symmetric here — either the target")
        print("     is not output-rank-bound at this scale, or the ablation")
        print("     needs more training / more seeds")
    if asym_A[3] < asym_I[3]:
        print("  => Output-rich beats input-rich  "
              "(rank-pathology localization confirmed)")
    else:
        print("  => Input-rich did better — bottleneck is not on the output side")


if __name__ == "__main__":
    main()
