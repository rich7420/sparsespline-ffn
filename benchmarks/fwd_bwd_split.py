"""Latency benchmark: forward vs backward split.

THEORY.md L.5 (citing FlashKAT) warns that the backward pass dominates
KAN-style kernel cost on GPU because gradient on the spline-coefficient
table requires atomic adds across tokens.  The reference path doesn't
use a custom kernel, but autograd through the gather-lerp still has its
own cost.

This benchmark times forward and backward SEPARATELY for both
FullMix-Tucker (form B reference) and MLPFFN at production-ish scale
and reports the bwd:fwd ratio for each.  This is the metric Phase 2
should target.

Auto-detects CUDA; bf16 on CUDA, fp32 on CPU.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from sparsespline_ffn import MLPFFN, FullMixTuckerConfig, FullMixTuckerFFN


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench_fwd_only(module, x_factory, device, warmup, iters):
    module.train(False)
    for _ in range(warmup):
        with torch.no_grad():
            module(x_factory())
    samples = []
    for _ in range(iters):
        x = x_factory()
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            module(x)
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def _bench_fwd_bwd(module, x_factory, device, warmup, iters):
    """Runs fwd to capture activations, then times the backward only."""
    module.train(True)
    for _ in range(warmup):
        x = x_factory().requires_grad_(True)
        y = module(x)
        y.pow(2).sum().backward()
        module.zero_grad(set_to_none=True)
    fwd = []
    bwd = []
    for _ in range(iters):
        x = x_factory().requires_grad_(True)
        _sync(device)
        t0 = time.perf_counter()
        y = module(x)
        _sync(device)
        fwd.append((time.perf_counter() - t0) * 1000)
        loss = y.pow(2).sum()
        _sync(device)
        t0 = time.perf_counter()
        loss.backward()
        _sync(device)
        bwd.append((time.perf_counter() - t0) * 1000)
        module.zero_grad(set_to_none=True)
    return fwd, bwd


def _summary(samples):
    if not samples:
        return {"med": float("nan"), "p10": float("nan"), "p90": float("nan")}
    s = sorted(samples)
    n = len(s)
    return {
        "med": statistics.median(s),
        "p10": s[max(0, n // 10)],
        "p90": s[min(n - 1, (9 * n) // 10)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--R_o", type=int, default=96)
    ap.add_argument("--R_i", type=int, default=96)
    ap.add_argument("--R_b", type=int, default=16)
    ap.add_argument("--G", type=int, default=20)
    ap.add_argument("--mlp_ratio", type=int, default=4)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 78)
    print("Forward / backward latency split: FullMix-Tucker vs MLP")
    print(f"device={device}, dtype={dtype}")
    print(f"d={args.d}, R_o={args.R_o}, R_i={args.R_i}, R_b={args.R_b}, "
          f"G={args.G}, mlp_ratio={args.mlp_ratio}")
    print(f"shape: B={args.B}, T={args.T} -> N={args.B * args.T} tokens")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print("=" * 78)

    cfg = FullMixTuckerConfig(
        d=args.d, m=args.d,
        R_o=args.R_o, R_i=args.R_i, R_b=args.R_b, G=args.G,
    )
    fm = FullMixTuckerFFN(cfg).to(device=device, dtype=dtype)
    mlp = MLPFFN(d=args.d, mlp_ratio=args.mlp_ratio).to(device=device, dtype=dtype)

    def x_factory():
        return torch.randn(args.B, args.T, args.d,
                           device=device, dtype=dtype)

    print(f"\n{'name':<24} {'mode':<10} {'med(ms)':>10} {'p10':>8} {'p90':>8}")
    print("-" * 64)

    rows = []
    for name, mod in [("MLPFFN", mlp), ("FullMixTuckerFFN", fm)]:
        fwd_only = _bench_fwd_only(mod, x_factory, device, args.warmup, args.iters)
        fwd, bwd = _bench_fwd_bwd(mod, x_factory, device, args.warmup, args.iters)
        rows.append((name, fwd_only, fwd, bwd))
        for mode, samples in [("fwd-noGrad", fwd_only),
                              ("fwd-train",  fwd),
                              ("bwd",        bwd)]:
            s = _summary(samples)
            print(f"{name:<24} {mode:<10} {s['med']:>10.3f} "
                  f"{s['p10']:>8.3f} {s['p90']:>8.3f}")

    print("\n" + "-" * 78)
    print("Backward-to-forward ratio (FlashKAT-relevant):")
    for name, _fo, fwd, bwd in rows:
        ratio = statistics.median(bwd) / max(statistics.median(fwd), 1e-9)
        print(f"  {name:<22} bwd/fwd = {ratio:.2f}x")

    # Cross-model headline
    fm_total = (statistics.median(rows[1][2]) + statistics.median(rows[1][3]))
    mlp_total = (statistics.median(rows[0][2]) + statistics.median(rows[0][3]))
    print(f"\n  total step (fwd+bwd) FullMix / MLP = {fm_total / mlp_total:.2f}x")
    print("\n  Phase 2 kernel target: end-to-end <= 0.7x MLP (i.e., 1.4x speedup).")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"  cuda peak alloc: {peak:.1f} MB")


if __name__ == "__main__":
    main()
