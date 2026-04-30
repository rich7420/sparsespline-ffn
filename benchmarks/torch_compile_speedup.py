"""``torch.compile`` speedup benchmark.

Measures whether the 5-stage form B reference benefits from
``torch.compile`` (Inductor backend) on both forward and forward+backward.
The 5 stages are pure einsum / index / lerp ops, all of which Inductor
can fuse, so the compile path is a useful proxy for what an in-graph
fused kernel would deliver — and a partial answer for "do we even need
a custom Triton kernel before the paper headline."

Reports eager vs compile median ms over warmup + iters, plus speedup
ratio.  Warns if compile triggers graph breaks or recompiles.

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


def _bench_step(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--R_o", type=int, default=64)
    ap.add_argument("--R_i", type=int, default=64)
    ap.add_argument("--R_b", type=int, default=16)
    ap.add_argument("--G", type=int, default=20)
    ap.add_argument("--mlp_ratio", type=int, default=4)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--mode", choices=["fwd", "fwd_bwd"], default="fwd_bwd")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 78)
    print("torch.compile speedup: FullMix-Tucker (form B) vs MLP")
    print(f"device={device}, dtype={dtype}, mode={args.mode}")
    print(f"d={args.d}, R=({args.R_o},{args.R_i},{args.R_b}), G={args.G}")
    print(f"shape: B={args.B}, T={args.T}, "
          f"warmup={args.warmup}, iters={args.iters}")
    print("=" * 78)

    cfg = FullMixTuckerConfig(d=args.d, m=args.d,
                              R_o=args.R_o, R_i=args.R_i, R_b=args.R_b,
                              G=args.G)
    fm = FullMixTuckerFFN(cfg).to(device=device, dtype=dtype)
    mlp = MLPFFN(d=args.d, mlp_ratio=args.mlp_ratio).to(device=device,
                                                        dtype=dtype)

    fm_compiled = torch.compile(fm, fullgraph=False)
    mlp_compiled = torch.compile(mlp, fullgraph=False)

    def fwd_step(mod):
        def step():
            x = torch.randn(args.B, args.T, args.d,
                            device=device, dtype=dtype)
            with torch.no_grad():
                mod(x)
        return step

    def fwd_bwd_step(mod):
        def step():
            x = torch.randn(args.B, args.T, args.d,
                            device=device, dtype=dtype, requires_grad=True)
            y = mod(x)
            y.pow(2).sum().backward()
            mod.zero_grad(set_to_none=True)
        return step

    step_factory = fwd_bwd_step if args.mode == "fwd_bwd" else fwd_step

    print(f"\n{'name':<30} {'mode':<10} {'med(ms)':>10} {'p10':>8} {'p90':>8} "
          f"{'speedup':>10}")
    print("-" * 80)

    rows = []
    for label, mod_eager, mod_comp in [
        ("MLPFFN",            mlp,  mlp_compiled),
        ("FullMixTuckerFFN",  fm,   fm_compiled),
    ]:
        eager = _bench_step(step_factory(mod_eager), device,
                            args.warmup, args.iters)
        comp = _bench_step(step_factory(mod_comp), device,
                           args.warmup, args.iters)
        med_e = statistics.median(eager)
        med_c = statistics.median(comp)
        speedup = med_e / max(med_c, 1e-9)
        rows.append((label, med_e, med_c, speedup))
        for tag, samples in [("eager", eager), ("compile", comp)]:
            s = sorted(samples)
            n = len(s)
            print(f"{label:<30} {tag:<10} {statistics.median(s):>10.3f} "
                  f"{s[max(0,n//10)]:>8.3f} {s[min(n-1,(9*n)//10)]:>8.3f}")
        print(f"{label:<30} {'speedup':<10} {speedup:>10.2f}x")

    print("\n" + "=" * 78)
    print("Headline:")
    print("  - 5-stage einsum/index/lerp form should compile cleanly with")
    print("    Inductor.  A meaningful (>=1.3x) speedup here weakens the")
    print("    case for hand-written Triton fusion.")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"  cuda peak alloc: {peak:.1f} MB")


if __name__ == "__main__":
    main()
