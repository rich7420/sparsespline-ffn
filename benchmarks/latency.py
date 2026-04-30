"""Forward + backward latency: FullMix-Tucker (form B) vs MLPFFN.

Auto-selects CUDA if available, else CPU.  Reports per-iteration ms over
warmup + measure phases, with fwd-only and fwd+bwd splits.

Usage:
    python benchmarks/latency.py                      # default config
    python benchmarks/latency.py --d 768 --B 4 --T 512 --warmup 5 --iters 20
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from sparsespline_ffn import MLPFFN, FullMixTuckerConfig, FullMixTuckerFFN


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, device: torch.device, iters: int) -> list[float]:
    samples = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def _bench(
    name: str,
    module: torch.nn.Module,
    x_factory,
    device: torch.device,
    warmup: int,
    iters: int,
    do_backward: bool,
) -> dict:
    module.train(do_backward)
    # warmup
    for _ in range(warmup):
        x = x_factory()
        if do_backward:
            x.requires_grad_(True)
        y = module(x)
        if do_backward:
            y.pow(2).sum().backward()
            module.zero_grad(set_to_none=True)

    def step():
        x = x_factory()
        if do_backward:
            x.requires_grad_(True)
        y = module(x)
        if do_backward:
            y.pow(2).sum().backward()
            module.zero_grad(set_to_none=True)

    samples = _time_call(step, device, iters)
    return {
        "name": name,
        "mode": "fwd+bwd" if do_backward else "fwd",
        "median_ms": statistics.median(samples),
        "p10_ms": sorted(samples)[max(0, int(0.1 * len(samples)))],
        "p90_ms": sorted(samples)[min(len(samples) - 1, int(0.9 * len(samples)))],
        "min_ms": min(samples),
    }


def _print_table(rows: list[dict], baseline_name: str) -> None:
    print(f"\n  {'name':<28} {'mode':<8} {'median(ms)':>12} "
          f"{'p10':>8} {'p90':>8} {'min':>8} {'vs ' + baseline_name:>14}")
    base = next((r for r in rows if r["name"] == baseline_name and r["mode"] == "fwd"), None)
    base_bwd = next(
        (r for r in rows if r["name"] == baseline_name and r["mode"] == "fwd+bwd"), None
    )
    for r in rows:
        b = base if r["mode"] == "fwd" else base_bwd
        ratio = (
            f"{r['median_ms'] / b['median_ms']:.2f}x" if b else "—"
        )
        print(
            f"  {r['name']:<28} {r['mode']:<8} {r['median_ms']:>12.3f} "
            f"{r['p10_ms']:>8.3f} {r['p90_ms']:>8.3f} {r['min_ms']:>8.3f} "
            f"{ratio:>14}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=768)
    parser.add_argument("--R_o", type=int, default=96)
    parser.add_argument("--R_i", type=int, default=96)
    parser.add_argument("--R_b", type=int, default=16)
    parser.add_argument("--G", type=int, default=20)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default=None,
                        help="default: bf16 on cuda, fp32 on cpu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print("=" * 78)
    print("Latency benchmark: FullMix-Tucker vs MLPFFN")
    print(f"device = {device}, dtype = {dtype}")
    print(f"d={args.d}, R_o={args.R_o}, R_i={args.R_i}, R_b={args.R_b}, "
          f"G={args.G}, mlp_ratio={args.mlp_ratio}")
    print(f"shape: B={args.B}, T={args.T} -> N={args.B * args.T} tokens "
          f"({args.warmup} warmup + {args.iters} iters)")
    print("=" * 78)

    cfg = FullMixTuckerConfig(
        d=args.d, m=args.d,
        R_o=args.R_o, R_i=args.R_i, R_b=args.R_b, G=args.G,
    )
    fm = FullMixTuckerFFN(cfg).to(device=device, dtype=dtype)
    mlp = MLPFFN(d=args.d, mlp_ratio=args.mlp_ratio).to(device=device, dtype=dtype)

    def x_factory() -> torch.Tensor:
        return torch.randn(args.B, args.T, args.d, device=device, dtype=dtype)

    rows = []
    for do_bwd in [False, True]:
        rows.append(_bench(
            "MLPFFN", mlp, x_factory, device, args.warmup, args.iters, do_bwd
        ))
        rows.append(_bench(
            "FullMixTuckerFFN", fm, x_factory, device, args.warmup, args.iters, do_bwd
        ))

    _print_table(rows, baseline_name="MLPFFN")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\n  cuda peak alloc: {peak_mb:.1f} MB (across all benches)")


if __name__ == "__main__":
    main()
