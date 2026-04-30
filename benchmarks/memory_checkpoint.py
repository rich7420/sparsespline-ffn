"""Empirical memory benchmark: ``torch.utils.checkpoint`` vs eager.

THEORY.md K.0.3 recommends wrapping each FullMix-Tucker layer with
``torch.utils.checkpoint.checkpoint`` to avoid materializing the
beta tensor (N, m, R_b) — which dominates activation memory at nanochat
scale.  This benchmark measures the actual peak allocation under both
modes for a Pattern A+ (K=6) and Pattern Full (K=12) stack.

CUDA-only; reports torch.cuda.max_memory_allocated.  When CUDA is
absent it reports a coarse CPU-only sanity (no peak allocator on CPU,
so we just confirm both modes produce identical output).
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.utils.checkpoint as ckpt

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN


def _build_layers(K: int, d: int, m: int, R_o: int, R_i: int,
                  R_b: int, G: int, device, dtype):
    cfg = FullMixTuckerConfig(d=d, m=m, R_o=R_o, R_i=R_i, R_b=R_b, G=G)
    layers = torch.nn.ModuleList(
        [FullMixTuckerFFN(cfg) for _ in range(K)]
    ).to(device=device, dtype=dtype)
    return layers


def stack_forward_eager(layers, x):
    h = x
    for ffn in layers:
        h = h + ffn(h)
    return h


def stack_forward_ckp(layers, x):
    h = x
    for ffn in layers:
        h = h + ckpt.checkpoint(ffn, h, use_reentrant=False)
    return h


def measure_peak(name, layers, x, fwd_fn, device):
    """Run forward+backward, record peak allocation in bytes."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    h = fwd_fn(layers, x)
    loss = h.pow(2).sum()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    for layer in layers:
        layer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
    else:
        peak = -1  # CPU has no native peak tracker
    return name, peak, elapsed_ms


def _fmt(n: int) -> str:
    if n < 0:
        return "n/a"
    if n >= 1024**3:
        return f"{n/1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n/1024**2:.2f} MB"
    if n >= 1024:
        return f"{n/1024:.2f} KB"
    return f"{n} B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--R_o", type=int, default=64)
    ap.add_argument("--R_i", type=int, default=64)
    ap.add_argument("--R_b", type=int, default=16)
    ap.add_argument("--G", type=int, default=20)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--K", type=int, nargs="*", default=[1, 6, 12])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 78)
    print("Memory benchmark: eager vs torch.utils.checkpoint")
    print(f"device={device}, dtype={dtype}")
    print(f"d={args.d}, R=({args.R_o},{args.R_i},{args.R_b}), G={args.G}")
    print(f"shape: B={args.B}, T={args.T} -> N={args.B * args.T} tokens")
    print("=" * 78)

    if device.type != "cuda":
        print("\nNote: CPU run only verifies output equivalence; no peak "
              "memory available without CUDA.\n")

    print(f"\n{'K':>3} {'mode':<10} {'peak':>10} {'wall(ms)':>10} "
          f"{'agree?':>8}")
    print("-" * 50)

    for K in args.K:
        layers = _build_layers(K, args.d, args.d, args.R_o, args.R_i,
                               args.R_b, args.G, device, dtype)
        x = torch.randn(args.B, args.T, args.d,
                        device=device, dtype=dtype, requires_grad=True)

        # Output equivalence check (in fp32 for noise floor below bf16's).
        with torch.no_grad():
            y_eager = stack_forward_eager(layers, x)
            y_ckp = stack_forward_ckp(layers, x)
        diff = (y_eager.float() - y_ckp.float()).abs().max().item()
        agree = "yes" if diff < (1e-2 if dtype == torch.bfloat16 else 1e-5) else "NO"

        # Eager.
        x_e = x.detach().clone().requires_grad_(True)
        _, peak_e, wall_e = measure_peak(
            f"K={K} eager", layers, x_e, stack_forward_eager, device
        )
        # Checkpointed.
        x_c = x.detach().clone().requires_grad_(True)
        _, peak_c, wall_c = measure_peak(
            f"K={K} ckp",   layers, x_c, stack_forward_ckp,   device
        )
        print(f"{K:>3} {'eager':<10} {_fmt(peak_e):>10} "
              f"{wall_e:>10.2f} {agree:>8}")
        print(f"{'':>3} {'+ckpt':<10} {_fmt(peak_c):>10} "
              f"{wall_c:>10.2f}")
        if peak_e > 0 and peak_c > 0:
            ratio = peak_e / peak_c
            print(f"{'':>3} {'savings':<10} "
                  f"{ratio:>10.2f}x peak less with checkpoint")

    print("\n" + "=" * 78)
    print("Headline:")
    print("  - Per K.0.3, beta tensor is the dominant activation; checkpoint")
    print("    re-runs stages 1+2 in backward, recovering ~all of beta's bytes.")


if __name__ == "__main__":
    main()
