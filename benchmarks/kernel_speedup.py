"""Standalone dQ kernel speedup benchmark.

Compares the Triton ``b1_backward_dq`` kernel against the PyTorch reference
(two ``index_add_`` calls equivalent to ``aten::_index_put_impl_``) at
production shape.

Gate A2 (strong target): kernel >= 50x speedup
Gate A2' (acceptable):                >= 20x  -- proceed to Phase B if B-tier
                                                 gates also met.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch


def torch_dq_reference(
    bin_idx: torch.Tensor, t: torch.Tensor, dbeta: torch.Tensor, L: int
) -> torch.Tensor:
    """The slow op the kernel replaces.

    Mirrors the gradient that PyTorch autograd produces for the
    ``Q[bin_idx]`` / ``Q[bin_idx+1]`` reads inside form-B forward.
    """
    bin_flat = bin_idx.reshape(-1).long()
    t_flat = t.reshape(-1)
    db_flat = dbeta.reshape(-1, dbeta.shape[-1])
    dQ = torch.zeros((L, db_flat.shape[-1]), dtype=db_flat.dtype, device=db_flat.device)
    dQ.index_add_(0, bin_flat, (1.0 - t_flat).unsqueeze(-1) * db_flat)
    dQ.index_add_(0, bin_flat + 1,        t_flat.unsqueeze(-1) * db_flat)
    return dQ


def time_op(fn, *, warmup: int = 10, iters: int = 50, device: torch.device) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1000)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "p10_ms": sorted(samples)[int(0.1 * len(samples))],
        "p90_ms": sorted(samples)[int(0.9 * len(samples))],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2048,
                    help="batch * seq_len (production: B=4 * T=512 = 2048)")
    ap.add_argument("--m", type=int, default=768)
    ap.add_argument("--R_b", type=int, default=16)
    ap.add_argument("--L", type=int, default=21)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print("=" * 78)
    print("dQ kernel speedup benchmark")
    print(f"  shape: N={args.N}, m={args.m}, R_b={args.R_b}, L={args.L}")
    print(f"  dtype: {dtype}")
    print(f"  warmup={args.warmup}, iters={args.iters}")
    print("=" * 78)

    torch.manual_seed(0)
    bin_idx = torch.randint(0, args.L - 1, (args.N, args.m), dtype=torch.int64, device=device)
    t = torch.rand(args.N, args.m, dtype=dtype, device=device)
    dbeta = torch.randn(args.N, args.m, args.R_b, dtype=dtype, device=device)

    # 1. PyTorch reference (the bottleneck)
    print("\n>>> PyTorch index_add_ reference")
    torch_stats = time_op(
        lambda: torch_dq_reference(bin_idx, t, dbeta, args.L),
        warmup=args.warmup, iters=args.iters, device=device,
    )
    print(f"  median: {torch_stats['median_ms']:.3f} ms  "
          f"(min {torch_stats['min_ms']:.3f}, "
          f"p10 {torch_stats['p10_ms']:.3f}, p90 {torch_stats['p90_ms']:.3f})")

    # 2. Triton kernel
    from sparsespline_ffn.kernels import b1_backward_dq

    # First call triggers autotune; do extra warmup so subsequent calls are clean.
    for _ in range(3):
        b1_backward_dq(bin_idx, t, dbeta, L=args.L)
    torch.cuda.synchronize(device)

    print("\n>>> Triton b1_backward_dq")
    triton_stats = time_op(
        lambda: b1_backward_dq(bin_idx, t, dbeta, L=args.L),
        warmup=args.warmup, iters=args.iters, device=device,
    )
    print(f"  median: {triton_stats['median_ms']:.3f} ms  "
          f"(min {triton_stats['min_ms']:.3f}, "
          f"p10 {triton_stats['p10_ms']:.3f}, p90 {triton_stats['p90_ms']:.3f})")

    speedup = torch_stats["median_ms"] / triton_stats["median_ms"]
    print(f"\n  speedup (median): {speedup:.1f}x")

    # Gate decision
    print("\nGate A2 verdict:")
    if speedup >= 50:
        print(f"  STRONG TARGET MET ({speedup:.1f}x >= 50x)")
        verdict = 0
    elif speedup >= 20:
        print(f"  ACCEPTABLE ({speedup:.1f}x in [20, 50)) -- proceed if Phase B "
              "also passes")
        verdict = 0
    else:
        print(f"  BELOW ACCEPTABLE ({speedup:.1f}x < 20x) -- consider two-pass "
              "partial reduction fallback")
        verdict = 2

    # Correctness check vs fp32 ground truth (the actual oracle).
    # The bf16 reference accumulates in bf16 too, so it's strictly *less*
    # accurate than the kernel's fp32 internal accumulation -- comparing the
    # kernel against the bf16 reference would mis-classify the kernel as wrong.
    print("\nCorrectness re-check at production shape (vs fp32 ground truth):")
    bin_fp32_ref = bin_idx
    t_fp32 = t.float()
    db_fp32 = dbeta.float()
    dQ_truth = torch_dq_reference(bin_fp32_ref, t_fp32, db_fp32, args.L)
    dQ_triton = b1_backward_dq(bin_idx, t, dbeta, L=args.L)
    rel = ((dQ_triton - dQ_truth).norm() / (dQ_truth.norm() + 1e-12)).item()
    # Kernel internal accum is fp32; outputs match fp32 truth even when inputs
    # are bf16 -- only loss is bf16 *load* precision.
    tol = 5e-3 if args.dtype == "bf16" else 1e-5
    if rel < tol:
        print(f"  rel = {rel:.3e} < {tol:.0e}  PASS")
    else:
        print(f"  rel = {rel:.3e} > {tol:.0e}  FAIL -- DO NOT TRUST KERNEL")
        verdict = 3

    # Also report the bf16-vs-bf16 reference gap, just to characterize
    # how much precision the kernel adds back.
    if args.dtype == "bf16":
        dQ_bf_ref = torch_dq_reference(bin_idx, t, dbeta, args.L).float()
        bf_rel = ((dQ_bf_ref - dQ_truth).norm() / (dQ_truth.norm() + 1e-12)).item()
        kernel_rel = rel
        print(f"\n  bf16-reference vs fp32-truth     rel = {bf_rel:.3e}")
        print(f"  kernel       vs fp32-truth     rel = {kernel_rel:.3e}")
        if bf_rel > kernel_rel:
            print(f"  -> kernel is more accurate than the bf16 PyTorch path "
                  f"(by {bf_rel/kernel_rel:.1f}x).")

    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
