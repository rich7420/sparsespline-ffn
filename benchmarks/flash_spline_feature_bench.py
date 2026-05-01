"""FlashSplineFeature microbench — Task B2.2.

Three workloads per v7 §R.6.9:
  (a) uniform   z ~ N(0, 1)         — typical; 99% in-range
  (b) skewed    z ~ N(2.0, 0.5)     — moderate clustering near right boundary
  (c) collapsed z ~ N(0, 0.05)      — adversarial; almost all in 1 bin

For each workload:
  - correctness: kernel vs reference, max abs diff, relative RMS diff
  - speed: kernel forward wall, reference forward wall, ratio
  - (optional) baseline GEMM same shape, for "≤ 1.5× single GEMM" gate
    in v7 §R.4.4 / §R.6.2

Pass criteria (subjective, refined by data):
  - correctness: rel-RMS diff ≤ 5e-2 in bf16 (matches kernel test
    tolerance from v7 §R.3.3.5)
  - speed:
      uniform/skewed: kernel ≤ 2× reference   (eventually want < 1×)
      collapsed:      kernel ≤ 5× reference   (per v7 §R.6.9 worst case)

This bench does NOT launch H100 — it runs locally on whatever CUDA
device is available (3080).  Scale to H100 will be a re-run.

Run:
  python benchmarks/flash_spline_feature_bench.py
  python benchmarks/flash_spline_feature_bench.py --shape big
  python benchmarks/flash_spline_feature_bench.py --json-out result.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch


def _gen_workload(
    name: str, N: int, h: int, dtype: torch.dtype, device: torch.device,
    grid_lo: float = -3.0, grid_hi: float = 3.0,
) -> torch.Tensor:
    if name == "uniform":
        return torch.randn(N, h, dtype=dtype, device=device)
    if name == "skewed":
        return (torch.randn(N, h, dtype=dtype, device=device) * 0.5 + 2.0)
    if name == "collapsed":
        return (torch.randn(N, h, dtype=dtype, device=device) * 0.05)
    raise ValueError(name)


def _time_callable(
    fn: Callable[[], torch.Tensor],
    warmup: int = 3,
    iters: int = 20,
) -> float:
    """Return median wall in ms over ``iters`` calls (after warmup)."""
    for _ in range(warmup):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().to(torch.float32)
    b32 = b.detach().to(torch.float32)
    rms_a = a32.pow(2).mean().sqrt().clamp_min(1e-12)
    rms_diff = (a32 - b32).pow(2).mean().sqrt()
    return float((rms_diff / rms_a).item())


def _abs_max(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _bench_baseline_gemm(
    N: int, h: int, r: int, dtype: torch.dtype, device: torch.device,
) -> float:
    """Wall of a single bf16/fp16 GEMM of shape (N, h) @ (h, r).

    Used as a reference: per v7 §R.4.4 the kernel should be within ~1.5×
    of this in eventual production tuning.
    """
    A = torch.randn(N, h, dtype=dtype, device=device)
    B = torch.randn(h, r, dtype=dtype, device=device)
    return _time_callable(lambda: A @ B)


def run_workload(
    name: str, *, N: int, h: int, r: int, G: int,
    dtype: torch.dtype, device: torch.device, grid_lo: float, grid_hi: float,
) -> dict:
    L = G + 2
    z = _gen_workload(name, N, h, dtype, device, grid_lo, grid_hi)
    C = (torch.randn(h, L, r, dtype=dtype, device=device) * 0.1)

    # Lazy imports
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference as ref_fwd,
    )
    if device.type == "cuda":
        from sparsespline_ffn.kernels.triton_flash_spline_feature import (
            flash_spline_feature_forward as kern_fwd,
        )
    else:
        kern_fwd = None

    # ---- Correctness
    f_ref = ref_fwd(z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G)
    if kern_fwd is not None:
        f_kern = kern_fwd(z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G)
    else:
        f_kern = f_ref.clone()

    # split base half (phi) and delta half — most numerical disagreement
    # lives in delta because of fp32 internal accumulation
    a_ref = f_ref[:, :h]; a_kern = f_kern[:, :h]
    d_ref = f_ref[:, h:]; d_kern = f_kern[:, h:]
    metrics = {
        "abs_max_phi":     _abs_max(a_kern, a_ref),
        "rel_rms_phi":     _rel_rms(a_kern, a_ref),
        "abs_max_delta":   _abs_max(d_kern, d_ref),
        "rel_rms_delta":   _rel_rms(d_kern, d_ref),
    }

    # ---- Speed (median ms)
    if kern_fwd is not None:
        ms_kern = _time_callable(
            lambda: kern_fwd(z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G)
        )
    else:
        ms_kern = float("nan")
    ms_ref = _time_callable(
        lambda: ref_fwd(z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G)
    )
    if device.type == "cuda":
        ms_gemm = _bench_baseline_gemm(N, h, r, dtype, device)
    else:
        ms_gemm = float("nan")

    metrics.update({
        "ms_kernel": ms_kern,
        "ms_reference": ms_ref,
        "ms_baseline_gemm": ms_gemm,
        "speedup_vs_reference":
            (ms_ref / ms_kern) if ms_kern == ms_kern and ms_kern > 0 else float("nan"),
        "ratio_vs_gemm":
            (ms_kern / ms_gemm) if (ms_kern == ms_kern and ms_gemm == ms_gemm
                                     and ms_gemm > 0) else float("nan"),
    })
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["small", "med", "big"], default="med",
                    help="Workload shape preset")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.shape == "small":
        N, h, r, G = 64, 128, 16, 10
    elif args.shape == "med":
        N, h, r, G = 512, 768, 64, 20
    else:  # big
        N, h, r, G = 2048, 1024, 64, 22

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available; falling back to CPU (kernel skipped).")
        device = torch.device("cpu")

    print(f"\n=== FlashSplineFeature microbench ({args.shape}) ===")
    print(f"  shape: N={N} h={h} r={r} G={G}  dtype={args.dtype}  device={device}")
    print()

    results = {}
    print(f"{'workload':<11} {'rel_rms_d':>10} {'abs_max_d':>10} "
          f"{'ms_ref':>8} {'ms_kern':>8} {'ms_gemm':>8} "
          f"{'speedup':>8} {'/gemm':>7}")
    print("-" * 80)
    for w in ["uniform", "skewed", "collapsed"]:
        m = run_workload(w, N=N, h=h, r=r, G=G, dtype=dtype, device=device,
                          grid_lo=-3.0, grid_hi=3.0)
        results[w] = m
        print(f"{w:<11} {m['rel_rms_delta']:>10.3e} {m['abs_max_delta']:>10.3e} "
              f"{m['ms_reference']:>8.2f} {m['ms_kernel']:>8.2f} "
              f"{m['ms_baseline_gemm']:>8.2f} "
              f"{m['speedup_vs_reference']:>7.2f}x {m['ratio_vs_gemm']:>6.2f}x")

    print()
    print("Gates (v7 §R.4.4 / §R.6.9):")
    for w, m in results.items():
        rel = m["rel_rms_delta"]
        gemm_ratio = m["ratio_vs_gemm"]
        ok_corr = rel <= 5e-2 if dtype != torch.float32 else rel <= 1e-4
        bound = 5.0 if w == "collapsed" else 2.0
        ok_speed = (gemm_ratio != gemm_ratio) or (gemm_ratio <= bound) or \
                   (m["ms_kernel"] <= bound * m["ms_reference"])
        tag_corr = "PASS" if ok_corr else "FAIL"
        tag_spd = "PASS" if ok_speed else "FAIL"
        print(f"  [{w:<11}]  correctness={tag_corr}  "
              f"speed_vs_gemm({bound}x)={tag_spd}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps({
            "shape": {"N": N, "h": h, "r": r, "G": G,
                      "dtype": args.dtype, "device": str(device)},
            "workloads": results,
        }, indent=2))
        print(f"\n[json written to {args.json_out}]")


if __name__ == "__main__":
    main()
