"""CUDA Graphs comparison: amortize launch overhead via graph capture.

Wraps each FFN variant with the existing CudaGraphFFN wrapper from
benchmarks/v_c_fusion_bench.py, then bench fwd+bwd as a single graph
replay.  Expected to compress launch-overhead-bound surrounding ops
significantly (we saw 1.5× for FullMix-Tucker).
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

# Import existing wrapper
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_c_fusion_bench import CudaGraphFFN

from sparsespline_ffn import MLPFFN
from sparsespline_ffn.simple_spline_mlp import SimpleSplineMLP, SimpleSplineConfig
from sparsespline_ffn.glu_ffn import SwiGLU, GLUConfig
from ffn_full_compare import _RLKVWrap


def median_ms(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        s.append((time.perf_counter() - t0) * 1000)
    s.sort(); return s[len(s) // 2]


def peak_after(fn, warmup=5, iters=10):
    """Measure peak VRAM after warmup."""
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        print("CUDA not available"); return
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    print(f"\n=== CUDA Graphs FFN compare ===")
    print(f"  d={args.d}  B={args.B}  T={args.T}  dtype={args.dtype}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    builders = {
        "mlp_h_4d":         lambda: MLPFFN(d=args.d, mlp_ratio=4),
        "mlp_h_d":          lambda: MLPFFN(d=args.d, mlp_ratio=1),
        "ss_h_d":           lambda: SimpleSplineMLP(SimpleSplineConfig(
                                d=args.d, h_ratio=1.0, G=20, use_kernel=True)),
        "swiglu_h_d":       lambda: SwiGLU(GLUConfig(d=args.d, mlp_ratio=1.0)),
        "rl_kv_r32_kernel": lambda: _RLKVWrap(d=args.d, r=32, use_kernel=True),
    }

    rows = []
    print(f"{'name':<18} {'ms_eager':>10} {'ms_graph':>10} {'speedup':>8} "
          f"{'peak_MB_eager':>14} {'peak_MB_graph':>14}")
    print("-" * 80)
    for name, builder in builders.items():
        torch.cuda.empty_cache(); gc.collect()
        torch.manual_seed(0)
        model = builder().to(device=device, dtype=dtype).train()
        target = torch.randn(args.B, args.T, args.d, device=device, dtype=dtype)
        x_const = torch.randn(args.B, args.T, args.d, device=device, dtype=dtype)

        # Eager step (uses x.requires_grad_)
        def eager_step():
            x = x_const.detach().requires_grad_(True)
            y = model(x)
            loss = (y - target).pow(2).sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            return loss

        try:
            ms_eager = median_ms(eager_step)
            peak_eager = peak_after(eager_step)
        except Exception as e:
            print(f"{name:<18}  eager FAIL: {e}")
            continue

        # CUDA graph version
        try:
            graph_ffn = CudaGraphFFN(builder().to(device=device, dtype=dtype),
                                       B=args.B, T=args.T, d=args.d,
                                       dtype=dtype, device=device, warmup_iters=5)
            def graph_step():
                graph_ffn.step(x_const)
            ms_graph = median_ms(graph_step)
            peak_graph = peak_after(graph_step)
        except Exception as e:
            print(f"{name:<18}  ms_eager={ms_eager:.3f}  graph FAIL: {e}")
            rows.append({"name": name, "ms_eager": ms_eager, "ms_graph": None,
                         "peak_eager": peak_eager})
            continue

        speedup = ms_eager / ms_graph
        print(f"{name:<18}  {ms_eager:>10.3f}  {ms_graph:>10.3f}  "
              f"{speedup:>6.2f}x  {peak_eager:>14.1f}  {peak_graph:>14.1f}")
        rows.append({
            "name": name, "ms_eager": ms_eager, "ms_graph": ms_graph,
            "speedup": speedup, "peak_eager": peak_eager, "peak_graph": peak_graph,
        })

        del model, graph_ffn
        torch.cuda.empty_cache(); gc.collect()

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\n[json written to {args.json_out}]")


if __name__ == "__main__":
    main()
