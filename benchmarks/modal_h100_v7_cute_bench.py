"""H100 speed bench: v7 cute bwd vs v5 vs Triton at production geometry.

Times each kernel with CUDA events: 10 warmup + 100 timed iterations, take
median ms / call. Reports the speedup ratios.

Gate (per the v7 plan):
  v7/Triton >= 1.15  -> continue to Phase 5 (warp-spec) confidently
  v7/Triton in [1.10, 1.15) -> yellow zone, need Nsight to decide
  v7/Triton < 1.10  -> stop, parity-only

Image is the same as v7_cute_parity (CUTLASS clone + PR#2171 leaf patch).
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "ninja")
    .run_commands(
        "git clone --depth 1 --branch v3.6.0 "
        "https://github.com/NVIDIA/cutlass.git /opt/cutlass"
    )
    .run_commands(
        "perl -i -0777 -pe "
        "'s/CUTE_DEVICE\\s+uint32_t\\s+cast_smem_ptr_to_uint/"
        "CUTE_HOST_DEVICE\\nuint32_t\\ncast_smem_ptr_to_uint/g' "
        "/opt/cutlass/include/cute/arch/util.hpp"
    )
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "dispatcher_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-v7-cute-bench-h100", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
def run_bench(N: int = 32768, H: int = 2560, L: int = 22, R: int = 32,
              warmup: int = 10, iters: int = 100) -> dict:
    import sys, json, time
    sys.path.insert(0, "/repo/src")
    import torch
    from torch.utils.cpp_extension import load

    print(f"\n{'=' * 72}", flush=True)
    print(f"  v7 cute bench vs v5 vs Triton — N={N}, H={H}, L={L}, R={R}", flush=True)
    print(f"  warmup={warmup}, iters={iters}", flush=True)
    print(f"{'=' * 72}", flush=True)

    # ---- Load v7 cute ----
    print("Compiling spline_kv_bwd_v7_cute.cu ...", flush=True)
    ext_v7 = load(
        name="spline_kv_bwd_v7_cute_ext",
        sources=["/repo/src/sparsespline_ffn/cuda_ext/spline_kv_bwd_v7_cute.cu"],
        extra_include_paths=[
            "/opt/cutlass/include",
            "/opt/cutlass/tools/util/include",
        ],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math",
            "-gencode", "arch=compute_90a,code=sm_90a",
            "-std=c++17", "--extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xptxas=-v", "-lineinfo",
            "-DCUTE_USE_PACKED_TUPLE=1",
        ],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )

    from sparsespline_ffn.cuda_ext import spline_kv_bwd_wgmma_v5_cuda
    # Triton bwd via the v3 (tensor-core tl.dot) implementation.
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward_v3 as triton_bwd,
    )

    # ---- Random inputs ----
    torch.manual_seed(0)
    device = torch.device("cuda")
    z       = torch.randn(N, H,        dtype=torch.bfloat16, device=device).contiguous() * 0.5
    C       = torch.randn(H, L, R,     dtype=torch.bfloat16, device=device).contiguous() * 0.1
    g_delta = torch.randn(N, R,        dtype=torch.bfloat16, device=device).contiguous() * 0.5

    grid_lo = -3.0
    grid_hi =  3.0
    G       = L - 2
    scale   = G / (grid_hi - grid_lo)

    # ---- Helpers ----
    def time_kernel(fn, warmup_iters=warmup, timed_iters=iters):
        # CUDA-event timing.
        for _ in range(warmup_iters):
            fn()
        torch.cuda.synchronize()
        ev_start = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
        ev_end   = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
        for i in range(timed_iters):
            ev_start[i].record()
            fn()
            ev_end[i].record()
        torch.cuda.synchronize()
        times = sorted([ev_start[i].elapsed_time(ev_end[i]) for i in range(timed_iters)])
        median = times[timed_iters // 2]
        return median, times[0], times[-1]

    out = {"N": N, "H": H, "L": L, "R": R}

    # ---- v5 reference ----
    print("Timing v5 ...", flush=True)
    def f_v5():
        spline_kv_bwd_wgmma_v5_cuda(z, C, g_delta, grid_lo, grid_hi, G)
    v5_med, v5_min, v5_max = time_kernel(f_v5)
    out["v5_ms"]     = v5_med
    out["v5_min_ms"] = v5_min
    out["v5_max_ms"] = v5_max
    print(f"  v5      : median={v5_med:.4f} ms  min={v5_min:.4f}  max={v5_max:.4f}", flush=True)

    # ---- v7 cute ----
    print("Timing v7 cute ...", flush=True)
    def f_v7():
        ext_v7.spline_kv_bwd_v7_cute_cuda(z, C, g_delta, grid_lo, scale, L)
    v7_med, v7_min, v7_max = time_kernel(f_v7)
    out["v7_ms"]     = v7_med
    out["v7_min_ms"] = v7_min
    out["v7_max_ms"] = v7_max
    print(f"  v7 cute : median={v7_med:.4f} ms  min={v7_min:.4f}  max={v7_max:.4f}", flush=True)

    # ---- Triton ----
    print("Timing Triton ...", flush=True)
    def f_triton():
        triton_bwd(z, C, g_delta, grid_lo, grid_hi, G)
    try:
        tr_med, tr_min, tr_max = time_kernel(f_triton)
        out["triton_ms"]     = tr_med
        out["triton_min_ms"] = tr_min
        out["triton_max_ms"] = tr_max
        print(f"  triton  : median={tr_med:.4f} ms  min={tr_min:.4f}  max={tr_max:.4f}", flush=True)
    except Exception as e:
        print(f"  triton  : FAILED ({e})", flush=True)
        out["triton_ms"] = None

    # ---- Speedup ratios ----
    print("\n" + "=" * 72, flush=True)
    print("  RATIOS (higher = v7 wins)", flush=True)
    print("=" * 72, flush=True)
    if out.get("triton_ms"):
        out["v7_over_triton"] = out["triton_ms"] / out["v7_ms"]
        out["v5_over_triton"] = out["triton_ms"] / out["v5_ms"]
        out["v7_over_v5"]     = out["v5_ms"]     / out["v7_ms"]
        print(f"  v7 / triton : {out['v7_over_triton']:.3f}×", flush=True)
        print(f"  v5 / triton : {out['v5_over_triton']:.3f}×", flush=True)
        print(f"  v7 / v5     : {out['v7_over_v5']:.3f}×", flush=True)

        gate_pass = out["v7_over_triton"] >= 1.15
        gate_yellow = out["v7_over_triton"] >= 1.10
        if gate_pass:
            verdict = "PASS  (>=1.15x Triton — continue to Phase 5 warp-spec)"
        elif gate_yellow:
            verdict = "YELLOW (1.10-1.15x — need Nsight to decide)"
        else:
            verdict = "STOP   (<1.10x — single-stage v7 is the ceiling on this geometry)"
        print(f"  verdict     : {verdict}", flush=True)
        out["gate"] = "pass" if gate_pass else ("yellow" if gate_yellow else "stop")

    print("\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main(n: int = 32768, h: int = 2560, l: int = 22, r: int = 32,
         warmup: int = 10, iters: int = 100):
    out = run_bench.remote(N=n, H=h, L=l, R=r, warmup=warmup, iters=iters)
    if out.get("gate") == "stop":
        raise SystemExit(2)
