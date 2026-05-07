"""H100 speed bench: v7ws (warp-spec) vs v5 vs Triton at production geometry."""
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
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "dispatcher_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-v7ws-cute-bench-h100", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
def run_bench(N: int = 32768, H: int = 2560, L: int = 22, R: int = 32,
              warmup: int = 10, iters: int = 100) -> dict:
    import sys, json
    sys.path.insert(0, "/repo/src")
    import torch
    from torch.utils.cpp_extension import load

    print(f"\n{'=' * 72}", flush=True)
    print(f"  v7ws bench — N={N}, H={H}, L={L}, R={R}", flush=True)
    print(f"{'=' * 72}", flush=True)

    print("Compiling spline_kv_bwd_v7ws_cute.cu ...", flush=True)
    ext = load(
        name="spline_kv_bwd_v7ws_cute_ext",
        sources=["/repo/src/sparsespline_ffn/cuda_ext/spline_kv_bwd_v7ws_cute.cu"],
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
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward_v3 as triton_bwd,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    z       = torch.randn(N, H,        dtype=torch.bfloat16, device=device).contiguous() * 0.5
    C       = torch.randn(H, L, R,     dtype=torch.bfloat16, device=device).contiguous() * 0.1
    g_delta = torch.randn(N, R,        dtype=torch.bfloat16, device=device).contiguous() * 0.5

    grid_lo = -3.0
    grid_hi =  3.0
    G       = L - 2
    scale   = G / (grid_hi - grid_lo)

    def time_kernel(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            ev_s[i].record(); fn(); ev_e[i].record()
        torch.cuda.synchronize()
        times = sorted([ev_s[i].elapsed_time(ev_e[i]) for i in range(iters)])
        return times[iters // 2], times[0], times[-1]

    out = {"N": N, "H": H, "L": L, "R": R}

    print("Timing v5 ...", flush=True)
    v5_med, v5_lo, v5_hi = time_kernel(
        lambda: spline_kv_bwd_wgmma_v5_cuda(z, C, g_delta, grid_lo, grid_hi, G))
    out["v5_ms"] = v5_med
    print(f"  v5     : median={v5_med:.4f} ms  [{v5_lo:.4f}, {v5_hi:.4f}]", flush=True)

    print("Timing v7ws ...", flush=True)
    v7ws_med, v7ws_lo, v7ws_hi = time_kernel(
        lambda: ext.spline_kv_bwd_v7ws_cute_cuda(z, C, g_delta, grid_lo, scale, L))
    out["v7ws_ms"] = v7ws_med
    print(f"  v7ws   : median={v7ws_med:.4f} ms  [{v7ws_lo:.4f}, {v7ws_hi:.4f}]", flush=True)

    print("Timing Triton ...", flush=True)
    try:
        tr_med, tr_lo, tr_hi = time_kernel(
            lambda: triton_bwd(z, C, g_delta, grid_lo, grid_hi, G))
        out["triton_ms"] = tr_med
        print(f"  triton : median={tr_med:.4f} ms  [{tr_lo:.4f}, {tr_hi:.4f}]", flush=True)
    except Exception as e:
        print(f"  triton : FAILED ({e})", flush=True)
        out["triton_ms"] = None

    print("\n" + "=" * 72, flush=True)
    if out.get("triton_ms"):
        out["v7ws_over_triton"] = out["triton_ms"] / out["v7ws_ms"]
        out["v5_over_triton"]   = out["triton_ms"] / out["v5_ms"]
        out["v7ws_over_v5"]     = out["v5_ms"]     / out["v7ws_ms"]
        print(f"  v7ws / triton : {out['v7ws_over_triton']:.3f}×", flush=True)
        print(f"  v5   / triton : {out['v5_over_triton']:.3f}×", flush=True)
        print(f"  v7ws / v5     : {out['v7ws_over_v5']:.3f}×", flush=True)

        gate = "PASS-1.5" if out["v7ws_over_triton"] >= 1.5 else \
               "PASS-1.3" if out["v7ws_over_triton"] >= 1.3 else \
               "PASS-1.15" if out["v7ws_over_triton"] >= 1.15 else \
               "BELOW-1.15"
        print(f"  verdict       : {gate}", flush=True)
        out["gate"] = gate

    print("\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main(n: int = 32768, h: int = 2560, l: int = 22, r: int = 32,
         warmup: int = 10, iters: int = 100):
    out = run_bench.remote(N=n, H=h, L=l, R=r, warmup=warmup, iters=iters)
