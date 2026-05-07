"""H100 parity test for v7ws (warp-specialized) bwd kernel vs v5.

Same image as v7_cute_parity (CUTLASS clone + PR#2171 leaf patch).
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
app = modal.App("sparsespline-v7ws-cute-parity-h100", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
def run_parity(N: int = 2048, H: int = 64, L: int = 22, R: int = 32) -> dict:
    import sys, json
    sys.path.insert(0, "/repo/src")
    import torch
    from torch.utils.cpp_extension import load

    print(f"\n{'=' * 72}", flush=True)
    print(f"  v7ws (warp-spec) parity vs v5 — N={N}, H={H}", flush=True)
    print(f"{'=' * 72}", flush=True)

    print("Compiling spline_kv_bwd_v7ws_cute.cu ...", flush=True)
    ext_v7ws = load(
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
        verbose=True,
    )
    print("v7ws compile OK.", flush=True)

    from sparsespline_ffn.cuda_ext import spline_kv_bwd_wgmma_v5_cuda

    torch.manual_seed(0)
    device = torch.device("cuda")
    z       = torch.randn(N, H,        dtype=torch.bfloat16, device=device).contiguous() * 0.5
    C       = torch.randn(H, L, R,     dtype=torch.bfloat16, device=device).contiguous() * 0.1
    g_delta = torch.randn(N, R,        dtype=torch.bfloat16, device=device).contiguous() * 0.5

    grid_lo = -3.0
    grid_hi =  3.0
    G       = L - 2
    scale   = G / (grid_hi - grid_lo)

    print("Running v5 ref ...", flush=True)
    dC_v5, dz_v5 = spline_kv_bwd_wgmma_v5_cuda(z, C, g_delta, grid_lo, grid_hi, G)
    torch.cuda.synchronize()

    print("Running v7ws ...", flush=True)
    dC_v7, dz_v7 = ext_v7ws.spline_kv_bwd_v7ws_cute_cuda(
        z, C, g_delta, grid_lo, scale, L)
    torch.cuda.synchronize()

    dC_diff = (dC_v7.float() - dC_v5.float()).abs()
    dz_diff = (dz_v7.float() - dz_v5.float()).abs()

    out = {
        "dC_max_abs": dC_diff.max().item(),
        "dC_v5_max":  dC_v5.float().abs().max().item(),
        "dC_v7_max":  dC_v7.float().abs().max().item(),
        "dz_max_abs": dz_diff.max().item(),
        "dz_v5_max":  dz_v5.float().abs().max().item(),
        "dz_v7_max":  dz_v7.float().abs().max().item(),
    }
    out["dC_max_rel"] = out["dC_max_abs"] / max(out["dC_v5_max"], 1e-9)
    out["dz_max_rel"] = out["dz_max_abs"] / max(out["dz_v5_max"], 1e-9)
    passed = (out["dC_max_rel"] < 5e-3) and (out["dz_max_rel"] < 1e-3)
    out["passed"] = passed

    print(f"\n  dC : v5_max={out['dC_v5_max']:.4f}  v7_max={out['dC_v7_max']:.4f}", flush=True)
    print(f"       max_abs_err={out['dC_max_abs']:.6f}  max_rel_err={out['dC_max_rel']:.6f}", flush=True)
    print(f"  dz : v5_max={out['dz_v5_max']:.4f}  v7_max={out['dz_v7_max']:.4f}", flush=True)
    print(f"       max_abs_err={out['dz_max_abs']:.6f}  max_rel_err={out['dz_max_rel']:.6f}", flush=True)
    print(f"  passed: {'YES' if passed else 'NO'}", flush=True)

    print("\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main(n: int = 2048, h: int = 64, l: int = 22, r: int = 32):
    out = run_parity.remote(N=n, H=h, L=l, R=r)
    if not out.get("passed"):
        raise SystemExit(1)
