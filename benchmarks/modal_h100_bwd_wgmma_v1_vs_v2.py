"""H100 bench + correctness: wgmma v1 (per-j atomic dC) vs wgmma v2 (output-parallel, no atomic).

v1: grid (N_TILE, H_TILE), 16-way atomicAdd contention into dC[h, L, R]
v2: grid (H_TILE, 1),       inner N-chunk loop, dC accumulated in SMEM, single bf16 store

Production shape: N=2048 H=384 L=22 R=32.
Reports max-abs-err, rel-err, and median-ms for both kernels.
"""
from __future__ import annotations

import modal


IMAGE = (
    # Full CUDA toolchain (nvcc) for JIT-compiling .cu extensions
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "ninja")
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-bwd-wgmma-v1v2-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_bench() -> str:
    import sys, time, json, math
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.cuda_ext import (
        spline_kv_bwd_wgmma_cuda,
        spline_kv_bwd_wgmma_v2_cuda,
    )
    from sparsespline_ffn.kernels.flash_spline_feature_backward_ref import (
        flash_spline_delta_backward_ref,
    )

    def med_ms(fn, w=8, it=40):
        for _ in range(w): fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            s.append((time.perf_counter()-t0)*1000)
        s.sort(); return s[len(s)//2]

    torch.manual_seed(0)
    device = torch.device("cuda")

    # Production shape (matches RL-KV B2 r32 L22 nanochat all12)
    N, H, L, R = 2048, 384, 22, 32
    G = L - 2  # 20
    grid_lo, grid_hi = -3.0, 3.0

    z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
    C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1
    g_delta = torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5
    # NOTE: production uses g_delta of shape [N, R] but kernel expects [N, R]
    # used as the gradient w.r.t. delta = sum_j W[n,j,...] @ C[j,...]; the
    # current bwd kernels assume shape [N, R] (one grad per token).

    # ---------------- correctness ----------------
    print("=== correctness ===", flush=True)
    dC_ref, dz_ref = flash_spline_delta_backward_ref(
        z.float(), C.float(), g_delta.float(),
        grid_lo=grid_lo, grid_hi=grid_hi, G=G,
    )

    dC_v1, dz_v1 = spline_kv_bwd_wgmma_cuda(
        z, C, g_delta, grid_lo, grid_hi, G,
    )
    dC_v2, dz_v2 = spline_kv_bwd_wgmma_v2_cuda(
        z, C, g_delta, grid_lo, grid_hi, G,
    )

    # v1 returns dC fp32, v2 returns dC bf16 — promote for compare
    dC_v1_f = dC_v1.float()
    dC_v2_f = dC_v2.float()

    def report(name, ours, ref):
        abs_err = (ours - ref).abs()
        rel = abs_err.max().item() / (ref.abs().max().item() + 1e-9)
        return {
            "kernel": name,
            "max_abs_err": abs_err.max().item(),
            "max_rel_err": rel,
            "mean_abs_err": abs_err.mean().item(),
        }

    out = {}
    out["dC_v1_vs_ref"] = report("v1", dC_v1_f, dC_ref.to(dC_v1_f.device))
    out["dC_v2_vs_ref"] = report("v2", dC_v2_f, dC_ref.to(dC_v2_f.device))
    out["dC_v2_vs_v1"]  = report("v2_vs_v1", dC_v2_f, dC_v1_f)
    out["dz_v1_vs_ref"] = report("dz_v1", dz_v1.float(), dz_ref.to(dz_v1.device))
    out["dz_v2_vs_ref"] = report("dz_v2", dz_v2.float(), dz_ref.to(dz_v2.device))
    out["dz_v2_vs_v1"]  = report("dz_v2_vs_v1", dz_v2.float(), dz_v1.float())

    # ---------------- speed ----------------
    print("=== speed ===", flush=True)
    t_v1 = med_ms(lambda: spline_kv_bwd_wgmma_cuda(z, C, g_delta, grid_lo, grid_hi, G))
    t_v2 = med_ms(lambda: spline_kv_bwd_wgmma_v2_cuda(z, C, g_delta, grid_lo, grid_hi, G))
    out["timing"] = {
        "v1_ms": t_v1, "v2_ms": t_v2,
        "speedup": t_v1 / t_v2 if t_v2 > 0 else 0.0,
    }
    print(json.dumps(out, indent=2), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main():
    res = run_bench.remote()
    print(res)
