"""H100 bwd v4 (scratch + reduce, BN=128, no global atomic) vs v1 probe."""
from __future__ import annotations

import modal


IMAGE = (
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
app = modal.App("sparsespline-bwd-v4-probe-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_probe() -> str:
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.cuda_ext import (
        spline_kv_bwd_wgmma_cuda,         # v1
        spline_kv_bwd_wgmma_v4_cuda,       # v4
    )
    from sparsespline_ffn.kernels.flash_spline_feature_backward_ref import (
        flash_spline_delta_backward_ref,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")

    N, H, L, R = 2048, 768, 22, 32
    G = L - 2
    grid_lo, grid_hi = -3.0, 3.0

    z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
    C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1
    g_delta = torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5

    print("=== correctness ===", flush=True)
    dC_ref, dz_ref = flash_spline_delta_backward_ref(
        z.float(), C.float(), g_delta.float(),
        grid_lo=grid_lo, grid_hi=grid_hi, G=G,
    )
    dC_v1, dz_v1 = spline_kv_bwd_wgmma_cuda(z, C, g_delta, grid_lo, grid_hi, G)
    dC_v4, dz_v4 = spline_kv_bwd_wgmma_v4_cuda(z, C, g_delta, grid_lo, grid_hi, G)

    def report(name, ours, ref):
        ours_f = ours.float() if ours.dtype != torch.float32 else ours
        ref_f = ref.float() if ref.dtype != torch.float32 else ref
        diff = (ours_f - ref_f.to(ours_f.device)).abs()
        return {
            "kernel": name,
            "max_abs_err": float(diff.max().item()),
            "max_rel_err": float(diff.max().item() / (ref_f.abs().max().item() + 1e-9)),
            "mean_abs_err": float(diff.mean().item()),
        }

    out = {}
    out["dC_v1_vs_ref"] = report("v1", dC_v1, dC_ref)
    out["dC_v4_vs_ref"] = report("v4", dC_v4, dC_ref)
    out["dC_v4_vs_v1"]  = report("v4_vs_v1", dC_v4, dC_v1)
    out["dz_v1_vs_ref"] = report("dz_v1", dz_v1, dz_ref)
    out["dz_v4_vs_ref"] = report("dz_v4", dz_v4, dz_ref)
    out["dz_v4_vs_v1"]  = report("dz_v4_vs_v1", dz_v4, dz_v1)

    print("=== speed ===", flush=True)
    def med_ms(fn, w=10, it=100):
        for _ in range(w): fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            s.append((time.perf_counter() - t0) * 1000)
        s.sort(); return s[len(s) // 2]

    t_v1 = med_ms(lambda: spline_kv_bwd_wgmma_cuda(z, C, g_delta, grid_lo, grid_hi, G))
    t_v4 = med_ms(lambda: spline_kv_bwd_wgmma_v4_cuda(z, C, g_delta, grid_lo, grid_hi, G))
    out["timing"] = {
        "v1_ms": t_v1, "v4_ms": t_v4,
        "speedup": t_v1 / t_v4 if t_v4 > 0 else 0.0,
    }
    print(json.dumps(out, indent=2), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main():
    print(run_probe.remote())
