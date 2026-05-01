"""H100: bf16-wmma CUDA bwd kernel vs Triton v3."""
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
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-wmma-h100", image=IMAGE)


@app.function(gpu="H100", timeout=2400)
def run_bench() -> str:
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.kernels.triton_flash_spline_feature import flash_spline_delta_backward_v3
    from sparsespline_ffn.kernels.flash_spline_feature_backward_ref import flash_spline_delta_backward_ref
    from sparsespline_ffn.cuda_ext import spline_kv_bwd_wmma_cuda

    print("Compiling wmma CUDA extension...")
    z0 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    C0 = torch.randn(64, 16, 32, device="cuda", dtype=torch.bfloat16) * 0.1
    g0 = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    spline_kv_bwd_wmma_cuda(z0, C0, g0, -3.0, 3.0, 14)
    torch.cuda.synchronize()
    print("wmma compiled.")

    def med_ms(fn, w=10, it=80):
        for _ in range(w): fn()
        torch.cuda.synchronize(); s = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            s.append((time.perf_counter()-t0)*1000)
        s.sort(); return s[len(s)//2]

    print(f"=== H100 Triton v3 vs CUDA wmma bwd ===")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    print(f"{'shape':<14} {'v3_ms':>9} {'wmma_ms':>9} {'v3/wmma':>9} "
          f"{'rel_dC':>10} {'rel_dz':>10}")
    print("-" * 64)
    results = []
    for r, L in [(32, 16), (32, 22), (64, 16), (64, 22)]:
        G = L - 2
        N, h = 2048, 768
        torch.manual_seed(0)
        z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
        C = (torch.randn(h, L, r, device="cuda", dtype=torch.bfloat16) * 0.1)
        g = torch.randn(N, r, device="cuda", dtype=torch.bfloat16)

        # warm
        flash_spline_delta_backward_v3(z, C, g, -3.0, 3.0, G)
        spline_kv_bwd_wmma_cuda(z, C, g, -3.0, 3.0, G)

        # correctness
        dC_ref, dz_ref = flash_spline_delta_backward_ref(z.float(), C.float(), g.float(), -3.0, 3.0, G)
        dC_w, dz_w = spline_kv_bwd_wmma_cuda(z, C, g, -3.0, 3.0, G)
        rel_dC = ((dC_w - dC_ref).pow(2).mean().sqrt() / dC_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
        rel_dz = ((dz_w - dz_ref).pow(2).mean().sqrt() / dz_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()

        t3 = med_ms(lambda: flash_spline_delta_backward_v3(z, C, g, -3.0, 3.0, G))
        tw = med_ms(lambda: spline_kv_bwd_wmma_cuda(z, C, g, -3.0, 3.0, G))
        line = (f"r={r:<2} L={L:<2}     {t3:>9.4f} {tw:>9.4f} {t3/tw:>8.2f}x "
                f"{rel_dC:>10.2e} {rel_dz:>10.2e}")
        print(line)
        results.append({"r": r, "L": L, "t3_ms": t3, "twmma_ms": tw,
                        "speedup": t3/tw, "rel_dC": rel_dC, "rel_dz": rel_dz})

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main() -> None:
    print(run_bench.remote())
