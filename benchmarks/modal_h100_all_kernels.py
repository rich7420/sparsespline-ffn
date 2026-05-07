"""H100 unified bench: Triton v3 vs hopper-wmma vs wgmma."""
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
app = modal.App("sparsespline-allkernels-h100", image=IMAGE)


@app.function(gpu="H100", timeout=2400)
def run_bench() -> str:
    import sys, time, json, traceback
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.kernels.triton_flash_spline_feature import flash_spline_delta_backward_v3
    from sparsespline_ffn.kernels.flash_spline_feature_backward_ref import flash_spline_delta_backward_ref
    from sparsespline_ffn.cuda_ext import (
        spline_kv_bwd_hopper_cuda,
        spline_kv_bwd_wgmma_cuda,
    )

    print("Compiling hopper kernel...")
    z0 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    C0 = torch.randn(64, 16, 32, device="cuda", dtype=torch.bfloat16) * 0.1
    g0 = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    spline_kv_bwd_hopper_cuda(z0, C0, g0, -3.0, 3.0, 14)
    print("hopper ok.")

    wgmma_works = False
    print("Compiling wgmma kernel...")
    try:
        spline_kv_bwd_wgmma_cuda(z0, C0, g0, -3.0, 3.0, 14)
        torch.cuda.synchronize()
        wgmma_works = True
        print("wgmma compiled and ran ok.")
    except Exception as e:
        print(f"wgmma FAILED: {e}")
        traceback.print_exc()

    def med_ms(fn, w=10, it=100):
        for _ in range(w): fn()
        torch.cuda.synchronize(); s = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            s.append((time.perf_counter()-t0)*1000)
        s.sort(); return s[len(s)//2]

    print(f"\n=== H100 all-kernel bench ===")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    print(f"{'shape':<12} {'v3':>9} {'hopp':>9} {'wgmma':>9} "
          f"{'v3/hopp':>8} {'v3/wgmma':>9} {'rel_dC_h':>10} {'rel_dC_w':>10}")
    print("-" * 88)
    results = []
    for r, L in [(32, 16), (32, 22), (64, 16), (64, 22)]:
        G = L - 2
        N, h = 2048, 768
        torch.manual_seed(0)
        z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
        C = (torch.randn(h, L, r, device="cuda", dtype=torch.bfloat16) * 0.1)
        g = torch.randn(N, r, device="cuda", dtype=torch.bfloat16)

        flash_spline_delta_backward_v3(z, C, g, -3.0, 3.0, G)
        spline_kv_bwd_hopper_cuda(z, C, g, -3.0, 3.0, G)

        dC_ref, _ = flash_spline_delta_backward_ref(z.float(), C.float(), g.float(), -3.0, 3.0, G)
        dC_h, _ = spline_kv_bwd_hopper_cuda(z, C, g, -3.0, 3.0, G)
        rel_dC_h = ((dC_h - dC_ref).pow(2).mean().sqrt() / dC_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()

        rel_dC_w = -1.0
        tw = -1.0
        if wgmma_works:
            try:
                spline_kv_bwd_wgmma_cuda(z, C, g, -3.0, 3.0, G)
                dC_w, _ = spline_kv_bwd_wgmma_cuda(z, C, g, -3.0, 3.0, G)
                rel_dC_w = ((dC_w - dC_ref).pow(2).mean().sqrt() / dC_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
                tw = med_ms(lambda: spline_kv_bwd_wgmma_cuda(z, C, g, -3.0, 3.0, G))
            except Exception as e:
                print(f"  wgmma error on r={r} L={L}: {e}")

        t3 = med_ms(lambda: flash_spline_delta_backward_v3(z, C, g, -3.0, 3.0, G))
        th = med_ms(lambda: spline_kv_bwd_hopper_cuda(z, C, g, -3.0, 3.0, G))
        v3_hopp = t3 / th if th > 0 else 0.0
        v3_wgmma = t3 / tw if tw > 0 else 0.0
        line = (f"r={r:<2} L={L:<2}  {t3:>9.4f} {th:>9.4f} {tw:>9.4f} "
                f"{v3_hopp:>7.2f}x {v3_wgmma:>8.2f}x {rel_dC_h:>10.2e} {rel_dC_w:>10.2e}")
        print(line)
        results.append({
            "r": r, "L": L,
            "t3_ms": t3, "thopp_ms": th, "twgmma_ms": tw,
            "rel_dC_h": rel_dC_h, "rel_dC_w": rel_dC_w,
        })
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main() -> None:
    print(run_bench.remote())
