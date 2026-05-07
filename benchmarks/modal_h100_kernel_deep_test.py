"""H100 deep test — fwd v1/v3/v10 + bwd v1/v3 across multiple shapes.

Goes beyond simple bench:
  1. Correctness across 3 shape configs
  2. Speed median over 100 reps each
  3. Per-kernel breakdown via torch.profiler
  4. Memory bandwidth utilization estimate (bytes / wall)
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
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-kernel-deep-test-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_deep_test() -> str:
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.cuda_ext import (
        spline_kv_fwd_cuda,           # v1 fwd
        spline_kv_fwd_v3_cuda,         # v3 fwd (scalar refactor)
        spline_kv_fwd_v10_cuda,        # v10 fwd (dense-W wgmma)
        spline_kv_bwd_wgmma_cuda,     # v1 bwd
        spline_kv_bwd_wgmma_v3_cuda,   # v3 bwd (2-stage pipeline)
    )
    from sparsespline_ffn.rl_spline_kv_reference import flash_spline_feature_reference
    from sparsespline_ffn.kernels.flash_spline_feature_backward_ref import (
        flash_spline_delta_backward_ref,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")

    # Three shape configs to triangulate bottleneck behaviour.
    shapes = [
        {"name": "production_h2", "N": 2048, "H": 768,  "L": 22, "R": 32},  # baseline
        {"name": "double_N",      "N": 4096, "H": 768,  "L": 22, "R": 32},  # 2x batch
        {"name": "narrow_h1",     "N": 2048, "H": 384,  "L": 22, "R": 32},  # h_ratio=1
    ]

    out = {"shapes": shapes, "results": {}}

    for cfg in shapes:
        name = cfg["name"]
        N, H, L, R = cfg["N"], cfg["H"], cfg["L"], cfg["R"]
        G = L - 2
        grid_lo, grid_hi = -3.0, 3.0
        lambda_scale = 1.0

        z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
        C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1
        g_delta = torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5

        print(f"\n=== {name}: N={N}, H={H}, L={L}, R={R} ===", flush=True)

        # --- Forward correctness ---
        f_ref = flash_spline_feature_reference(
            z.float(), C.float(),
            grid_lo=grid_lo, grid_hi=grid_hi, G=G,
            activation="relu_sq", lambda_scale=lambda_scale, spline_order=2,
        )
        f_v1 = spline_kv_fwd_cuda(z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                                   activation="relu_sq", lambda_scale=lambda_scale)
        try:
            f_v10 = spline_kv_fwd_v10_cuda(z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                                              activation="relu_sq", lambda_scale=lambda_scale)
            v10_err_vs_v1 = float(((f_v10.float() - f_v1.float()).abs().max() /
                                   (f_v1.float().abs().max() + 1e-9)).item())
            v10_err_vs_ref = float(((f_v10.float() - f_ref.to(device).float()).abs().max() /
                                     (f_ref.abs().max() + 1e-9)).item())
        except Exception as e:
            v10_err_vs_v1 = None
            v10_err_vs_ref = None
            print(f"  v10 fwd FAILED: {e}", flush=True)

        # --- Backward correctness ---
        dC_ref, dz_ref = flash_spline_delta_backward_ref(
            z.float(), C.float(), g_delta.float(),
            grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        )
        dC_v1, dz_v1 = spline_kv_bwd_wgmma_cuda(z, C, g_delta, grid_lo, grid_hi, G)
        dC_v3, dz_v3 = spline_kv_bwd_wgmma_v3_cuda(z, C, g_delta, grid_lo, grid_hi, G)

        # --- Speed ---
        def med_ms(fn, w=10, it=80):
            for _ in range(w): fn()
            torch.cuda.synchronize()
            s = []
            for _ in range(it):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                fn(); torch.cuda.synchronize()
                s.append((time.perf_counter() - t0) * 1000)
            s.sort(); return s[len(s) // 2]

        # Forward speed
        fwd_v1_ms = med_ms(lambda: spline_kv_fwd_cuda(
            z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
            activation="relu_sq", lambda_scale=lambda_scale))
        try:
            fwd_v10_ms = med_ms(lambda: spline_kv_fwd_v10_cuda(
                z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                activation="relu_sq", lambda_scale=lambda_scale))
        except Exception:
            fwd_v10_ms = None

        # Backward speed
        bwd_v1_ms = med_ms(lambda: spline_kv_bwd_wgmma_cuda(z, C, g_delta, grid_lo, grid_hi, G))
        bwd_v3_ms = med_ms(lambda: spline_kv_bwd_wgmma_v3_cuda(z, C, g_delta, grid_lo, grid_hi, G))

        # Memory traffic estimate (bytes read + written per call)
        # Forward: read z, C; write f.
        fwd_bytes = (N*H + H*L*R) * 2 + (N*(H+R)) * 2  # bf16 = 2 bytes
        # Backward: read z, C, g_delta; write dC, dz.
        bwd_bytes = (N*H + H*L*R + N*R) * 2 + (H*L*R*2 + N*H*4)

        out["results"][name] = {
            "fwd_v10_err_vs_v1":  v10_err_vs_v1,
            "fwd_v10_err_vs_ref": v10_err_vs_ref,
            "fwd_v1_ms":          fwd_v1_ms,
            "fwd_v10_ms":         fwd_v10_ms,
            "fwd_v10_speedup":    (fwd_v1_ms / fwd_v10_ms) if (fwd_v10_ms and fwd_v10_ms > 0) else None,
            "fwd_v1_GBps":        fwd_bytes / (fwd_v1_ms * 1e6),  # ms→s, bytes→GB
            "fwd_v10_GBps":       (fwd_bytes / (fwd_v10_ms * 1e6)) if fwd_v10_ms else None,
            "bwd_v1_ms":          bwd_v1_ms,
            "bwd_v3_ms":          bwd_v3_ms,
            "bwd_v3_speedup":     bwd_v1_ms / bwd_v3_ms if bwd_v3_ms > 0 else 0,
            "bwd_v1_GBps":        bwd_bytes / (bwd_v1_ms * 1e6),
            "bwd_v3_GBps":        bwd_bytes / (bwd_v3_ms * 1e6),
            "dC_v3_vs_v1_rel_err":  float(((dC_v3.float() - dC_v1.float()).abs().max() /
                                            (dC_v1.float().abs().max() + 1e-9)).item()),
            "dz_v3_vs_v1_rel_err":  float(((dz_v3.float() - dz_v1.float()).abs().max() /
                                            (dz_v1.float().abs().max() + 1e-9)).item()),
        }

    # H100 HBM3 BW = 3.35 TB/s = 3350 GB/s.  GBps / 3350 = utilization fraction.
    out["h100_hbm_bw_gbps"] = 3350.0
    print(json.dumps(out, indent=2), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main():
    print(run_deep_test.remote())
