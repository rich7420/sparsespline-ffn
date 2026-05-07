"""H100 deep forward profile — split CUDA vs Triton path, 12-layer scaling.

Three angles:
  1. Per-path single-FFN forward: Triton vs CUDA delta kernels
  2. 12-layer wall scaling — measure per-layer overhead growth
  3. Component breakdown of CUDA path (activation+delta+pack 3 sub-kernels)
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
app = modal.App("sparsespline-fwd-deep-profile-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_profile() -> str:
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch
    import torch.nn as nn
    from sparsespline_ffn.rl_spline_kv_reference import (
        RLSplineKVConfig, RLSplineKVReference,
    )
    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        FlashSplineFeature,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")

    B, T, d = 2, 1024, 384
    h = 2 * d   # h_ratio=2
    r = 32
    G = 20
    L = G + 2

    z = torch.randn(B * T, h, device=device, dtype=torch.bfloat16) * 1.5
    C = torch.randn(h, L, r, device=device, dtype=torch.bfloat16) * 0.1

    out = {"shape": {"B": B, "T": T, "d": d, "h": h, "r": r, "G": G, "L": L}}

    def med_ms(fn, w=10, it=80):
        for _ in range(w): fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            s.append((time.perf_counter() - t0) * 1000)
        s.sort(); return s[len(s) // 2]

    # ============= 1. Triton delta only (bf16 native) =============
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_forward_v4 as _triton_delta,
    )
    out["triton_delta_bf16_ms"] = med_ms(
        lambda: _triton_delta(z, C, grid_lo=-3.0, grid_hi=3.0, G=G)
    )
    # cast version (what the autograd code actually runs)
    out["triton_delta_path_with_cast_ms"] = med_ms(lambda: (
        _triton_delta(z, C, grid_lo=-3.0, grid_hi=3.0, G=G).to(torch.bfloat16)
    ))

    # ============= 2. CUDA delta+activation+pack (full spline_kv_fwd_cuda) =============
    from sparsespline_ffn.cuda_ext import spline_kv_fwd_cuda
    out["cuda_full_fwd_ms"] = med_ms(lambda: spline_kv_fwd_cuda(
        z, C, grid_lo=-3.0, grid_hi=3.0, G=G, activation="relu_sq", lambda_scale=1.0
    ))

    # ============= 3. PyTorch reference (CPU-style) =============
    # We don't bench reference (too slow); just sanity check correctness later.

    # ============= 4. FlashSplineFeature autograd (production path) =============
    # This is what the model actually uses.  Routes via fwd_kernel="auto" → CUDA;
    # we compare with fwd_kernel="triton".
    cfg_auto = RLSplineKVConfig(
        d=d, h_ratio=2.0, r=r, G=G, spline_order=2,
        use_kernel=True, bwd_kernel="wgmma_cuda", fwd_kernel="auto",
        gating_mode="additive",
    )
    cfg_triton = RLSplineKVConfig(
        d=d, h_ratio=2.0, r=r, G=G, spline_order=2,
        use_kernel=True, bwd_kernel="wgmma_cuda", fwd_kernel="triton",
        gating_mode="additive",
    )

    # Build single-layer FFN modules and time forward only
    rl_auto = RLSplineKVReference(cfg_auto).to(device).to(torch.bfloat16)
    rl_triton = RLSplineKVReference(cfg_triton).to(device).to(torch.bfloat16)

    x = torch.randn(B, T, d, device=device, dtype=torch.bfloat16)
    out["single_ffn_fwd_auto_cuda_ms"]   = med_ms(lambda: rl_auto(x))
    out["single_ffn_fwd_triton_ms"]      = med_ms(lambda: rl_triton(x))

    # ============= 5. 12-layer scaling test =============
    # Stack 12 of the same FFN module sequentially (no attention) — measures
    # whether per-layer time grows non-linearly due to memory pressure / cache
    # thrashing / sync overhead.

    class StackedFFN(nn.Module):
        def __init__(self, cfg, n_layers):
            super().__init__()
            self.layers = nn.ModuleList([
                RLSplineKVReference(cfg) for _ in range(n_layers)
            ])
        def forward(self, x):
            for ffn in self.layers:
                x = x + ffn(x)  # residual
            return x

    for n_layers in (1, 2, 4, 8, 12):
        # auto/cuda
        m = StackedFFN(cfg_auto, n_layers).to(device).to(torch.bfloat16)
        out[f"stack_fwd_auto_{n_layers}layer_ms"] = med_ms(lambda: m(x))
        del m
        # triton
        m = StackedFFN(cfg_triton, n_layers).to(device).to(torch.bfloat16)
        out[f"stack_fwd_triton_{n_layers}layer_ms"] = med_ms(lambda: m(x))
        del m
        torch.cuda.empty_cache()

    # ============= 6. CUDA path internal sub-kernels via PyTorch profiler =============
    print("=== CUDA path profiler breakdown ===", flush=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(50):
            spline_kv_fwd_cuda(z, C, grid_lo=-3.0, grid_hi=3.0, G=G,
                                activation="relu_sq", lambda_scale=1.0)
        torch.cuda.synchronize()

    cuda_subs = []
    for ev in prof.key_averages():
        if ev.device_time_total <= 0:
            continue
        cuda_subs.append({
            "op": ev.key[:80],
            "ms_per_call": (ev.device_time_total / 1000.0) / max(1, ev.count),
            "calls": ev.count,
        })
    cuda_subs.sort(key=lambda r: r["ms_per_call"] * r["calls"], reverse=True)
    out["cuda_fwd_subkernels"] = cuda_subs[:10]

    # ============= 7. Triton path internal sub-kernels via PyTorch profiler =============
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(50):
            d_v4 = _triton_delta(z, C, grid_lo=-3.0, grid_hi=3.0, G=G)
        torch.cuda.synchronize()
    triton_subs = []
    for ev in prof.key_averages():
        if ev.device_time_total <= 0:
            continue
        triton_subs.append({
            "op": ev.key[:80],
            "ms_per_call": (ev.device_time_total / 1000.0) / max(1, ev.count),
            "calls": ev.count,
        })
    triton_subs.sort(key=lambda r: r["ms_per_call"] * r["calls"], reverse=True)
    out["triton_fwd_subkernels"] = triton_subs[:10]

    print(json.dumps(out, indent=2), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main():
    print(run_profile.remote())
