"""Modal H100 microbench for FlashSplineFeature kernel versions.

Runs v1/v2/v3/v4 head-to-head on H100 against:
  - PyTorch reference forward (numerical oracle)
  - bf16 GEMM same shape (production target: kernel ≤ 1.5× GEMM per v7 §R.4.4)

No training data needed — kernel takes random tensors.  ~5 min H100 wall
including Modal cold start + autotune compile.

Usage:
  modal run benchmarks/modal_h100_kernel_bench.py
  modal run benchmarks/modal_h100_kernel_bench.py --shape big
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.9.1",
        "triton",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install("numpy", "pytest")
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[
            ".venv/**", ".git/**",
            "nanochat/**",  # not needed for kernel bench
            "benchmark_runs/**",
            "**/__pycache__/**", "**/*.pyc",
        ],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)

app = modal.App("flash-spline-feature-h100-bench", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_bench(shape: str = "med") -> str:
    import os, sys, time, json, io, contextlib
    sys.path.insert(0, "/repo/src")
    import torch

    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward,
    )
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference,
    )

    if shape == "small":
        N, h, r, G = 64, 128, 16, 10
    elif shape == "med":
        N, h, r, G = 512, 768, 64, 20
    elif shape == "big":
        N, h, r, G = 2048, 1024, 64, 22
    else:
        raise ValueError(shape)
    L = G + 2

    print(f"\n=== Modal H100 kernel bench (shape={shape}) ===", flush=True)
    print(f"  N={N} h={h} r={r} L={L}  device=cuda  dtype=bf16", flush=True)
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  HBM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)
    print(f"  SM count: {torch.cuda.get_device_properties(0).multi_processor_count}", flush=True)

    def median_ms(fn, warmup=8, iters=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)
        samples.sort()
        return samples[len(samples) // 2]

    results = {}
    for wname, gen in [
        ("uniform",   lambda: torch.randn(N, h, dtype=torch.bfloat16, device="cuda")),
        ("skewed",    lambda: torch.randn(N, h, dtype=torch.bfloat16, device="cuda")*0.5+2.0),
        ("collapsed", lambda: torch.randn(N, h, dtype=torch.bfloat16, device="cuda")*0.05),
    ]:
        z = gen()
        C = (torch.randn(h, L, r, dtype=torch.bfloat16, device="cuda")*0.1)

        # Warm autotune for v2/v3/v4 (suppresses tens of seconds of compile log)
        for v in ["v2", "v3", "v4"]:
            print(f"  [{wname}] warming {v} autotune...", flush=True)
            flash_spline_feature_forward(z, C, grid_lo=-3, grid_hi=3, G=G, version=v)

        # Numerical check vs reference
        f_ref  = flash_spline_feature_reference(z, C, grid_lo=-3, grid_hi=3, G=G)
        f_v1   = flash_spline_feature_forward(z, C, grid_lo=-3, grid_hi=3, G=G, version="v1")
        f_v4   = flash_spline_feature_forward(z, C, grid_lo=-3, grid_hi=3, G=G, version="v4")
        rel_v4_ref = ((f_v4 - f_ref).pow(2).mean().sqrt()
                       / f_ref.pow(2).mean().sqrt()).item()
        rel_v4_v1 = ((f_v4 - f_v1).pow(2).mean().sqrt()
                      / f_v1.pow(2).mean().sqrt()).item()

        # Speed
        ms = {}
        for v in ["v1", "v2", "v3", "v4"]:
            ms[v] = median_ms(lambda: flash_spline_feature_forward(
                z, C, grid_lo=-3, grid_hi=3, G=G, version=v))
        A = torch.randn(N, h, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(h, r, dtype=torch.bfloat16, device="cuda")
        ms_gemm = median_ms(lambda: A @ B)

        line = (f"  {wname:<11}  v1={ms['v1']:.3f}  v2={ms['v2']:.3f}  "
                f"v3={ms['v3']:.3f}  v4={ms['v4']:.3f}  gemm={ms_gemm:.3f}  "
                f"v4/v1={ms['v4']/ms['v1']:.2f}x  v4/gemm={ms['v4']/ms_gemm:.1f}x  "
                f"rel_v4_ref={rel_v4_ref:.2e}")
        print(line, flush=True)
        results[wname] = {
            "v1_ms": ms["v1"], "v2_ms": ms["v2"],
            "v3_ms": ms["v3"], "v4_ms": ms["v4"],
            "gemm_ms": ms_gemm,
            "rel_v4_ref": rel_v4_ref, "rel_v4_v1": rel_v4_v1,
        }

    print("\n=== Summary ===")
    print(f"{'workload':<11} {'v1':>7} {'v2':>7} {'v3':>7} {'v4':>7} "
          f"{'gemm':>7} {'v4/gemm':>8} {'best/gemm':>10}")
    for w, r_w in results.items():
        ms_min = min(r_w[f"{v}_ms"] for v in ("v1","v2","v3","v4"))
        print(f"{w:<11} {r_w['v1_ms']:>7.3f} {r_w['v2_ms']:>7.3f} "
              f"{r_w['v3_ms']:>7.3f} {r_w['v4_ms']:>7.3f} {r_w['gemm_ms']:>7.3f} "
              f"{r_w['v4_ms']/r_w['gemm_ms']:>7.2f}x {ms_min/r_w['gemm_ms']:>9.2f}x")

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(shape: str = "med") -> None:
    print(f"Launching H100 kernel bench (shape={shape})")
    out = run_bench.remote(shape=shape)
    print(out)
