"""H100 sweep over (r, L) kernel shapes — sprint task 2.

For Phase 2 v7 we want r=32 L=16 to be the smallest production shape;
r=64 L=22 the quality-leaning variant.  This bench compares the v4
forward kernel + B2.4 backward kernel against MLP at the relevant
shapes.

Shapes to sweep:
  (r=32, L=16)   smallest — speed/VRAM win candidate
  (r=32, L=22)   medium
  (r=64, L=22)   quality-leaning

Outputs ms_fwd, ms_bwd, peak_MB for each shape, all on H100.
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy")
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[
            ".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
            "**/__pycache__/**", "**/*.pyc",
        ],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-shape-sweep-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_sweep(d: int = 768, b: int = 2, t: int = 1024) -> str:
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch

    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_forward_v4 as kernel_fwd,
        flash_spline_delta_backward as kernel_bwd,
    )

    N = b * t

    def median_ms(fn, warmup=8, iters=40):
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

    print(f"\n=== H100 (r, L) shape sweep ===")
    print(f"  d={d} N=B*T={N} dtype=bf16")
    print(f"  GPU: {torch.cuda.get_device_name(0)}\n")

    print(f"{'r':>3} {'L':>3} {'G':>3}  {'fwd_ms':>8} {'bwd_ms':>8} "
          f"{'gemm_ms':>8} {'fwd/g':>7} {'tot/g':>7} {'C_MB':>7}")
    print("-" * 70)
    results = []
    for r, L in [(32, 16), (32, 22), (64, 16), (64, 22)]:
        G = L - 2
        h = d
        z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
        C = (torch.randn(h, L, r, device="cuda", dtype=torch.bfloat16) * 0.1)
        g = torch.randn(N, r, device="cuda", dtype=torch.bfloat16)

        # Warm autotune
        kernel_fwd(z, C, -3.0, 3.0, G)
        kernel_bwd(z, C, g, -3.0, 3.0, G)

        ms_fwd = median_ms(lambda: kernel_fwd(z, C, -3.0, 3.0, G))
        ms_bwd = median_ms(lambda: kernel_bwd(z, C, g, -3.0, 3.0, G))

        # Reference GEMM same shape (just (N, h) @ (h, r))
        A = torch.randn(N, h, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(h, r, dtype=torch.bfloat16, device="cuda")
        ms_gemm = median_ms(lambda: A @ B)

        c_mb = h * L * r * 2 / 1024**2  # bf16 size of C

        print(f"{r:>3} {L:>3} {G:>3}  {ms_fwd:>8.3f} {ms_bwd:>8.3f} "
              f"{ms_gemm:>8.3f} {ms_fwd/ms_gemm:>6.2f}x "
              f"{(ms_fwd+ms_bwd)/ms_gemm:>6.2f}x {c_mb:>6.2f}")
        results.append({
            "r": r, "L": L, "G": G,
            "fwd_ms": ms_fwd, "bwd_ms": ms_bwd, "gemm_ms": ms_gemm,
            "fwd_over_gemm": ms_fwd / ms_gemm,
            "total_over_gemm": (ms_fwd + ms_bwd) / ms_gemm,
            "c_table_MB": c_mb,
        })

    return json.dumps({
        "shape": {"d": d, "N": N},
        "results": results,
    }, indent=2)


@app.local_entrypoint()
def main(d: int = 768, b: int = 2, t: int = 1024) -> None:
    print(f"H100 (r, L) sweep: d={d} B={b} T={t}")
    out = run_sweep.remote(d=d, b=b, t=t)
    print(out)
