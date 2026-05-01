"""H100 bench: v1 (atomic_add) vs v2 (partial-buffer + reduce) backward.

Goal: identify which approach wins on H100 for r=32 L=16, r=32 L=22, r=64 L=22.
Also measures the reduce-pass time separately so we can see if the
partial-buffer scheme is bottlenecked by reduction.
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
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-bwd-v1v2-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_bench() -> str:
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward,        # v1: atomic_add
        flash_spline_delta_backward_v2,     # v2: partial buffer + reduce
    )

    def med_ms(fn, warm=8, it=40):
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            s.append((time.perf_counter()-t0)*1000)
        s.sort(); return s[len(s)//2]

    print("\n=== H100 bwd v1 (atomic) vs v2 (partial-buf) ===")
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).multi_processor_count} SMs)")
    print()
    print(f"{'shape':<14} {'v1_ms':>9} {'v2_ms':>9} {'v1/v2':>7} "
          f"{'partial_MB':>11} {'rel_err':>10}")
    print("-" * 72)
    results = []
    for r, L in [(32, 16), (32, 22), (64, 16), (64, 22)]:
        G = L - 2
        N, h = 2048, 768
        torch.manual_seed(0)
        z = torch.randn(N, h, device="cuda", dtype=torch.float32)
        C = (torch.randn(h, L, r, device="cuda", dtype=torch.float32) * 0.1)
        g = torch.randn(N, r, device="cuda", dtype=torch.float32)

        # warm autotune
        flash_spline_delta_backward(z, C, g, -3.0, 3.0, G)
        flash_spline_delta_backward_v2(z, C, g, -3.0, 3.0, G)

        # correctness
        dC1, dz1 = flash_spline_delta_backward(z, C, g, -3.0, 3.0, G)
        dC2, dz2 = flash_spline_delta_backward_v2(z, C, g, -3.0, 3.0, G)
        rel = ((dC1 - dC2).pow(2).mean().sqrt()
               / dC1.pow(2).mean().sqrt().clamp_min(1e-9)).item()

        # timing
        t1 = med_ms(lambda: flash_spline_delta_backward(z, C, g, -3.0, 3.0, G))
        t2 = med_ms(lambda: flash_spline_delta_backward_v2(z, C, g, -3.0, 3.0, G))

        # partial buffer size (BLOCK_N=128 default in v2)
        BLOCK_N = 128
        N_CHUNKS = (N + BLOCK_N - 1) // BLOCK_N
        partial_mb = N_CHUNKS * h * L * r * 4 / 1024**2

        line = (f"r={r:<2} L={L:<2}     {t1:>9.3f} {t2:>9.3f} "
                f"{t1/t2:>6.2f}x {partial_mb:>10.1f} {rel:>10.2e}")
        print(line)
        results.append({
            "r": r, "L": L, "t1_ms": t1, "t2_ms": t2,
            "speedup_v1_to_v2": t1 / t2,
            "partial_MB": partial_mb, "rel_err": rel,
        })

    print()
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main() -> None:
    print(run_bench.remote())
