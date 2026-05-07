"""H100 bench: v3 (per-j tl.dot) vs v5 (batched-j tl.dot).

v5 collapses BLOCK_J_BATCH j's into one big tl.dot for tensor-core
saturation.  Bigger M dim (BLOCK_J*L_PAD) should better utilize wmma.
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
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-bwd-v3v5-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_bench() -> str:
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward_v3,
        flash_spline_delta_backward_v5,
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

    print("\n=== H100 bwd v3 (per-j) vs v5 (batched-j) ===")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  bf16 inputs (production path)")
    print()
    print(f"{'shape':<14} {'v3_ms':>9} {'v5_ms':>9} {'v3/v5':>8} "
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

        # warm autotune
        flash_spline_delta_backward_v3(z, C, g, -3.0, 3.0, G)
        flash_spline_delta_backward_v5(z, C, g, -3.0, 3.0, G)

        # Correctness vs fp64-ish reference
        z_f, C_f, g_f = z.float(), C.float(), g.float()
        dC_ref, dz_ref = flash_spline_delta_backward_ref(z_f, C_f, g_f, -3.0, 3.0, G)
        dC_v5, dz_v5 = flash_spline_delta_backward_v5(z, C, g, -3.0, 3.0, G)
        rel_dC = ((dC_v5 - dC_ref).pow(2).mean().sqrt()
                  / dC_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()
        rel_dz = ((dz_v5 - dz_ref).pow(2).mean().sqrt()
                  / dz_ref.pow(2).mean().sqrt().clamp_min(1e-9)).item()

        t3 = med_ms(lambda: flash_spline_delta_backward_v3(z, C, g, -3.0, 3.0, G))
        t5 = med_ms(lambda: flash_spline_delta_backward_v5(z, C, g, -3.0, 3.0, G))

        line = (f"r={r:<2} L={L:<2}     {t3:>9.3f} {t5:>9.3f} "
                f"{t3/t5:>7.2f}x {rel_dC:>10.2e} {rel_dz:>10.2e}")
        print(line)
        results.append({"r": r, "L": L, "t3_ms": t3, "t5_ms": t5,
                        "speedup_v3_to_v5": t3 / t5,
                        "rel_dC": rel_dC, "rel_dz": rel_dz})
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main() -> None:
    print(run_bench.remote())
