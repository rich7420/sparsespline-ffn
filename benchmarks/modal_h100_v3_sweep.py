"""H100 v3 autotune wider sweep.

Triggers Triton autotune over the expanded config space (now incl. BN=256
and num_stages=5) for production shapes, then reports best config and time.
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
app = modal.App("sparsespline-v3-sweep-h100", image=IMAGE)


@app.function(gpu="H100", timeout=2400)
def run_bench() -> str:
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward_v3,
        _flash_spline_feature_delta_bwd_v3,
    )

    def med_ms(fn, w=10, it=80):
        for _ in range(w): fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            s.append((time.perf_counter() - t0) * 1000)
        s.sort()
        return s[len(s)//2]

    print(f"=== H100 v3 wider autotune sweep ===")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    print(f"{'shape':<14} {'best_ms':>10} {'best_config':>50}")
    print("-" * 80)
    results = []
    for r, L in [(32, 16), (32, 22), (64, 16), (64, 22)]:
        G = L - 2
        N, h = 2048, 768
        torch.manual_seed(0)
        z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
        C = (torch.randn(h, L, r, device="cuda", dtype=torch.bfloat16) * 0.1)
        g = torch.randn(N, r, device="cuda", dtype=torch.bfloat16)

        # Trigger autotune (this compiles + benches all configs, slow first call)
        flash_spline_delta_backward_v3(z, C, g, -3.0, 3.0, G)
        torch.cuda.synchronize()

        # Read picked config from triton.autotune cache
        cache = _flash_spline_feature_delta_bwd_v3.cache
        try:
            picked = list(cache.values())[-1]
            cfg_str = (f"BN={picked.kwargs.get('BLOCK_N')} "
                       f"BH={picked.kwargs.get('BLOCK_H')} "
                       f"nw={picked.num_warps} "
                       f"ns={picked.num_stages}")
        except Exception:
            cfg_str = "(unknown)"

        t = med_ms(lambda: flash_spline_delta_backward_v3(z, C, g, -3.0, 3.0, G))
        line = f"r={r:<2} L={L:<2}     {t:>10.4f} {cfg_str:>50}"
        print(line)
        results.append({"r": r, "L": L, "ms": t, "config": cfg_str})

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main() -> None:
    print(run_bench.remote())
