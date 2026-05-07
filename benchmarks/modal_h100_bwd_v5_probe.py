"""H100 v5 bwd kernel correctness + speed probe.

Compares v5 (register-resident dC + fp16 wgmma) against v1 (production)
on production-shape inputs.  Checks dC, dz match v1 within noise tolerance.

Metrics:
  - dC max_abs_err / max_rel_err / mean_signed_err
  - dz max_abs_err / max_rel_err / mean_signed_err
  - median wall ms vs v1
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
app = modal.App("sparsespline-bwd-v5-probe-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_probe() -> dict:
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.cuda_ext import (
        spline_kv_bwd_wgmma_cuda,     # v1
        spline_kv_bwd_wgmma_v5_cuda,  # new
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    out: dict = {}

    # Production-realistic shape (nanochat 124M h_ratio=2)
    N, H, L, R = 2048, 1536, 22, 32
    G = L - 2
    grid_lo, grid_hi = -3.0, 3.0

    z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
    C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1
    g = torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5

    print("=== correctness ===", flush=True)
    dC_v1, dz_v1 = spline_kv_bwd_wgmma_cuda(
        z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
    )
    dC_v5, dz_v5 = spline_kv_bwd_wgmma_v5_cuda(
        z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
    )

    def stat(name: str, ours: torch.Tensor, ref: torch.Tensor) -> dict:
        ours_f = ours.float(); ref_f = ref.float()
        diff = ours_f - ref_f
        diff_abs = diff.abs()
        ref_max = ref_f.abs().max().item()
        return {
            "name": name,
            "max_abs_err": float(diff_abs.max().item()),
            "max_rel_err": float(diff_abs.max().item() / (ref_max + 1e-9)),
            "mean_abs_err": float(diff_abs.mean().item()),
            "mean_signed_err": float(diff.mean().item()),
            "ref_max": float(ref_max),
        }

    out["dC"] = stat("dC v5-vs-v1", dC_v5, dC_v1)
    out["dz"] = stat("dz v5-vs-v1", dz_v5, dz_v1)
    for k, v in out.items():
        print(f"  {k}: signed={v['mean_signed_err']:+.3e} "
               f"max_abs={v['max_abs_err']:.3e} "
               f"max_rel={v['max_rel_err']:.3e} "
               f"mean_abs={v['mean_abs_err']:.3e}", flush=True)

    print("\n=== speed ===", flush=True)
    def med_ms(fn, w=10, it=50):
        for _ in range(w): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort(); return ts[len(ts) // 2]

    t_v1 = med_ms(lambda: spline_kv_bwd_wgmma_cuda(
        z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G))
    t_v5 = med_ms(lambda: spline_kv_bwd_wgmma_v5_cuda(
        z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G))
    out["timing"] = {
        "v1_ms": t_v1, "v5_ms": t_v5,
        "speedup": t_v1 / t_v5 if t_v5 > 0 else 0.0,
    }
    print(f"  v1: {t_v1:.4f} ms", flush=True)
    print(f"  v5: {t_v5:.4f} ms  ({t_v1/t_v5:.3f}x v1)", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main():
    print(run_probe.remote())
