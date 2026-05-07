"""H100 forward v3 vs v1 probe — bit-equivalence + speed.

Validates spline_kv_fwd_v3_cuda matches spline_kv_fwd_cuda (v1) within
bf16 noise, then benchmarks both.

§0.1 of PLAN_KERNEL_REWRITE_v9.md mandates:
  - max_rel_err on f within 5e-3 (bf16 noise threshold)
  - max_abs_err on f bounded by typical magnitude × 5e-3
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
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-fwd-v3-probe-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_probe() -> str:
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.cuda_ext import (
        spline_kv_fwd_cuda,
        spline_kv_fwd_v3_cuda,
    )
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")

    # Production shape (matches RL-KV B2 r32 L22 nanochat all12 h_ratio=2)
    N, H, L, R = 2048, 768, 22, 32
    G = L - 2
    grid_lo, grid_hi = -3.0, 3.0
    lambda_scale = 1.0

    z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
    C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1

    # ---- correctness ----
    print("=== correctness ===", flush=True)
    f_ref = flash_spline_feature_reference(
        z.float(), C.float(),
        grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        activation="relu_sq", lambda_scale=lambda_scale, spline_order=2,
    )

    f_v1 = spline_kv_fwd_cuda(z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                                activation="relu_sq", lambda_scale=lambda_scale)
    f_v3 = spline_kv_fwd_v3_cuda(z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                                   activation="relu_sq", lambda_scale=lambda_scale)

    def report(name, ours, ref):
        ours_f = ours.float()
        ref_f = ref.float() if ref.is_cuda else ref.to(ours.device).float()
        diff = (ours_f - ref_f).abs()
        return {
            "kernel": name,
            "max_abs_err": float(diff.max().item()),
            "max_rel_err": float(diff.max().item() / (ref_f.abs().max().item() + 1e-9)),
            "mean_abs_err": float(diff.mean().item()),
        }

    out = {}
    out["v1_vs_ref"] = report("v1", f_v1, f_ref)
    out["v3_vs_ref"] = report("v3", f_v3, f_ref)
    out["v3_vs_v1"]  = report("v3_vs_v1", f_v3, f_v1)
    # Split into a-part and δ-part for clarity
    out["v3_vs_v1_a_part"]     = report("v3_vs_v1 [a]",  f_v3[:, :H], f_v1[:, :H])
    out["v3_vs_v1_delta_part"] = report("v3_vs_v1 [δ]",  f_v3[:, H:], f_v1[:, H:])

    # ---- speed ----
    print("=== speed ===", flush=True)
    def med_ms(fn, w=10, it=100):
        for _ in range(w): fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            s.append((time.perf_counter() - t0) * 1000)
        s.sort(); return s[len(s) // 2]

    t_v1 = med_ms(lambda: spline_kv_fwd_cuda(
        z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        activation="relu_sq", lambda_scale=lambda_scale))
    t_v3 = med_ms(lambda: spline_kv_fwd_v3_cuda(
        z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        activation="relu_sq", lambda_scale=lambda_scale))
    out["timing"] = {
        "v1_ms": t_v1, "v3_ms": t_v3,
        "speedup": t_v1 / t_v3 if t_v3 > 0 else 0.0,
    }
    print(json.dumps(out, indent=2), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main():
    print(run_probe.remote())
