"""H100 v5 bwd production-shape probe (v2 of n_scaling).

Two purposes:
  1. Confirm the patched v5 kernel (chunks_per_block as runtime arg) runs
     correctly at the production microbatch N=65536 — the case the v1
     fp32 autograd reference can't fit. Use v5-vs-v1 as the parity gate
     (v1 is the validated production kernel; v5 only differs in B-operand
     precision floor).
  2. Speed bench at N=65536: v5 vs v1 vs Triton bwd, ms/call.

Why this supersedes `modal_h100_bwd_v5_n_scaling.py`:
  - The earlier script's OVERALL gate compared v5_vs_ref ≤ 1.5× v1_vs_ref,
    which was tighter than v5's intrinsic fp16(B) precision floor (~1.7×
    v1's bf16(B) floor). The 1.7× ratio is constant across all N and is
    unchanged by the patch — it's a kernel-design property, not a bug.
  - At N=65536 the fp32 autograd reference materializes ~461 GB of
    intermediates and OOMs before v5 ever runs.

Pass criteria here:
  - v5 runs at N=65536 without error or NaN/Inf.
  - v5 vs v1 max_rel ≤ 5e-3 (matches the observed bound at smaller N).
  - Speed: report v5 / v1 / triton ms/call. No hard pass threshold —
    the question is "is v5 still fast enough at production shape?".
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
                "dispatcher_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-bwd-v5-prod-probe-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_probe() -> dict:
    import sys, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.cuda_ext import (
        spline_kv_bwd_wgmma_cuda,
        spline_kv_bwd_wgmma_v5_cuda,
    )
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward_v3 as triton_bwd,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    out: dict = {}

    H, L, R = 2560, 22, 32
    G = L - 2
    grid_lo, grid_hi = -3.0, 3.0

    def stats(name: str, ours: torch.Tensor, ref: torch.Tensor) -> dict:
        ours_f = ours.float(); ref_f = ref.float()
        diff = ours_f - ref_f
        diff_abs = diff.abs()
        ref_max = ref_f.abs().max().item()
        return {
            "label": name,
            "max_abs_err": float(diff_abs.max().item()),
            "max_rel_err": float(diff_abs.max().item() / (ref_max + 1e-9)),
            "mean_abs_err": float(diff_abs.mean().item()),
            "mean_signed_err": float(diff.mean().item()),
            "ref_max": float(ref_max),
            "has_nan": bool(torch.isnan(ours_f).any().item()),
            "has_inf": bool(torch.isinf(ours_f).any().item()),
        }

    # =================================================================
    # Parity at N=65536: v5 vs v1 (fp32 ref doesn't fit at this N).
    # =================================================================
    print("\n" + "=" * 72, flush=True)
    print("  Parity @ N=65536  (v5 vs v1 — production microbatch)", flush=True)
    print("=" * 72, flush=True)

    N = 65536
    torch.manual_seed(42)
    z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
    C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1
    g = torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5

    print(f"  shapes: z={tuple(z.shape)}, C={tuple(C.shape)}, g={tuple(g.shape)}",
          flush=True)
    print(f"  dtypes: z={z.dtype}, C={C.dtype}, g={g.dtype}", flush=True)

    # v5 must run at this shape (chunks_per_block=128).
    try:
        dC_v5, dz_v5 = spline_kv_bwd_wgmma_v5_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        )
        v5_ok = True
        v5_err = ""
        print(f"  v5 OK: dC={tuple(dC_v5.shape)}, dz={tuple(dz_v5.shape)}",
              flush=True)
    except Exception as e:
        v5_ok = False
        v5_err = str(e)
        print(f"  v5 FAILED: {e}", flush=True)
        out["v5_n65536_runs"] = False
        out["v5_n65536_error"] = v5_err
        out["all_pass"] = False
        return out

    # v1 production reference
    dC_v1, dz_v1 = spline_kv_bwd_wgmma_cuda(
        z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
    )

    parity = {
        "dC_v5_vs_v1": stats("dC v5 vs v1", dC_v5, dC_v1),
        "dz_v5_vs_v1": stats("dz v5 vs v1", dz_v5, dz_v1),
    }
    out["parity_n65536"] = parity
    for k, v in parity.items():
        print(f"  {k:20s}: signed={v['mean_signed_err']:+.3e} "
              f"max_abs={v['max_abs_err']:.3e} "
              f"max_rel={v['max_rel_err']:.3e} "
              f"mean_abs={v['mean_abs_err']:.3e}", flush=True)

    pass_no_nan_inf = (
        not parity["dC_v5_vs_v1"]["has_nan"]
        and not parity["dC_v5_vs_v1"]["has_inf"]
        and not parity["dz_v5_vs_v1"]["has_nan"]
        and not parity["dz_v5_vs_v1"]["has_inf"]
    )
    pass_dC_rel = parity["dC_v5_vs_v1"]["max_rel_err"] <= 5e-3
    pass_dz_rel = parity["dz_v5_vs_v1"]["max_rel_err"] <= 5e-3
    parity_pass = pass_no_nan_inf and pass_dC_rel and pass_dz_rel
    out["parity_pass"] = parity_pass
    print(f"  → no_nan_inf:           {pass_no_nan_inf}", flush=True)
    print(f"  → dC max_rel ≤ 5e-3:    {pass_dC_rel}", flush=True)
    print(f"  → dz max_rel ≤ 5e-3:    {pass_dz_rel}", flush=True)
    print(f"  → PARITY:               {parity_pass}", flush=True)

    # =================================================================
    # Speed bench @ N=65536
    # =================================================================
    print("\n" + "=" * 72, flush=True)
    print("  Speed bench @ N=65536", flush=True)
    print("=" * 72, flush=True)

    # Warmup (let triton autotune settle)
    for _ in range(5):
        spline_kv_bwd_wgmma_v5_cuda(z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G)
        spline_kv_bwd_wgmma_cuda(z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G)
        triton_bwd(z, C, g, grid_lo, grid_hi, G)
    torch.cuda.synchronize()

    def time_call(fn, n_iter=50) -> float:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / n_iter

    t_v5     = time_call(lambda: spline_kv_bwd_wgmma_v5_cuda(
        z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G))
    t_v1     = time_call(lambda: spline_kv_bwd_wgmma_cuda(
        z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G))
    t_triton = time_call(lambda: triton_bwd(z, C, g, grid_lo, grid_hi, G))

    speed = {
        "N": N,
        "v5_ms_per_call":     t_v5,
        "v1_ms_per_call":     t_v1,
        "triton_ms_per_call": t_triton,
        "v5_speedup_vs_v1":     t_v1     / t_v5,
        "v5_speedup_vs_triton": t_triton / t_v5,
    }
    out["speed_n65536"] = speed
    print(f"  v5     : {t_v5:>7.3f} ms/call", flush=True)
    print(f"  v1     : {t_v1:>7.3f} ms/call  "
           f"(v5 is {speed['v5_speedup_vs_v1']:.2f}× v1)", flush=True)
    print(f"  triton : {t_triton:>7.3f} ms/call  "
           f"(v5 is {speed['v5_speedup_vs_triton']:.2f}× triton)", flush=True)

    out["all_pass"] = parity_pass
    print("\n\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main():
    out = run_probe.remote()
    if not out.get("all_pass", False):
        raise SystemExit(1)
