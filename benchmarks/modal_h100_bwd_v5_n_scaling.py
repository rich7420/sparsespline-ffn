"""H100 v5 bwd N-scaling parity probe.

Validates the v5 bwd patch that drops `CHUNKS_PER_BLOCK` from the kernel
template and makes it a runtime arg, lifting the N ∈ {1024, 2048, 4096}
cap. New supported range: any N s.t. N % (NPARTS × BN) == 0.

For each N, compares v5 (new) against:
  1. autograd-fp32 reference — true mathematical gradient
  2. v1 bwd (`spline_kv_bwd_wgmma_cuda`) — known-good production kernel

Pass criteria:
  - v5_vs_ref max_abs_err is within the same order of magnitude as
    v1_vs_ref at every N (v5 must not regress precision vs v1).
  - No NaN / Inf in any v5 output.

Shapes mirror nanochat d20 production: H=2560, L=22, R=32, with N swept
over the values that come up in our pipeline:
  - 1024  : was supported (chunks_per_block=2)
  - 2048  : was supported (chunks_per_block=4)   ← original test point
  - 4096  : was supported (chunks_per_block=8)
  - 8192  : was REJECTED  (chunks_per_block=16)  ← Smoke A microbatch
  - 16384 : was REJECTED  (chunks_per_block=32)
  - 65536 : was REJECTED  (chunks_per_block=128) ← production d20 microbatch
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
                "dispatcher_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-bwd-v5-n-scaling-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_probe() -> dict:
    import sys, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.cuda_ext import (
        spline_kv_bwd_wgmma_cuda,
        spline_kv_bwd_wgmma_v5_cuda,
    )
    from sparsespline_ffn.rl_spline_kv_reference import flash_spline_feature_reference

    torch.manual_seed(0)
    device = torch.device("cuda")
    out: dict = {"runs": []}

    # Production shape: nanochat d20 has d=1280, h=2560 (h_ratio=2), L=22, R=32
    H, L, R = 2560, 22, 32
    G = L - 2
    grid_lo, grid_hi = -3.0, 3.0

    # Sweep N: covers smoke + production, including all previously-unsupported
    # chunks_per_block values (16, 32, 128).  NPARTS=4 internally, BN=128, so
    # chunks_per_block = N / 512.
    n_values = [1024, 2048, 4096, 8192, 16384, 65536]

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

    for N in n_values:
        chunks_per_block = (N // 4) // 128
        was_previously_supported = chunks_per_block in (2, 4, 8)
        print(f"\n{'='*72}", flush=True)
        print(f"  N = {N:>6d}  (chunks_per_block = {chunks_per_block:>3d}, "
              f"{'previously supported' if was_previously_supported else 'PREVIOUSLY REJECTED'})",
              flush=True)
        print(f"{'='*72}", flush=True)

        torch.manual_seed(42)
        z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
        C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1
        g = torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5

        # Reference: fp32 autograd through PyTorch reference impl
        z_t = z.detach().requires_grad_(True).float()
        C_t = C.detach().requires_grad_(True).float()
        with torch.enable_grad():
            f = flash_spline_feature_reference(
                z_t, C_t, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                activation="relu_sq", lambda_scale=1.0, spline_order=2,
            )
            grad_out = torch.zeros_like(f)
            grad_out[:, H:] = g.float()
            grads = torch.autograd.grad(f, [z_t, C_t], grad_outputs=grad_out)
            dz_ref = grads[0]
            dC_ref = grads[1]

        # v5 bwd (PATCHED — should now work for any N)
        try:
            dC_v5, dz_v5 = spline_kv_bwd_wgmma_v5_cuda(
                z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
            )
            v5_ok = True
            v5_err = ""
        except Exception as e:
            v5_ok = False
            v5_err = str(e)
            print(f"  v5 FAILED: {e}", flush=True)

        # v1 bwd (production reference)
        dC_v1, dz_v1 = spline_kv_bwd_wgmma_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        )

        run = {
            "N": N,
            "chunks_per_block": chunks_per_block,
            "was_previously_supported": was_previously_supported,
            "v5_launched": v5_ok,
            "v5_error": v5_err,
        }

        if v5_ok:
            run["dC_v5_vs_ref"] = stats("dC v5 vs ref", dC_v5, dC_ref)
            run["dz_v5_vs_ref"] = stats("dz v5 vs ref", dz_v5, dz_ref)
            run["dC_v1_vs_ref"] = stats("dC v1 vs ref", dC_v1, dC_ref)
            run["dz_v1_vs_ref"] = stats("dz v1 vs ref", dz_v1, dz_ref)
            run["dC_v5_vs_v1"]  = stats("dC v5 vs v1",  dC_v5, dC_v1)

            # Verdict: v5 should be no worse than v1 against the fp32 ref
            v5_dC = run["dC_v5_vs_ref"]["max_abs_err"]
            v1_dC = run["dC_v1_vs_ref"]["max_abs_err"]
            v5_dz = run["dz_v5_vs_ref"]["max_abs_err"]
            v1_dz = run["dz_v1_vs_ref"]["max_abs_err"]
            run["pass_dC_no_regression"] = v5_dC <= v1_dC * 1.5
            run["pass_dz_no_regression"] = v5_dz <= v1_dz * 1.5
            run["pass_no_nan_inf"] = (
                not run["dC_v5_vs_ref"]["has_nan"]
                and not run["dC_v5_vs_ref"]["has_inf"]
                and not run["dz_v5_vs_ref"]["has_nan"]
                and not run["dz_v5_vs_ref"]["has_inf"]
            )
            run["overall_pass"] = (
                run["pass_dC_no_regression"]
                and run["pass_dz_no_regression"]
                and run["pass_no_nan_inf"]
            )
            for k in ("dC_v1_vs_ref", "dC_v5_vs_ref", "dz_v1_vs_ref",
                      "dz_v5_vs_ref", "dC_v5_vs_v1"):
                v = run[k]
                print(f"  {k:22s}: signed={v['mean_signed_err']:+.3e} "
                      f"max_abs={v['max_abs_err']:.3e} "
                      f"max_rel={v['max_rel_err']:.3e}", flush=True)
            print(f"  → pass_dC_no_regression: {run['pass_dC_no_regression']}",
                  flush=True)
            print(f"  → pass_dz_no_regression: {run['pass_dz_no_regression']}",
                  flush=True)
            print(f"  → pass_no_nan_inf:       {run['pass_no_nan_inf']}",
                  flush=True)
            print(f"  → OVERALL:               {run['overall_pass']}",
                  flush=True)
        else:
            run["overall_pass"] = False

        out["runs"].append(run)

        # Free memory before next N (especially N=65536 is large)
        del z, C, g, dC_v1, dz_v1
        if v5_ok:
            del dC_v5, dz_v5
        del z_t, C_t, dC_ref, dz_ref
        torch.cuda.empty_cache()

    out["all_pass"] = all(r["overall_pass"] for r in out["runs"])
    out["unblocked_n_values"] = [
        r["N"] for r in out["runs"]
        if not r["was_previously_supported"] and r["overall_pass"]
    ]

    print("\n\n" + "=" * 72, flush=True)
    print(f"  ALL_PASS: {out['all_pass']}", flush=True)
    print(f"  Newly-unblocked N values: {out['unblocked_n_values']}", flush=True)
    print("=" * 72, flush=True)

    # =================================================================
    # Speed bench at production shape (N=65536) — confirms v5 retains
    # its speed advantage at the previously-unsupported microbatch.
    # =================================================================
    print("\n\n" + "=" * 72, flush=True)
    print("  Speed bench @ N=65536 (production d20 microbatch)", flush=True)
    print("=" * 72, flush=True)

    N_speed = 65536
    torch.manual_seed(42)
    z = torch.randn(N_speed, H, device=device, dtype=torch.bfloat16) * 1.5
    C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1
    g = torch.randn(N_speed, R, device=device, dtype=torch.bfloat16) * 0.5

    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward_v3 as triton_bwd,
    )

    # Warmup (let triton autotune settle, ensure CUDA caches are warm)
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
        return start.elapsed_time(end) / n_iter   # ms/call

    t_v5     = time_call(lambda: spline_kv_bwd_wgmma_v5_cuda(
        z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G))
    t_v1     = time_call(lambda: spline_kv_bwd_wgmma_cuda(
        z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G))
    t_triton = time_call(lambda: triton_bwd(z, C, g, grid_lo, grid_hi, G))

    speed = {
        "N": N_speed,
        "v5_ms_per_call":     t_v5,
        "v1_ms_per_call":     t_v1,
        "triton_ms_per_call": t_triton,
        "v5_speedup_vs_v1":     t_v1     / t_v5,
        "v5_speedup_vs_triton": t_triton / t_v5,
    }
    print(f"  v5     : {t_v5:>7.3f} ms/call", flush=True)
    print(f"  v1     : {t_v1:>7.3f} ms/call  (v5 is {speed['v5_speedup_vs_v1']:.2f}× v1)",
           flush=True)
    print(f"  triton : {t_triton:>7.3f} ms/call  "
           f"(v5 is {speed['v5_speedup_vs_triton']:.2f}× triton)", flush=True)
    out["speed_n65536"] = speed

    print("\n\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main():
    out = run_probe.remote()
    if not out["all_pass"]:
        raise SystemExit(1)
