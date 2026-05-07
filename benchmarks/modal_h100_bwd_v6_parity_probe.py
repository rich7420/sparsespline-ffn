"""H100 v6 bwd parity + speed probe.

Phase-aware probe — gates v6 each phase with the same battery:

  Phase v6.0  — v5-equivalent clone. Expectation: dC_v6 vs dC_v5 max_rel ≤ 1e-6
                (essentially bit-identical since the kernel body is the same).
                Speed within ±2% of v5.

  Phase v6.1a — TMA for C_smem only. Expectation: dC_v6 vs dC_v1 max_rel ≤ 5e-3,
                dC_v6 vs dC_v5 may differ slightly (TMA OOB fill semantics).
                Speed: small movement either way (C load is amortized).

  Phase v6.1b+ — see v6.cu top comment for phase plan.

Pass criteria (all phases):
  - dC_v6 vs dC_v1 max_rel ≤ 5e-3 (production parity gate)
  - dz_v6 vs dz_v1 max_rel ≤ 5e-3
  - no NaN / Inf in any v6 output
  - v6 runs at N=32768 AND N=65536 (production microbatches)

Speed comparison:
  - v6 vs v5 (the kernel we're trying to overtake)
  - v6 vs Triton (the launcher default)
  - v6 vs v1 (the baseline)
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
app = modal.App("sparsespline-bwd-v6-parity-h100", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
def run_probe() -> dict:
    import sys, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.cuda_ext import (
        spline_kv_bwd_wgmma_cuda,         # v1
        spline_kv_bwd_wgmma_v5_cuda,      # v5
        spline_kv_bwd_v6_cuda,            # v6 (this probe)
    )
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_backward_v3 as triton_bwd,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    out: dict = {"phase": "v6.0", "shapes": {}}

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

    def time_call(fn, n_iter=50, probe_first=True) -> float:
        if probe_first:
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record(); fn(); t1.record(); t1.synchronize()
            single_ms = t0.elapsed_time(t1)
            if single_ms > 500.0:
                n_iter = 5
                print(f"  [time_call] single call = {single_ms:.1f} ms — using {n_iter} iters", flush=True)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / n_iter

    for N in (32768, 65536):
        chunks_per_block = (N // 4) // 128
        print("\n" + "=" * 72, flush=True)
        print(f"  N = {N}  (chunks_per_block = {chunks_per_block})", flush=True)
        print("=" * 72, flush=True)

        torch.manual_seed(42)
        z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
        C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1
        g = torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5

        # ---- v6 sanity: it runs at this shape ----
        try:
            dC_v6, dz_v6 = spline_kv_bwd_v6_cuda(
                z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
            )
            v6_runs = True
            print(f"  v6 OK: dC={tuple(dC_v6.shape)}, dz={tuple(dz_v6.shape)}",
                  flush=True)
        except Exception as e:
            v6_runs = False
            print(f"  v6 FAILED: {e}", flush=True)
            out["shapes"][N] = {"v6_runs": False, "error": str(e)}
            continue

        # ---- v5 + v1 references ----
        dC_v5, dz_v5 = spline_kv_bwd_wgmma_v5_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        )
        dC_v1, dz_v1 = spline_kv_bwd_wgmma_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        )

        # ---- Parity: v6 vs v5 (should be ~bit-identical for v6.0) ----
        parity_v6_v5 = {
            "dC": stats("dC v6 vs v5", dC_v6, dC_v5),
            "dz": stats("dz v6 vs v5", dz_v6, dz_v5),
        }
        # ---- Parity: v6 vs v1 (production gate) ----
        parity_v6_v1 = {
            "dC": stats("dC v6 vs v1", dC_v6, dC_v1),
            "dz": stats("dz v6 vs v1", dz_v6, dz_v1),
        }

        for label, p in [("v6 vs v5", parity_v6_v5), ("v6 vs v1", parity_v6_v1)]:
            for k, v in p.items():
                print(f"  [{label:9s}] {k}: signed={v['mean_signed_err']:+.3e} "
                      f"max_abs={v['max_abs_err']:.3e} "
                      f"max_rel={v['max_rel_err']:.3e}", flush=True)

        pass_no_nan_inf = all(
            not p[k][f] for p in [parity_v6_v5, parity_v6_v1]
            for k in ("dC", "dz") for f in ("has_nan", "has_inf")
        )
        # v6.0 phase: very tight gate vs v5 (clone should be ≈identical)
        pass_v6_v5_tight = (
            parity_v6_v5["dC"]["max_rel_err"] <= 1e-5
            and parity_v6_v5["dz"]["max_rel_err"] <= 1e-5
        )
        # production gate: same as v5 has been passing
        pass_v6_v1_prod = (
            parity_v6_v1["dC"]["max_rel_err"] <= 5e-3
            and parity_v6_v1["dz"]["max_rel_err"] <= 5e-3
        )

        # ---- Speed bench: v6 vs v5 vs v1 vs Triton ----
        for _ in range(5):
            spline_kv_bwd_v6_cuda(z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G)
            spline_kv_bwd_wgmma_v5_cuda(z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G)
            spline_kv_bwd_wgmma_cuda(z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G)
            triton_bwd(z, C, g, grid_lo, grid_hi, G)
        torch.cuda.synchronize()

        t_v6     = time_call(lambda: spline_kv_bwd_v6_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G))
        t_v5     = time_call(lambda: spline_kv_bwd_wgmma_v5_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G))
        t_v1     = time_call(lambda: spline_kv_bwd_wgmma_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G))
        t_triton = time_call(lambda: triton_bwd(z, C, g, grid_lo, grid_hi, G))

        speed = {
            "N": N,
            "v6_ms_per_call":     t_v6,
            "v5_ms_per_call":     t_v5,
            "v1_ms_per_call":     t_v1,
            "triton_ms_per_call": t_triton,
            "v6_speedup_vs_v5":     t_v5     / t_v6,
            "v6_speedup_vs_v1":     t_v1     / t_v6,
            "v6_speedup_vs_triton": t_triton / t_v6,
        }
        print(f"  v6     : {t_v6:>7.3f} ms/call", flush=True)
        print(f"  v5     : {t_v5:>7.3f} ms/call  "
               f"(v6 is {speed['v6_speedup_vs_v5']:.3f}× v5)", flush=True)
        print(f"  v1     : {t_v1:>7.3f} ms/call  "
               f"(v6 is {speed['v6_speedup_vs_v1']:.3f}× v1)", flush=True)
        print(f"  triton : {t_triton:>7.3f} ms/call  "
               f"(v6 is {speed['v6_speedup_vs_triton']:.3f}× triton)", flush=True)

        out["shapes"][N] = {
            "v6_runs": True,
            "parity_v6_v5": parity_v6_v5,
            "parity_v6_v1": parity_v6_v1,
            "speed": speed,
            "pass_no_nan_inf":   pass_no_nan_inf,
            "pass_v6_v5_tight":  pass_v6_v5_tight,   # v6.0 gate
            "pass_v6_v1_prod":   pass_v6_v1_prod,    # production gate
        }

        del z, C, g, dC_v1, dz_v1, dC_v5, dz_v5, dC_v6, dz_v6
        torch.cuda.empty_cache()

    # ---- Verdict ----
    print("\n" + "=" * 72, flush=True)
    print("  PHASE v6.0 VERDICT", flush=True)
    print("=" * 72, flush=True)
    all_v6_v5_tight = all(s.get("pass_v6_v5_tight", False) for s in out["shapes"].values())
    all_v6_v1_prod  = all(s.get("pass_v6_v1_prod",  False) for s in out["shapes"].values())
    all_no_nan_inf  = all(s.get("pass_no_nan_inf",  False) for s in out["shapes"].values())
    speeds_vs_v5    = [s["speed"]["v6_speedup_vs_v5"] for s in out["shapes"].values()
                       if s.get("speed")]
    speed_within_2pct = (speeds_vs_v5 and
                        all(0.98 <= s <= 1.02 for s in speeds_vs_v5))

    out["phase_v6_0_pass_v6_v5_tight"]  = all_v6_v5_tight
    out["phase_v6_0_pass_v6_v1_prod"]   = all_v6_v1_prod
    out["phase_v6_0_pass_no_nan_inf"]   = all_no_nan_inf
    out["phase_v6_0_pass_speed_within_2pct"] = bool(speed_within_2pct)

    print(f"  v6 == v5 (max_rel ≤ 1e-5):    {all_v6_v5_tight}", flush=True)
    print(f"  v6 vs v1 production gate:     {all_v6_v1_prod}", flush=True)
    print(f"  no NaN/Inf:                   {all_no_nan_inf}", flush=True)
    print(f"  speed within ±2% of v5:       {speed_within_2pct}", flush=True)

    overall = all_v6_v5_tight and all_v6_v1_prod and all_no_nan_inf and speed_within_2pct
    out["phase_v6_0_overall"] = overall
    print(f"\n  PHASE v6.0 OVERALL: {'✅ PASS' if overall else '❌ FAIL'}", flush=True)
    print(f"  → {'proceed to v6.1a' if overall else 'investigate before continuing'}",
          flush=True)

    print("\n\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main():
    out = run_probe.remote()
    if not out.get("phase_v6_0_overall", False):
        raise SystemExit(1)
