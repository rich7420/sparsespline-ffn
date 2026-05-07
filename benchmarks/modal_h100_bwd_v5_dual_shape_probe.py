"""H100 v5 bwd dual-shape production probe (N=32768 + N=65536).

Two production microbatch shapes for nanochat d20:
  - N = device_batch × seq_len = 16 × 2048 = 32768   (current, with mem fix)
  - N =                          32 × 2048 = 65536   (target, if mem allows)

For each N: parity (v5 vs v1) + speed (v5 vs v1 vs Triton).

Pass criteria (per shape):
  - v5 runs without NaN/Inf
  - dC_v5_vs_v1 max_rel ≤ 5e-3
  - dz_v5_vs_v1 max_rel ≤ 5e-3

Decision (after both shapes report):
  if v5_speedup_vs_triton ≥ 1.15 AT BOTH N: switch launcher default to v5
  if v5_speedup ≥ 1.0 (par or better): keep Triton default; v5 documented
  if v5_speedup < 1.0: keep Triton default; investigate further

Reports per-call peak memory + scratch tensor size for memory-side comparison.
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
app = modal.App("sparsespline-bwd-v5-dual-shape-h100", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
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
    out: dict = {"shapes": {}}

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
        # Probe first: time 1 call. If it's > 500ms, the kernel is slow enough
        # that doing 50 iters will likely timeout. Fall back to 5 iters.
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
        print(f"  N = {N}  (device_batch×seq_len = "
              f"{N // 2048} × 2048; chunks_per_block = {chunks_per_block})",
              flush=True)
        print("=" * 72, flush=True)

        torch.manual_seed(42)
        z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
        C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1
        g = torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5

        # ---- Parity ----
        try:
            dC_v5, dz_v5 = spline_kv_bwd_wgmma_v5_cuda(
                z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
            )
            v5_ok = True
            print(f"  v5 OK: dC={tuple(dC_v5.shape)}, dz={tuple(dz_v5.shape)}",
                  flush=True)
        except Exception as e:
            v5_ok = False
            print(f"  v5 FAILED: {e}", flush=True)
            out["shapes"][N] = {"v5_runs": False, "error": str(e)}
            continue

        dC_v1, dz_v1 = spline_kv_bwd_wgmma_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        )

        parity = {
            "dC_v5_vs_v1": stats("dC v5 vs v1", dC_v5, dC_v1),
            "dz_v5_vs_v1": stats("dz v5 vs v1", dz_v5, dz_v1),
        }
        for k, v in parity.items():
            print(f"  {k:20s}: signed={v['mean_signed_err']:+.3e} "
                  f"max_abs={v['max_abs_err']:.3e} "
                  f"max_rel={v['max_rel_err']:.3e}", flush=True)

        pass_no_nan_inf = (
            not parity["dC_v5_vs_v1"]["has_nan"]
            and not parity["dC_v5_vs_v1"]["has_inf"]
            and not parity["dz_v5_vs_v1"]["has_nan"]
            and not parity["dz_v5_vs_v1"]["has_inf"]
        )
        pass_dC_rel = parity["dC_v5_vs_v1"]["max_rel_err"] <= 5e-3
        pass_dz_rel = parity["dz_v5_vs_v1"]["max_rel_err"] <= 5e-3
        parity_pass = pass_no_nan_inf and pass_dC_rel and pass_dz_rel
        print(f"  → PARITY: {parity_pass}", flush=True)

        # ---- Speed bench ----
        # Warmup (let triton autotune settle)
        for _ in range(5):
            spline_kv_bwd_wgmma_v5_cuda(z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G)
            spline_kv_bwd_wgmma_cuda(z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G)
            triton_bwd(z, C, g, grid_lo, grid_hi, G)
        torch.cuda.synchronize()

        t_v5     = time_call(lambda: spline_kv_bwd_wgmma_v5_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G))
        t_v1     = time_call(lambda: spline_kv_bwd_wgmma_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G))
        t_triton = time_call(lambda: triton_bwd(z, C, g, grid_lo, grid_hi, G))

        speed = {
            "N": N,
            "chunks_per_block": chunks_per_block,
            "v5_ms_per_call":     t_v5,
            "v1_ms_per_call":     t_v1,
            "triton_ms_per_call": t_triton,
            "v5_speedup_vs_v1":     t_v1     / t_v5,
            "v5_speedup_vs_triton": t_triton / t_v5,
        }
        print(f"  v5     : {t_v5:>7.3f} ms/call", flush=True)
        print(f"  v1     : {t_v1:>7.3f} ms/call  "
               f"(v5 is {speed['v5_speedup_vs_v1']:.2f}× v1)", flush=True)
        print(f"  triton : {t_triton:>7.3f} ms/call  "
               f"(v5 is {speed['v5_speedup_vs_triton']:.2f}× triton)", flush=True)

        out["shapes"][N] = {
            "v5_runs": True,
            "parity": parity,
            "parity_pass": parity_pass,
            "speed": speed,
        }

        del z, C, g, dC_v1, dz_v1, dC_v5, dz_v5
        torch.cuda.empty_cache()

    # Decision
    print("\n" + "=" * 72, flush=True)
    print("  DECISION", flush=True)
    print("=" * 72, flush=True)
    all_parity = all(s.get("parity_pass", False) for s in out["shapes"].values())
    speedups = [s["speed"]["v5_speedup_vs_triton"]
                for s in out["shapes"].values() if s.get("speed")]
    out["all_parity_pass"] = all_parity
    out["min_speedup_vs_triton"] = min(speedups) if speedups else 0.0
    out["max_speedup_vs_triton"] = max(speedups) if speedups else 0.0

    print(f"  all_parity_pass:      {all_parity}", flush=True)
    print(f"  min v5/triton speedup: {out['min_speedup_vs_triton']:.3f}",
          flush=True)
    print(f"  max v5/triton speedup: {out['max_speedup_vs_triton']:.3f}",
          flush=True)
    if all_parity and out["min_speedup_vs_triton"] >= 1.15:
        verdict = "SWITCH_TO_V5_DEFAULT"
    elif all_parity and out["min_speedup_vs_triton"] >= 1.0:
        verdict = "PARITY_KEEP_TRITON"
    elif all_parity:
        verdict = "V5_SLOWER_KEEP_TRITON"
    else:
        verdict = "PARITY_FAILED_DO_NOT_USE"
    out["verdict"] = verdict
    print(f"  verdict:               {verdict}", flush=True)

    print("\n\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main():
    out = run_probe.remote()
    if not out.get("all_parity_pass", False):
        raise SystemExit(1)
