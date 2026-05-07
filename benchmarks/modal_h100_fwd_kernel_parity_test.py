"""H100 numerical-parity test for the three forward-kernel implementations.

Why this exists
---------------
v10 fwd was advertised as bit-equivalent to v1 within bf16 noise (max_rel_err
0.06%) on a single random-input call.  But our 100M v10fwd run produced
val_loss=5.018 vs the v1/triton baseline 4.78 — a 0.24 nat regression that
should NOT come from a 0.06%-level numerical perturbation.

This test isolates the cause by:

1.  Running the same input through `triton`, `v1 CUDA`, and `v10 CUDA` fwd
    kernels and reporting *signed* mean error (bias) — not just absolute err.
2.  Sweeping multiple input distributions (uniform, heavy-tailed, near-grid-
    edge, fully-out-of-grid).
3.  Running the exact 500-step nanochat smoke training under each kernel
    with a fixed seed and comparing the loss curves directly.  If the
    kernels diverge here, the bug is real and reproducible.

The test does *not* attempt to diagnose the bug — it only confirms or
refutes the hypothesis "v10 (or v1) fwd produces a systematically different
result vs triton when accumulated over many gradient steps."
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "numpy", "ninja", "pyarrow", "tokenizers", "tiktoken",
        "regex", "huggingface-hub",
    )
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
DATA_VOLUME = modal.Volume.from_name("sparsefuse-phase3-data",
                                       create_if_missing=False)
app = modal.App("sparsespline-fwd-parity-h100", image=IMAGE)


# =============================================================================
# Microbench: pure-output comparison across 3 fwd kernels.
# =============================================================================
@app.function(gpu="H100", timeout=900)
def run_microbench() -> dict:
    import sys, json
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.cuda_ext import (
        spline_kv_fwd_cuda,
        spline_kv_fwd_v10_cuda,
        spline_kv_fwd_v11_cuda,
    )
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward as triton_fwd,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    out: dict = {"distributions": {}}

    # Production-realistic shapes (nanochat 124M h_ratio=2)
    N, H, L, R = 2048, 1536, 22, 32
    G = L - 2
    grid_lo, grid_hi = -3.0, 3.0
    lambda_scale = 1.0

    # Several input distributions to expose hidden bias.
    cases = {
        "uniform_std1.5": lambda: (
            torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5,
            torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1,
        ),
        "small_z_std0.5": lambda: (
            torch.randn(N, H, device=device, dtype=torch.bfloat16) * 0.5,
            torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1,
        ),
        "heavy_tailed": lambda: (
            (torch.randn(N, H, device=device, dtype=torch.float32) ** 3
             ).to(torch.bfloat16),
            torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1,
        ),
        "near_grid_edge": lambda: (
            (torch.randn(N, H, device=device, dtype=torch.bfloat16) * 0.3
             + 2.7),
            torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1,
        ),
        "out_of_range": lambda: (
            (torch.randn(N, H, device=device, dtype=torch.bfloat16) * 0.5
             + 5.0),
            torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1,
        ),
    }

    def stats(name: str, ours: torch.Tensor, ref: torch.Tensor) -> dict:
        ours_f = ours.float()
        ref_f = ref.float()
        diff = ours_f - ref_f
        diff_abs = diff.abs()
        ref_max = ref_f.abs().max().item()
        return {
            "label": name,
            "max_abs_err": float(diff_abs.max().item()),
            "max_rel_err": float(diff_abs.max().item() / (ref_max + 1e-9)),
            "mean_abs_err": float(diff_abs.mean().item()),
            "mean_signed_err": float(diff.mean().item()),  # bias
            "p99_abs_err": float(diff_abs.flatten().kthvalue(
                int(0.99 * diff_abs.numel())).values.item()),
            "ref_max": float(ref_max),
        }

    for case_name, sample_fn in cases.items():
        torch.manual_seed(42)
        z, C = sample_fn()

        f_triton = triton_fwd(
            z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
            activation="relu_sq", lambda_scale=lambda_scale, version="v4",
        )
        f_v1 = spline_kv_fwd_cuda(
            z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
            activation="relu_sq", lambda_scale=lambda_scale,
        )
        f_v10 = spline_kv_fwd_v10_cuda(
            z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
            activation="relu_sq", lambda_scale=lambda_scale,
        )
        f_v11 = spline_kv_fwd_v11_cuda(
            z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
            activation="relu_sq", lambda_scale=lambda_scale,
        )

        # Compare on the δ half (last R cols) — that's the kernel-specific
        # part. The activation half (first H cols) is computed independently
        # by each path so will differ slightly in implementation but should
        # always match at bf16 level.
        comp = {}
        comp["triton_vs_v1"]   = stats("triton-vs-v1",   f_triton[:, H:], f_v1[:, H:])
        comp["triton_vs_v10"]  = stats("triton-vs-v10",  f_triton[:, H:], f_v10[:, H:])
        comp["triton_vs_v11"]  = stats("triton-vs-v11",  f_triton[:, H:], f_v11[:, H:])
        comp["v1_vs_v10"]      = stats("v1-vs-v10",      f_v1[:, H:],     f_v10[:, H:])
        comp["v1_vs_v11"]      = stats("v1-vs-v11",      f_v1[:, H:],     f_v11[:, H:])
        # Also activation half — should be ~exact since all use relu_sq(z)
        comp["activation_triton_vs_v10"] = stats(
            "activation triton-vs-v10", f_triton[:, :H], f_v10[:, :H])
        comp["activation_triton_vs_v11"] = stats(
            "activation triton-vs-v11", f_triton[:, :H], f_v11[:, :H])

        # Speed: v10 and v11 should be the same (both fp16/bf16 wgmma at
        # 989 TFLOP/s).  Measure once on uniform_std1.5 only.
        if case_name == "uniform_std1.5":
            import time
            def med_ms(fn, w=10, it=50):
                for _ in range(w): fn()
                torch.cuda.synchronize()
                ts = []
                for _ in range(it):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    fn(); torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
                ts.sort(); return ts[len(ts) // 2]
            t_v1  = med_ms(lambda: spline_kv_fwd_cuda(
                z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                activation="relu_sq", lambda_scale=lambda_scale))
            t_v10 = med_ms(lambda: spline_kv_fwd_v10_cuda(
                z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                activation="relu_sq", lambda_scale=lambda_scale))
            t_v11 = med_ms(lambda: spline_kv_fwd_v11_cuda(
                z, C, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                activation="relu_sq", lambda_scale=lambda_scale))
            comp["timing_ms"] = {"v1": t_v1, "v10": t_v10, "v11": t_v11}
            print(f"  speed v1={t_v1:.4f}ms v10={t_v10:.4f}ms v11={t_v11:.4f}ms",
                  flush=True)
        out["distributions"][case_name] = comp
        print(f"=== {case_name} ===", flush=True)
        for k, v in comp.items():
            if k == "timing_ms":
                continue
            print(f"  {k}: signed={v['mean_signed_err']:+.3e} "
                   f"max_abs={v['max_abs_err']:.3e} "
                   f"max_rel={v['max_rel_err']:.3e} "
                   f"p99_abs={v['p99_abs_err']:.3e}", flush=True)

    print("\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


# =============================================================================
# Mini training loop: 500-step nanochat smoke under 3 fwd-kernel choices,
# all other config identical (same seed, same data, same bwd kernel).
# =============================================================================
@app.function(gpu="H100", timeout=1800,
              volumes={"/data": DATA_VOLUME})
def run_mini_training() -> dict:
    import sys, os, json, subprocess
    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo/nanochat")

    # Use SPARSE_SPLINE_FWD_KERNEL env var to force the kernel.
    # All 3 cells share the same arch (h=2, r=32, L=22, B2 spline) so the
    # ONLY difference is the fwd math path.
    common = dict(
        steps=500,
        mb=2,
        seq_len=1024,
        peak_lr=3e-4,
        warmup_steps=100,  # smaller warmup at 500 steps
        eval_every=100,
        eval_batches=10,
        diag_every=50,
        cell="rl_kv_B2_r32_L22_wgmmaCUDA_h2_all12",  # arch baseline
    )
    out: dict = {}
    runs = [
        ("triton",     "triton"),
        ("v1_cuda",    "wgmma_cuda"),
        ("v10_cuda",   "v10_cuda"),
    ]
    for label, kernel in runs:
        env = {
            **os.environ,
            "PYTHONPATH": "/repo/nanochat:/repo/src",
            "NANOCHAT_BASE_DIR": "/data/nanochat",
            "SPARSE_SPLINE_FWD_KERNEL": kernel,
        }
        out_json = f"/tmp/parity_{label}.json"
        cmd = [
            "python", "/repo/nanochat/nanochat_integration/nanochat_v41_redesign.py",
            "--mode", common["cell"],
            "--num-steps", str(common["steps"]),
            "--warmup-steps", str(common["warmup_steps"]),
            "--peak-lr", str(common["peak_lr"]),
            "--mb", str(common["mb"]),
            "--seq-len", str(common["seq_len"]),
            "--eval-every", str(common["eval_every"]),
            "--eval-batches", str(common["eval_batches"]),
            "--checkpoint-every", "999999",  # never
            "--diag-every", str(common["diag_every"]),
            "--dump-json", out_json,
            "--use-kernel",
            "--cuda-graph",
        ]
        print(f"\n========== running label={label} (env={kernel}) ==========",
              flush=True)
        proc = subprocess.run(cmd, cwd="/repo/nanochat", env=env,
                                capture_output=True, text=True, timeout=1200)
        if proc.returncode != 0:
            print(f"FAILED rc={proc.returncode}", flush=True)
            print("STDERR:", proc.stderr[-3000:], flush=True)
            out[label] = {"error": "rc != 0",
                            "stderr": proc.stderr[-3000:]}
            continue
        # Parse the JSON dump
        try:
            with open(out_json) as f:
                blob = json.load(f)
            # Extract loss curve from diagnostic rows
            losses = []
            for row in blob.get("diagnostics", []):
                if "step" in row and "train_loss" in row:
                    losses.append((row["step"], row["train_loss"]))
            out[label] = {
                "final_val_loss": blob.get("final_val_loss"),
                "loss_curve": losses,
                "wall_seconds": blob.get("wall_seconds"),
            }
        except Exception as e:
            out[label] = {"error": str(e),
                            "stdout": proc.stdout[-2000:]}

    print("\n=== loss curves comparison ===", flush=True)
    for label in [r[0] for r in runs]:
        d = out.get(label, {})
        if "error" in d:
            print(f"  {label}: ERROR {d['error']}", flush=True)
        else:
            curve = d.get("loss_curve", [])
            tail = curve[-3:] if curve else []
            print(f"  {label}: final={d.get('final_val_loss')} "
                   f"tail={tail}", flush=True)
    return out


@app.local_entrypoint()
def main(microbench: bool = True, training: bool = True):
    if microbench:
        print("\n>>>>> MICROBENCH (kernel output comparison) <<<<<\n")
        m = run_microbench.remote()
        # Already printed inside the function.
    if training:
        print("\n>>>>> MINI TRAINING (500-step under 3 kernels) <<<<<\n")
        t = run_mini_training.remote()
        print("\n=== mini training final ===")
        for label, data in t.items():
            if "error" in data:
                print(f"  {label}: ERROR")
            else:
                print(f"  {label}: final_val_loss={data.get('final_val_loss')}")
