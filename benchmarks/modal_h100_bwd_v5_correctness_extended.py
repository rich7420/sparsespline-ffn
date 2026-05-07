"""H100 v5 bwd extended numerical correctness probe.

Why this exists
---------------
The basic v5 probe compared v5 against v1 — but v1 itself has a known
bf16(B) precision floor.  This probe instead compares v5 (and v1) against
the *PyTorch autograd reference* computed in fp32, which gives the true
mathematical gradient.  We want:
  - v5_vs_ref_dC ≤ v1_vs_ref_dC                         (v5 not worse than v1)
  - v5_vs_ref_dz close to ULP-fp32                       (dz is actually fp32)
  - mean_signed_err small + non-systematic across distributions

Distributions tested:
  - uniform_std1.5 (production-realistic)
  - small_z_std0.5
  - heavy_tailed
  - near_grid_edge
  - out_of_range
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
app = modal.App("sparsespline-bwd-v5-correctness-h100", image=IMAGE)


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
    out: dict = {"distributions": {}}

    # Production shape
    N, H, L, R = 2048, 1536, 22, 32
    G = L - 2
    grid_lo, grid_hi = -3.0, 3.0

    cases = {
        "uniform_std1.5": lambda: (
            torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5,
            torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1,
            torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5,
        ),
        "small_z_std0.5": lambda: (
            torch.randn(N, H, device=device, dtype=torch.bfloat16) * 0.5,
            torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1,
            torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5,
        ),
        "heavy_tailed": lambda: (
            (torch.randn(N, H, device=device, dtype=torch.float32) ** 3
             ).to(torch.bfloat16),
            torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1,
            torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5,
        ),
        "near_grid_edge": lambda: (
            (torch.randn(N, H, device=device, dtype=torch.bfloat16) * 0.3
             + 2.7),
            torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1,
            torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5,
        ),
        "out_of_range": lambda: (
            (torch.randn(N, H, device=device, dtype=torch.bfloat16) * 0.5
             + 5.0),
            torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1,
            torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5,
        ),
    }

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
        }

    for case_name, sample_fn in cases.items():
        torch.manual_seed(42)
        z, C, g = sample_fn()

        # Fake `g_a` and the lambda chain — we just want gradients of δ wrt z, C
        # given upstream gradient g.  Use the autograd reference to compute the
        # ground-truth dz_spline and dC.
        z_t = z.detach().requires_grad_(True).float()  # fp32 for true ref
        C_t = C.detach().requires_grad_(True).float()
        from sparsespline_ffn.rl_spline_kv_reference import flash_spline_feature_reference
        with torch.enable_grad():
            f = flash_spline_feature_reference(
                z_t, C_t, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                activation="relu_sq", lambda_scale=1.0, spline_order=2,
            )
            # Take grad wrt only the δ half; supply g as the upstream grad for δ.
            # f = [a; δ] → grad_output = [zeros, g].
            grad_out = torch.zeros_like(f)
            grad_out[:, H:] = g.float()
            grads = torch.autograd.grad(f, [z_t, C_t], grad_outputs=grad_out)
            dz_ref = grads[0]
            dC_ref = grads[1]

        # v1 bwd (production)
        dC_v1, dz_v1 = spline_kv_bwd_wgmma_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        )
        # v5 bwd (new)
        dC_v5, dz_v5 = spline_kv_bwd_wgmma_v5_cuda(
            z, C, g, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
        )

        # NOTE: ref's dz includes both ReLU² and δ contributions.  The CUDA
        # bwd kernels return ONLY the δ half (dz_spline).  So to compare,
        # we compute the corresponding ref-δ-only gradient.
        # f = [a(z); δ(z, C)].  When grad_out has zeros on the a half and g on
        # the δ half, the dz from autograd is purely δ's contribution, since
        # the a half doesn't see any upstream gradient.  ✓

        comp = {}
        comp["dC_v1_vs_ref"] = stats("dC v1 vs ref", dC_v1, dC_ref)
        comp["dC_v5_vs_ref"] = stats("dC v5 vs ref", dC_v5, dC_ref)
        comp["dz_v1_vs_ref"] = stats("dz v1 vs ref", dz_v1, dz_ref)
        comp["dz_v5_vs_ref"] = stats("dz v5 vs ref", dz_v5, dz_ref)
        comp["dC_v5_vs_v1"]  = stats("dC v5 vs v1",  dC_v5, dC_v1)
        comp["dz_v5_vs_v1"]  = stats("dz v5 vs v1",  dz_v5, dz_v1)

        out["distributions"][case_name] = comp
        print(f"\n=== {case_name} ===", flush=True)
        for k, v in comp.items():
            print(f"  {k:22s}: signed={v['mean_signed_err']:+.3e} "
                   f"max_abs={v['max_abs_err']:.3e} "
                   f"max_rel={v['max_rel_err']:.3e} "
                   f"mean_abs={v['mean_abs_err']:.3e}", flush=True)

    print("\n\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main():
    print(run_probe.remote())
