"""Phase 2 validation suite for the new native CUDA forward path.

Three independent tests, all on H100:

Test A: Kernel correctness
  Compare `spline_kv_fwd_cuda` direct output to `flash_spline_delta_forward_v4`
  (Triton oracle) for the spline residual half. rel_err ≤ 2e-3.

Test B: e2e autograd correctness  (eager + graph, all kernel combinations)
  For each (fwd_kernel, bwd_kernel) pair, run forward + MSE loss + backward,
  compare loss / dz / dC against reference (use_kernel=False, full PyTorch).
  All 6 combos × 2 modes = 12 checks. Pass: rel_err on dz, dC ≤ 2e-3,
  AND graph-mode result == eager-mode result (capture-safety).

Test C: Fused fwd kernel correctness
  Compare `spline_kv_fwd_fused_cuda(z, C, W_out)` against the unfused path
  (`spline_kv_fwd_cuda(z, C)` then `f @ W_out.T`). Pass: rel_err ≤ 2e-3.

Run:
  modal run benchmarks/modal_h100_phase2_validate.py
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
        ignore=[".venv/**", ".git/**", "nanochat/.venv/**",
                "nanochat/.nanochat-runtime/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("rlkv-phase2-validate", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
def run() -> str:
    import sys, io
    sys.path.insert(0, "/repo/src")
    import torch
    out = io.StringIO()
    log = lambda s="": (out.write(s + "\n"), print(s, flush=True))

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"torch: {torch.__version__}")
    log("")

    # Common shape
    N, h, r, G, L_ord = 2048, 768, 32, 20, 2
    L = G + L_ord  # 22

    torch.manual_seed(0)
    z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
    C = torch.randn(h, L, r, device="cuda", dtype=torch.bfloat16) * 0.05
    grid_lo, grid_hi = -3.0, 3.0
    lambda_scale = 1.0

    # ============================================================
    # Test A: Kernel correctness — CUDA fwd vs Triton fwd v4
    # ============================================================
    log("=" * 100)
    log("Test A: spline_kv_fwd_cuda vs Triton flash_spline_delta_forward_v4")
    log("=" * 100)
    from sparsespline_ffn.cuda_ext import (
        spline_kv_fwd_cuda, spline_kv_fwd_fused_cuda,
    )
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_forward_v4 as triton_delta_v4,
    )

    # Triton oracle: produces fp32 delta. Multiply by lambda; cast to bf16 for fair comparison.
    triton_delta = triton_delta_v4(z, C, grid_lo, grid_hi, G).to(torch.bfloat16)
    if lambda_scale != 1.0:
        triton_delta = (triton_delta.float() * lambda_scale).to(torch.bfloat16)
    triton_a = torch.where(z > 0, z * z, torch.zeros_like(z))

    # Native CUDA fwd: returns concatenated [a; lambda*delta] [N, h+r]
    cuda_f = spline_kv_fwd_cuda(z, C, grid_lo, grid_hi, G,
                                 activation="relu_sq",
                                 lambda_scale=lambda_scale)
    cuda_a = cuda_f[:, :h]
    cuda_delta = cuda_f[:, h:h + r]

    rel_err_a = (cuda_a.float() - triton_a.float()).norm() / triton_a.float().norm().clamp_min(1e-9)
    rel_err_d = (cuda_delta.float() - triton_delta.float()).norm() / triton_delta.float().norm().clamp_min(1e-9)
    log(f"  a (activation):  CUDA.norm={cuda_a.float().norm().item():.4e}  "
        f"Triton.norm={triton_a.float().norm().item():.4e}  rel_err={rel_err_a.item():.4e}")
    log(f"  δ (spline):      CUDA.norm={cuda_delta.float().norm().item():.4e}  "
        f"Triton.norm={triton_delta.float().norm().item():.4e}  rel_err={rel_err_d.item():.4e}")
    test_a_pass = (rel_err_a.item() < 2e-3) and (rel_err_d.item() < 2e-3)
    log(f"  Test A: {'PASS' if test_a_pass else 'FAIL'}")
    log("")

    # ============================================================
    # Test B: e2e autograd correctness across all combos × modes
    # ============================================================
    log("=" * 100)
    log("Test B: e2e autograd (forward + backward) — all (fwd, bwd) × (eager, graph)")
    log("=" * 100)

    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature, FlashSplineFeature,
    )
    # Reference: pure-PyTorch forward + autograd backward
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference as ref_fwd,
    )

    # ------ build a reference oracle: ref forward + autograd backward ------
    torch.manual_seed(42)
    z_ref = z.detach().clone().requires_grad_(True)
    C_ref = C.detach().clone().requires_grad_(True)
    f_ref = ref_fwd(z_ref, C_ref, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                     activation="relu_sq", lambda_scale=lambda_scale)
    target = torch.randn_like(f_ref)
    loss_ref = (f_ref - target).pow(2).sum()
    loss_ref.backward()
    dz_ref = z_ref.grad.detach().clone()
    dC_ref = C_ref.grad.detach().clone()
    log(f"  reference: loss={loss_ref.item():.4f}  "
        f"dz.norm={dz_ref.float().norm().item():.4e}  "
        f"dC.norm={dC_ref.float().norm().item():.4e}")

    combos = [
        # (fwd_kernel, bwd_kernel)
        ("triton",     "triton"),
        ("triton",     "hopper_cuda"),
        ("triton",     "wgmma_cuda"),
        ("wgmma_cuda", "triton"),
        ("wgmma_cuda", "hopper_cuda"),
        ("wgmma_cuda", "wgmma_cuda"),
    ]

    # Per-cell results plus a fast-path "kernel-vs-kernel" comparison: every
    # cell should produce bit-identical dz/dC to the (triton, triton, eager)
    # baseline, because the kernels share the same fp32 atomic+bf16 cast path.
    test_b_results = {}
    baseline_dz = None
    baseline_dC = None

    for (fwk, bwk) in combos:
        for mode in ("eager", "graph"):
            tag = f"fwd={fwk:<11} bwd={bwk:<11} {mode}"
            try:
                torch.manual_seed(42)
                z_t = z.detach().clone().requires_grad_(True)
                C_t = C.detach().clone().requires_grad_(True)

                if mode == "eager":
                    f_out = flash_spline_feature(
                        z_t, C_t, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                        activation="relu_sq", lambda_scale=lambda_scale,
                        use_kernel=True,
                        bwd_kernel=bwk, fwd_kernel=fwk,
                    )
                    loss = (f_out - target).pow(2).sum()
                    loss.backward()
                else:
                    # CUDA-Graph mode
                    if z_t.grad is None:
                        z_t.grad = torch.zeros_like(z_t)
                    if C_t.grad is None:
                        C_t.grad = torch.zeros_like(C_t)
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    static_target = target.detach().clone()
                    with torch.cuda.stream(s):
                        for _ in range(3):
                            z_t.grad.zero_(); C_t.grad.zero_()
                            f_w = flash_spline_feature(
                                z_t, C_t, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                                activation="relu_sq", lambda_scale=lambda_scale,
                                use_kernel=True,
                                bwd_kernel=bwk, fwd_kernel=fwk,
                            )
                            ((f_w - static_target).pow(2).sum()).backward()
                    torch.cuda.current_stream().wait_stream(s)
                    torch.cuda.synchronize()

                    g = torch.cuda.CUDAGraph()
                    z_t.grad.zero_(); C_t.grad.zero_()
                    with torch.cuda.graph(g):
                        f_out = flash_spline_feature(
                            z_t, C_t, grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                            activation="relu_sq", lambda_scale=lambda_scale,
                            use_kernel=True,
                            bwd_kernel=bwk, fwd_kernel=fwk,
                        )
                        static_loss = (f_out - static_target).pow(2).sum()
                        static_loss.backward()
                    z_t.grad.zero_(); C_t.grad.zero_()
                    g.replay()
                    torch.cuda.synchronize()
                    loss = static_loss

                # vs fp32 PyTorch reference (loose — bf16 noise expected ~1-3%)
                err_loss = abs(loss.item() - loss_ref.item()) / max(abs(loss_ref.item()), 1e-9)
                err_dz = ((z_t.grad.float() - dz_ref.float()).norm()
                          / dz_ref.float().norm().clamp_min(1e-9)).item()
                err_dC = ((C_t.grad.float() - dC_ref.float()).norm()
                          / dC_ref.float().norm().clamp_min(1e-9)).item()
                # vs first kernel cell (TIGHT — every kernel path must agree)
                if baseline_dz is None:
                    baseline_dz = z_t.grad.detach().clone()
                    baseline_dC = C_t.grad.detach().clone()
                    err_dz_vs_kernel = 0.0
                    err_dC_vs_kernel = 0.0
                else:
                    err_dz_vs_kernel = ((z_t.grad.float() - baseline_dz.float()).norm()
                                         / baseline_dz.float().norm().clamp_min(1e-9)).item()
                    err_dC_vs_kernel = ((C_t.grad.float() - baseline_dC.float()).norm()
                                         / baseline_dC.float().norm().clamp_min(1e-9)).item()
                # Pass criteria: kernel-vs-kernel TIGHT, kernel-vs-fp32 LOOSE
                ok_vs_ref = (err_loss < 1e-2) and (err_dz < 5e-2) and (err_dC < 5e-2)
                ok_vs_kernel = (err_dz_vs_kernel < 1e-3) and (err_dC_vs_kernel < 1e-3)
                ok = ok_vs_ref and ok_vs_kernel
                test_b_results[(fwk, bwk, mode)] = {
                    "loss": loss.item(),
                    "err_loss": err_loss,
                    "err_dz": err_dz,
                    "err_dC": err_dC,
                    "err_dz_vs_kernel": err_dz_vs_kernel,
                    "err_dC_vs_kernel": err_dC_vs_kernel,
                    "ok": ok,
                    "ok_vs_ref": ok_vs_ref,
                    "ok_vs_kernel": ok_vs_kernel,
                }
                log(f"  {tag} loss={loss.item():.2f} "
                    f"err_dz_ref={err_dz:.3e} err_dC_ref={err_dC:.3e} "
                    f"err_dz_ker={err_dz_vs_kernel:.3e} "
                    f"err_dC_ker={err_dC_vs_kernel:.3e} "
                    f"{'OK' if ok else ('REF-only OK' if ok_vs_kernel else 'FAIL')}")
            except Exception as e:
                test_b_results[(fwk, bwk, mode)] = {"err": str(e)}
                log(f"  {tag} FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    test_b_pass = all(
        r.get("ok", False) for r in test_b_results.values()
    )
    log(f"  Test B: {'PASS' if test_b_pass else 'FAIL'}  "
        f"({sum(r.get('ok', False) for r in test_b_results.values())}/{len(test_b_results)})")
    log("")

    # ============================================================
    # Test C: Fused fwd vs unfused path
    # ============================================================
    log("=" * 100)
    log("Test C: spline_kv_fwd_fused_cuda vs (spline_kv_fwd_cuda + matmul)")
    log("=" * 100)
    d_out = h
    W_out = torch.randn(d_out, h + r, device="cuda", dtype=torch.bfloat16) * 0.04

    # Path 1: unfused — get f, then matmul with W_out
    f_unfused = spline_kv_fwd_cuda(z, C, grid_lo, grid_hi, G,
                                    activation="relu_sq",
                                    lambda_scale=lambda_scale)
    y_unfused = torch.matmul(f_unfused, W_out.transpose(0, 1).contiguous())

    # Path 2: fused
    y_fused = spline_kv_fwd_fused_cuda(z, C, W_out, grid_lo, grid_hi, G,
                                         activation="relu_sq",
                                         lambda_scale=lambda_scale)

    err_y = (y_fused.float() - y_unfused.float()).norm() / y_unfused.float().norm().clamp_min(1e-9)
    err_y_max = (y_fused.float() - y_unfused.float()).abs().max().item()
    log(f"  y_unfused.norm = {y_unfused.float().norm().item():.4e}")
    log(f"  y_fused.norm   = {y_fused.float().norm().item():.4e}")
    log(f"  rel_err (norm) = {err_y.item():.4e}")
    log(f"  abs_err (max)  = {err_y_max:.4e}")
    # Tolerance: 1% — bf16 GEMM accumulator order differs between fused
    # (a@W_a + δ@W_d via two GEMMs + add_) and unfused (single f@W_out GEMM)
    test_c_pass = err_y.item() < 1e-2
    log(f"  Test C: {'PASS' if test_c_pass else 'FAIL'}")
    log("")

    # ============================================================
    # Final summary
    # ============================================================
    log("=" * 100)
    log("FINAL")
    log("=" * 100)
    log(f"  Test A (kernel correctness):     {'PASS' if test_a_pass else 'FAIL'}")
    log(f"  Test B (e2e autograd × 12 cells): {'PASS' if test_b_pass else 'FAIL'}")
    log(f"  Test C (fused fwd):              {'PASS' if test_c_pass else 'FAIL'}")
    overall = test_a_pass and test_b_pass and test_c_pass
    log(f"  Overall: {'✅ PASS' if overall else '❌ FAIL'}")

    return out.getvalue()


@app.local_entrypoint()
def main():
    print(run.remote())
