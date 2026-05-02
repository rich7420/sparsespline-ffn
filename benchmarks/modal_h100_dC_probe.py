"""H100 single-layer dC probe — isolates the dead-spline-branch hypothesis.

The training run shows mean_C_norm = 0.0, mean_C_grad_norm = 0.0,
delta_over_base = 0.0 across 500 steps. C is dead. This probe creates a
SINGLE RL-KV layer (not the full nanochat model), runs forward + backward on
random inputs, and reports C.grad.norm for each backward path:

  PyTorch reference (use_kernel=False)
  Triton            (use_kernel=True,  bwd_kernel=triton)
  hopper_cuda       (use_kernel=True,  bwd_kernel=hopper_cuda)
  wgmma_cuda        (use_kernel=True,  bwd_kernel=wgmma_cuda)

If reference shows nonzero dC and CUDA paths show zero → kernel bug.
If all paths show nonzero → integration bug (likely CUDA Graph + autograd Function).
If all show zero → upstream bug (lambda_scale, W_out_delta init, etc).

This is the minimum test to localize the issue.

Run:
  modal run benchmarks/modal_h100_dC_probe.py
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
app = modal.App("rlkv-dC-probe", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run() -> str:
    import sys, io
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn.rl_spline_kv_reference import (
        RLSplineKVReference, RLSplineKVConfig,
    )

    out = io.StringIO()
    log = lambda s="": (out.write(s + "\n"), print(s, flush=True))

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"torch: {torch.__version__}")
    log("")

    # Match the actual nanochat shapes
    d, h_ratio, r, G, L_ord = 768, 1.0, 32, 20, 2
    h = int(d * h_ratio)
    L = G + L_ord  # 22
    N = 2048  # 2 * 1024 tokens (B=2, T=1024)

    log(f"Shape: N={N}  d={d}  h={h}  r={r}  G={G}  L={L}  spline_order={L_ord}")
    log(f"       all bf16 on H100; init_C_zero=True (cold-start v7 spec)")
    log("")

    paths = [
        # (label, use_kernel, bwd_kernel)
        ("reference (PyTorch, use_kernel=False)", False, "triton"),  # bwd_kernel ignored
        ("Triton (use_kernel=True, bwd=triton)",   True,  "triton"),
        ("hopper_cuda backward",                   True,  "hopper_cuda"),
        ("wgmma_cuda backward",                    True,  "wgmma_cuda"),
    ]

    # Same input + init across all paths (deterministic)
    torch.manual_seed(42)
    x_init = torch.randn(N, d, device="cuda", dtype=torch.bfloat16)
    target_init = torch.randn(N, d, device="cuda", dtype=torch.bfloat16)

    # Snapshot initial K, W_out so all paths start identical
    cfg_seed = RLSplineKVConfig(
        d=d, h_ratio=h_ratio, r=r, G=G, spline_order=L_ord,
        activation="relu_sq", lambda_scale=1.0, init_C_zero=True,
        use_kernel=False, bwd_kernel="triton",
    )
    seed_ffn = RLSplineKVReference(cfg_seed).cuda().to(torch.bfloat16)
    K_w_seed = seed_ffn.K.weight.detach().clone()
    W_out_w_seed = seed_ffn.W_out.weight.detach().clone()
    C_seed = seed_ffn.C.detach().clone()  # zeros from init
    del seed_ffn

    log(f"Init checks:  K.weight.norm = {K_w_seed.norm().item():.4f}")
    log(f"              W_out.weight.norm = {W_out_w_seed.norm().item():.4f}  "
        f"(W_out_delta cols [{h}:].norm = {W_out_w_seed[:, h:].norm().item():.4f})")
    log(f"              C.norm = {C_seed.norm().item():.4f}  "
        f"(should be 0 from init_C_zero)")
    log("")
    log("=" * 100)

    def build_ffn(use_kernel: bool, bwd_kernel: str):
        cfg = RLSplineKVConfig(
            d=d, h_ratio=h_ratio, r=r, G=G, spline_order=L_ord,
            activation="relu_sq", lambda_scale=1.0, init_C_zero=True,
            use_kernel=use_kernel, bwd_kernel=bwd_kernel,
        )
        ffn = RLSplineKVReference(cfg).cuda().to(torch.bfloat16)
        with torch.no_grad():
            ffn.K.weight.copy_(K_w_seed)
            ffn.W_out.weight.copy_(W_out_w_seed)
            ffn.C.copy_(C_seed)
        return ffn

    def report(tag: str, ffn, loss_val: float, x_grad):
        c_grad = ffn.C.grad
        c_grad_norm = c_grad.norm().item() if c_grad is not None else 0.0
        c_grad_max = c_grad.abs().max().item() if c_grad is not None else 0.0
        wo_grad = ffn.W_out.weight.grad
        ko_grad = ffn.K.weight.grad
        wo_a = wo_grad[:, :h].norm().item() if wo_grad is not None else 0.0
        wo_d = wo_grad[:, h:].norm().item() if wo_grad is not None else 0.0
        log(f"  [{tag}] loss = {loss_val:.4f}  C.grad.norm = {c_grad_norm:.6e}  "
            f"C.grad.max = {c_grad_max:.4e}")
        log(f"  [{tag}] K.weight.grad.norm = "
            f"{(ko_grad.norm().item() if ko_grad is not None else 0.0):.4e}  "
            f"W_out base [:{h}] = {wo_a:.4e}  W_out delta [{h}:] = {wo_d:.4e}")
        if x_grad is not None:
            log(f"  [{tag}] x.grad.norm = {x_grad.norm().item():.4e}")
        return c_grad_norm

    results: dict[str, dict[str, float]] = {}
    for label, use_kernel, bwd_kernel in paths:
        log(f"\n--- {label} ---")
        results[label] = {}
        # ----- (1) eager pass -----
        try:
            torch.manual_seed(42)
            ffn = build_ffn(use_kernel, bwd_kernel)
            x = x_init.detach().clone().requires_grad_(True)
            ffn.zero_grad(set_to_none=True)
            y = ffn(x)
            loss = (y - target_init).pow(2).sum()
            loss.backward()
            results[label]["eager"] = report("eager", ffn, loss.item(), x.grad)
            del ffn
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            log(f"  [eager] FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            results[label]["eager"] = float("nan")

        # ----- (2) graph pass -----
        try:
            torch.manual_seed(42)
            ffn = build_ffn(use_kernel, bwd_kernel)
            # Pre-allocate static gradients (CUDA Graph requirement)
            for p in ffn.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            static_x = x_init.detach().clone().requires_grad_(True)
            static_x.grad = torch.zeros_like(static_x)
            static_target = target_init.detach().clone()

            # Warmup on side stream (PyTorch CUDA Graph protocol)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    for p in ffn.parameters():
                        p.grad.zero_()
                    static_x.grad.zero_()
                    yw = ffn(static_x)
                    lw = (yw - static_target).pow(2).sum()
                    lw.backward()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            # Capture
            g = torch.cuda.CUDAGraph()
            for p in ffn.parameters():
                p.grad.zero_()
            static_x.grad.zero_()
            with torch.cuda.graph(g):
                y_g = ffn(static_x)
                static_loss = (y_g - static_target).pow(2).sum()
                static_loss.backward()

            # Replay (zero grads outside graph; graph accumulates)
            for p in ffn.parameters():
                p.grad.zero_()
            static_x.grad.zero_()
            g.replay()
            torch.cuda.synchronize()

            results[label]["graph"] = report(
                "graph", ffn, static_loss.item(), static_x.grad,
            )
            del ffn, g
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            log(f"  [graph] FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            results[label]["graph"] = float("nan")

    # ============================================================
    # Summary
    # ============================================================
    log("")
    log("=" * 100)
    log("=== SUMMARY: C.grad.norm by path × mode")
    log("=" * 100)
    log(f"{'path':<48} {'eager':>16} {'graph':>16} {'eager == graph?':>18}")
    ref_eager = results.get("reference (PyTorch, use_kernel=False)", {}).get("eager", 0.0)
    ref_graph = results.get("reference (PyTorch, use_kernel=False)", {}).get("graph", 0.0)
    for label, _, _ in paths:
        e = results.get(label, {}).get("eager", float("nan"))
        gph = results.get(label, {}).get("graph", float("nan"))
        match = "YES" if (
            isinstance(e, float) and isinstance(gph, float)
            and abs(e - gph) / max(abs(e), 1e-9) < 0.05
        ) else "NO"
        log(f"{label:<48} {e:>16.4e} {gph:>16.4e} {match:>18}")

    log("")
    log("Interpretation:")
    log("  PASS: every path shows eager == graph (≈ same C.grad.norm) AND > 0")
    log("        ⇒ CUDA Graph stream-safety fix is correct for that path")
    log("  FAIL: graph column = 0 while eager > 0  ⇒ kernel still launches on stream 0")
    log("  Reference and Triton paths must always pass (PyTorch + Triton handle streams)")
    log("  hopper_cuda / wgmma_cuda: this is the post-fix verification")

    return out.getvalue()


@app.local_entrypoint()
def main():
    print(run.remote())
