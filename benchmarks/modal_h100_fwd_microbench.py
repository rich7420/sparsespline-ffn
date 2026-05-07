"""Forward-kernel root-cause microbench.

Goal: pinpoint why the new CUDA fwd is slower than Triton fwd in nanochat.
Two layers of measurement:

  Phase A — bare-kernel wall (no autograd, no Python boundary)
    For each implementation, measure 100-iteration mean wall of the inner
    delta computation only. Confirms whether the kernel itself is slower
    or whether it's integration overhead.

  Phase B — torch.profiler op-level breakdown
    Run a 12-layer FFN stack twice (once Triton fwd, once CUDA fwd),
    capture per-op cuda time. Identifies which CUDA op dominates.

Implementations tested:
  triton_v4_no_lambda :  triton.flash_spline_delta_forward_v4 (delta only)
  cuda_fwd_full       :  spline_kv_fwd_cuda (a + delta + pack, our path)
  cuda_fused_fwd      :  spline_kv_fwd_fused_cuda (with W_out matmul fused)

Run:
  modal run benchmarks/modal_h100_fwd_microbench.py
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
        ignore=[".venv/**", ".git/**", "nanochat/.venv/**",
                "nanochat/.nanochat-runtime/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("rlkv-fwd-microbench", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
def run() -> str:
    import sys, io, time, statistics
    sys.path.insert(0, "/repo/src")
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function
    out = io.StringIO()
    log = lambda s="": (out.write(s + "\n"), print(s, flush=True))

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"torch: {torch.__version__}")
    log("")

    from sparsespline_ffn.cuda_ext import (
        spline_kv_fwd_cuda, spline_kv_fwd_fused_cuda,
    )
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_delta_forward_v4 as triton_delta_v4,
        flash_spline_feature_forward as triton_full_fwd,
    )

    # nanochat shape
    N, h, r, G, L_ord = 2048, 768, 32, 20, 2
    L = G + L_ord
    grid_lo, grid_hi = -3.0, 3.0
    lambda_scale = 1.0

    torch.manual_seed(0)
    z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
    C = (torch.randn(h, L, r, device="cuda", dtype=torch.bfloat16) * 0.05)
    W_out = (torch.randn(h, h + r, device="cuda", dtype=torch.bfloat16) * 0.04)

    def time_fn(fn, n_iters=100, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / n_iters

    # =========================================================
    # Phase A — bare-kernel wall comparison
    # =========================================================
    log("=" * 100)
    log("Phase A — bare-kernel wall (single layer at nanochat shape)")
    log("=" * 100)
    log(f"  N={N} h={h} r={r} G={G} L={L}")
    log("")

    impls = {
        "triton_delta_v4 (delta only)":
            lambda: triton_delta_v4(z, C, grid_lo, grid_hi, G),
        "triton_full_fwd (a + delta + cat, version=v4)":
            lambda: triton_full_fwd(z, C, grid_lo, grid_hi, G,
                                      activation="relu_sq",
                                      lambda_scale=lambda_scale,
                                      version="v4"),
        "cuda_fwd (activation + delta + pack)":
            lambda: spline_kv_fwd_cuda(z, C, grid_lo, grid_hi, G,
                                         activation="relu_sq",
                                         lambda_scale=lambda_scale),
        "cuda_fused_fwd (+ W_out matmul)":
            lambda: spline_kv_fwd_fused_cuda(z, C, W_out, grid_lo, grid_hi, G,
                                               activation="relu_sq",
                                               lambda_scale=lambda_scale),
    }

    bench: dict = {}
    for name, fn in impls.items():
        try:
            ms = time_fn(fn)
            bench[name] = ms
            log(f"  {name:<55} {ms:>8.3f} ms / call")
        except Exception as e:
            bench[name] = float("nan")
            log(f"  {name:<55} FAILED: {type(e).__name__}: {e}")

    log("")
    log("Ratios:")
    if "triton_full_fwd (a + delta + cat, version=v4)" in bench and \
            "cuda_fwd (activation + delta + pack)" in bench:
        t = bench["triton_full_fwd (a + delta + cat, version=v4)"]
        c = bench["cuda_fwd (activation + delta + pack)"]
        log(f"  cuda_fwd / triton_full = {c/t:.3f}× ({c-t:+.3f} ms / call)")
    log("")

    # =========================================================
    # Phase B — torch.profiler per-op breakdown (12-layer stack)
    # =========================================================
    log("=" * 100)
    log("Phase B — torch.profiler per-op breakdown (12-layer FFN stack)")
    log("=" * 100)

    import torch.nn as nn
    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature,
    )
    n_layers = 12

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = nn.Linear(h, h, bias=False)
            self.C = nn.Parameter(torch.zeros(h, L, r))
            self.W_out = nn.Linear(h + r, h, bias=False)
            with torch.no_grad():
                s_in = (3.0 / h) ** 0.5
                s_h = (3.0 / (h + r)) ** 0.5
                nn.init.uniform_(self.K.weight, -s_in, s_in)
                nn.init.uniform_(self.W_out.weight, -s_h, s_h)
                nn.init.normal_(self.C, std=0.01)

    class Stack(nn.Module):
        def __init__(self, fwk: str):
            super().__init__()
            self.fwk = fwk
            self.layers = nn.ModuleList([Layer() for _ in range(n_layers)])

        def forward(self, x):
            for L_ in self.layers:
                z = L_.K(x)
                f = flash_spline_feature(
                    z, L_.C,
                    grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                    activation="relu_sq", lambda_scale=lambda_scale,
                    use_kernel=True,
                    fwd_kernel=self.fwk, bwd_kernel="wgmma_cuda",
                )
                x = x + L_.W_out(f)
            return x

    for fwk in ("triton", "wgmma_cuda"):
        log(f"\n--- fwk={fwk} ---")
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        model = Stack(fwk).cuda().to(torch.bfloat16).train()
        x_const = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
        target = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)

        # warmup
        for _ in range(15):
            model.zero_grad(set_to_none=True)
            x = x_const.detach().requires_grad_(True)
            y = model(x)
            ((y - target) ** 2).sum().backward()
        torch.cuda.synchronize()

        # profile 5 steps
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                      record_shapes=False) as prof:
            for _ in range(5):
                model.zero_grad(set_to_none=True)
                with record_function("STEP"):
                    with record_function("fwd"):
                        x = x_const.detach().requires_grad_(True)
                        y = model(x)
                    with record_function("bwd"):
                        ((y - target) ** 2).sum().backward()

        # Use the built-in table() — handles both old (cuda_time_total) and new
        # (device_time_total) PyTorch attribute names.
        log(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=18))

        # Wall measurement (just for reference)
        ms = time_fn(lambda: (model.zero_grad(set_to_none=True),
                                ((model(x_const.detach().requires_grad_(True))
                                   - target) ** 2).sum().backward())[1],
                      n_iters=30, warmup=10)
        log(f"  ({fwk}) eager step wall = {ms:.3f} ms / step")

        del model
        torch.cuda.empty_cache()

    return out.getvalue()


@app.local_entrypoint()
def main():
    print(run.remote())
