"""Tight A/B/C microbench: triton fwd vs cuda fwd (+ no_base variant),
all under CUDA Graph capture. Same model, same input, same wgmma bwd.

Verifies the hypothesis: under cuda_graph, is the CUDA fwd actually
slower than Triton fwd?

Test cells:
  A. triton fwd     + wgmma_cuda bwd  + with base
  B. wgmma_cuda fwd + wgmma_cuda bwd  + with base    (current default)
  C. wgmma_cuda fwd + wgmma_cuda bwd  + NO base      (Plan A)

For each: 12-layer FFN-only stack at nanochat shape, captured graph,
measure median step wall over 30 replays.

Run:
  modal run benchmarks/modal_h100_fwd_ab_graph.py
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
app = modal.App("rlkv-fwd-ab-graph", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
def run() -> str:
    import sys, io, time, statistics
    sys.path.insert(0, "/repo/src")
    import torch
    import torch.nn as nn
    out = io.StringIO()
    log = lambda s="": (out.write(s + "\n"), print(s, flush=True))

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"torch: {torch.__version__}")
    log("")

    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature, flash_spline_delta,
    )

    N, h, r, G = 2048, 768, 32, 20
    L = G + 2
    grid_lo, grid_hi = -3.0, 3.0
    n_layers = 12

    log(f"Shape: N={N} h={h} r={r} G={G} L={L} n_layers={n_layers}")
    log(f"All cells: cuda_graph=ON, bwd_kernel=wgmma_cuda")
    log("")

    class WithBaseLayer(nn.Module):
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

    class NoBaseLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = nn.Linear(h, h, bias=False)
            self.C = nn.Parameter(torch.zeros(h, L, r))
            self.W_out = nn.Linear(r, h, bias=False)        # only r → d
            with torch.no_grad():
                s_in = (3.0 / h) ** 0.5
                s_h = (3.0 / r) ** 0.5
                nn.init.uniform_(self.K.weight, -s_in, s_in)
                nn.init.uniform_(self.W_out.weight, -s_h, s_h)
                nn.init.normal_(self.C, std=0.02)

    class WithBaseStack(nn.Module):
        def __init__(self, fwk: str):
            super().__init__()
            self.fwk = fwk
            self.layers = nn.ModuleList([WithBaseLayer() for _ in range(n_layers)])

        def forward(self, x):
            for L_ in self.layers:
                z = L_.K(x)
                f = flash_spline_feature(
                    z, L_.C,
                    grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                    activation="relu_sq", lambda_scale=1.0,
                    use_kernel=True,
                    fwd_kernel=self.fwk, bwd_kernel="wgmma_cuda",
                )
                x = x + L_.W_out(f)
            return x

    class NoBaseStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([NoBaseLayer() for _ in range(n_layers)])

        def forward(self, x):
            for L_ in self.layers:
                z = L_.K(x)
                delta = flash_spline_delta(
                    z, L_.C,
                    grid_lo=grid_lo, grid_hi=grid_hi, G=G,
                    lambda_scale=1.0, bwd_kernel="wgmma_cuda",
                )
                x = x + L_.W_out(delta)
            return x

    cells = [
        ("A. triton_fwd  + wgmma_bwd + base",  lambda: WithBaseStack("triton")),
        ("B. cuda_fwd    + wgmma_bwd + base",  lambda: WithBaseStack("wgmma_cuda")),
        ("C. cuda_fwd    + wgmma_bwd + NO base", lambda: NoBaseStack()),
    ]

    def bench_graph(model, x_const, target, time_n=30):
        static_x = torch.empty_like(x_const).requires_grad_(True)
        static_target = torch.empty_like(target)
        for p in model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(11):
                for p in model.parameters():
                    p.grad.zero_()
                static_x.data.copy_(x_const)
                static_target.copy_(target)
                y = model(static_x)
                ((y - static_target) ** 2).sum().backward()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        for p in model.parameters():
            p.grad.zero_()
        with torch.cuda.graph(g):
            y_g = model(static_x)
            static_loss = ((y_g - static_target) ** 2).sum()
            static_loss.backward()

        samples = []
        for _ in range(time_n):
            static_x.data.copy_(x_const)
            static_target.copy_(target)
            for p in model.parameters():
                p.grad.zero_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(samples), max(samples), min(samples)

    def peak_mb():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    log("=" * 100)
    log(f"{'cell':<42} {'graph wall (ms)':>16} {'min':>8} {'max':>8} {'peak (MB)':>11}")
    log("-" * 100)

    results = {}
    for label, builder in cells:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0)
        model = builder().cuda().to(torch.bfloat16).train()
        x_const = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
        target = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
        try:
            wall_med, wall_max, wall_min = bench_graph(model, x_const, target)
            peak = peak_mb()
            results[label] = (wall_med, wall_min, wall_max, peak)
            log(f"{label:<42} {wall_med:>16.4f} {wall_min:>8.4f} {wall_max:>8.4f} {peak:>11.1f}")
        except Exception as e:
            import traceback
            log(f"{label:<42} FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc())
        finally:
            del model
            torch.cuda.empty_cache()

    log("=" * 100)
    log("")
    if "A. triton_fwd  + wgmma_bwd + base" in results and \
            "B. cuda_fwd    + wgmma_bwd + base" in results:
        a = results["A. triton_fwd  + wgmma_bwd + base"][0]
        b = results["B. cuda_fwd    + wgmma_bwd + base"][0]
        log(f"  B / A   = {b/a:.4f}× ({b-a:+.4f} ms)  ← if > 1.0× then user's hypothesis holds")
    if "B. cuda_fwd    + wgmma_bwd + base" in results and \
            "C. cuda_fwd    + wgmma_bwd + NO base" in results:
        b = results["B. cuda_fwd    + wgmma_bwd + base"][0]
        c = results["C. cuda_fwd    + wgmma_bwd + NO base"][0]
        log(f"  C / B   = {c/b:.4f}× ({c-b:+.4f} ms)  ← Plan A speed effect")

    return out.getvalue()


@app.local_entrypoint()
def main():
    print(run.remote())
