"""H100 hypothesis-probe profiler for RL-KV vs MLP wall gap.

Runs three things on H100:

  (A) torch.profiler per-op breakdown — for each cell, eager + CUDA-Graphed,
      reports wall + top-25 cuda ops + launch count.
      → tests H1 (which kernel dominates), H2 (does graph close the gap),
        H3 (is GEMM on cuBLAS fast path), H4 (launch count).

  (B) NCU sectioned profile — for each RL-KV cell (B2 hopperCUDA + B2 wgmmaCUDA),
      captures per-kernel SM occupancy, HBM throughput, register spill,
      tensor core utilization. Identifies WHY each kernel is slow.

  (C) Summary table that interprets results against the four hypotheses.

Cells profiled:
  mlp_h_4d            — baseline ReLU² MLP
  rl_kv_hopperCUDA    — B2 r=32 L=22 with hopper_cuda backward
  rl_kv_wgmmaCUDA     — B2 r=32 L=22 with wgmma_cuda backward

Run:
  modal run benchmarks/modal_h100_hypothesis_probe.py
"""
from __future__ import annotations

import modal


IMAGE = (
    # devel image ships ncu (Nsight Compute CLI) by default.
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
app = modal.App("rlkv-hypothesis-probe", image=IMAGE)


CELLS = ["mlp_h_4d", "rl_kv_hopperCUDA", "rl_kv_wgmmaCUDA"]
RL_KV_CELLS = ["rl_kv_hopperCUDA", "rl_kv_wgmmaCUDA"]


# ============================================================
# (A) torch.profiler breakdown
# ============================================================
@app.function(gpu="H100", timeout=900)
def torch_profiler(
    d: int = 768, b: int = 2, t: int = 1024,
    n_layers: int = 12, r: int = 32, l_grid: int = 22,
) -> str:
    import sys, io, statistics, time
    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo/benchmarks")
    import torch, torch.nn as nn
    from torch.profiler import profile, ProfilerActivity, record_function

    out = io.StringIO()
    log = lambda s="": (out.write(s + "\n"), print(s, flush=True))

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"shape: B={b} T={t} d={d} n_layers={n_layers} r={r} l_grid={l_grid}")
    log(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")
    log("")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    from sparsespline_ffn import MLPFFN
    from sparsespline_ffn.rl_spline_kv_reference import RLSplineKVConfig
    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature, flash_spline_delta,
    )

    class RLKVCell(nn.Module):
        def __init__(self, d, r, l_grid, bwd_kernel):
            super().__init__()
            self.d = d
            G = max(1, l_grid - 2)  # L = G + spline_order
            self.cfg = RLSplineKVConfig(
                d=d, h_ratio=1.0, r=r, G=G,
                spline_order=2, lambda_scale=1.0,
                grid_lo=-3.0, grid_hi=3.0, activation="relu_sq",
            )
            h = d
            self.K = nn.Linear(d, h, bias=False)
            self.C = nn.Parameter(torch.zeros(h, l_grid, r))
            self.W_out = nn.Linear(h + r, d, bias=False)
            self.bwd_kernel = bwd_kernel
            with torch.no_grad():
                s_in = (3.0 / d) ** 0.5
                s_h = (3.0 / (h + r)) ** 0.5
                nn.init.uniform_(self.K.weight, -s_in, s_in)
                nn.init.uniform_(self.W_out.weight, -s_h, s_h)
                nn.init.normal_(self.C, std=0.01)

        def forward(self, x):
            shape = x.shape
            x_flat = x.reshape(-1, self.d)
            z = self.K(x_flat)
            f = flash_spline_feature(
                z, self.C,
                grid_lo=float(self.cfg.grid_lo),
                grid_hi=float(self.cfg.grid_hi),
                G=int(self.cfg.G),
                activation=self.cfg.activation,
                lambda_scale=float(self.cfg.lambda_scale),
                use_kernel=True, bwd_kernel=self.bwd_kernel,
                fwd_kernel="triton",  # match production cells (graph-friendly)
            )
            return self.W_out(f).reshape(shape)

    class RLKVNoBaseCell(nn.Module):
        """Production NOBASE cell — matches nanochat rl_kv_*_NOBASE_all12.

        - W_out is Linear(r, d), not Linear(h+r, d) (no activation half).
        - Uses FlashSplineDelta (Triton delta-only fwd + wgmma cuda bwd).
        - C is zero-initialised (cold start, learns from gradient flow).
        """
        def __init__(self, d, r, l_grid, bwd_kernel):
            super().__init__()
            self.d = d
            G = max(1, l_grid - 2)
            self.cfg = RLSplineKVConfig(
                d=d, h_ratio=1.0, r=r, G=G,
                spline_order=2, lambda_scale=1.0,
                grid_lo=-3.0, grid_hi=3.0, activation="relu_sq",
                no_base=True,
            )
            h = d
            self.K = nn.Linear(d, h, bias=False)
            self.C = nn.Parameter(torch.zeros(h, l_grid, r))
            self.W_out = nn.Linear(r, d, bias=False)
            self.bwd_kernel = bwd_kernel
            with torch.no_grad():
                s_in = (3.0 / d) ** 0.5
                s_h = (3.0 / r) ** 0.5
                nn.init.uniform_(self.K.weight, -s_in, s_in)
                nn.init.uniform_(self.W_out.weight, -s_h, s_h)
                # C zero-init (Plan A Fix 1)

        def forward(self, x):
            shape = x.shape
            x_flat = x.reshape(-1, self.d)
            z = self.K(x_flat)
            delta = flash_spline_delta(
                z, self.C,
                grid_lo=float(self.cfg.grid_lo),
                grid_hi=float(self.cfg.grid_hi),
                G=int(self.cfg.G),
                lambda_scale=float(self.cfg.lambda_scale),
                bwd_kernel=self.bwd_kernel,
            )
            return self.W_out(delta).reshape(shape)

    class Stack(nn.Module):
        def __init__(self, builder, n_layers):
            super().__init__()
            self.layers = nn.ModuleList([builder() for _ in range(n_layers)])

        def forward(self, x):
            for l in self.layers:
                x = x + l(x)
            return x

    builders = {
        "mlp_h_4d":             lambda: Stack(lambda: MLPFFN(d=d, mlp_ratio=4), n_layers),
        "rl_kv_hopperCUDA":     lambda: Stack(lambda: RLKVCell(d, r, l_grid, "hopper_cuda"),
                                                n_layers),
        "rl_kv_wgmmaCUDA":      lambda: Stack(lambda: RLKVCell(d, r, l_grid, "wgmma_cuda"),
                                                n_layers),
        "rl_kv_NOBASE_wgmma":   lambda: Stack(lambda: RLKVNoBaseCell(d, r, l_grid, "wgmma_cuda"),
                                                n_layers),
        "rl_kv_NOBASE_hopper":  lambda: Stack(lambda: RLKVNoBaseCell(d, r, l_grid, "hopper_cuda"),
                                                n_layers),
    }

    def run_one(name: str, model: nn.Module, *,
                use_graph: bool, prof_n: int = 5, time_n: int = 30,
                warmup: int = 15):
        x_const = torch.randn(b, t, d, device=device, dtype=dtype)
        target = torch.randn(b, t, d, device=device, dtype=dtype)

        if not use_graph:
            for _ in range(warmup):
                model.zero_grad(set_to_none=True)
                x = x_const.detach().requires_grad_(True)
                y = model(x)
                ((y - target) ** 2).sum().backward()
            torch.cuda.synchronize()

            samples = []
            for _ in range(time_n):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                x = x_const.detach().requires_grad_(True)
                y = model(x)
                ((y - target) ** 2).sum().backward()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - t0) * 1000.0)
            wall_ms = statistics.median(samples)

            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                          record_shapes=False) as prof:
                for _ in range(prof_n):
                    model.zero_grad(set_to_none=True)
                    with record_function("STEP"):
                        with record_function("fwd"):
                            x = x_const.detach().requires_grad_(True)
                            y = model(x)
                        with record_function("bwd"):
                            ((y - target) ** 2).sum().backward()
            prof_table = prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=25)
            kernel_events = [e for e in prof.events()
                             if e.device_type == torch.autograd.DeviceType.CUDA
                             and e.cuda_time > 0]
            launches = len(kernel_events) // max(prof_n, 1)
            return wall_ms, prof_table, launches

        # CUDA Graph path
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
            loss_g = ((y_g - static_target) ** 2).sum()
            loss_g.backward()

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
        wall_ms = statistics.median(samples)

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                      record_shapes=False) as prof:
            for _ in range(prof_n):
                static_x.data.copy_(x_const)
                static_target.copy_(target)
                for p in model.parameters():
                    p.grad.zero_()
                with record_function("STEP_GRAPH"):
                    g.replay()
        prof_table = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=25)
        kernel_events = [e for e in prof.events()
                         if e.device_type == torch.autograd.DeviceType.CUDA
                         and e.cuda_time > 0]
        launches = len(kernel_events) // max(prof_n, 1)
        return wall_ms, prof_table, launches

    summary = {}
    for cell_name, builder in builders.items():
        for use_graph in [False, True]:
            tag = "graph" if use_graph else "eager"
            log("=" * 100)
            log(f"=== {cell_name}  ({tag})")
            log("=" * 100)
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            model = None
            try:
                model = builder().to(device=device, dtype=dtype).train()
                wall, table, launches = run_one(
                    cell_name, model, use_graph=use_graph)
                summary[(cell_name, tag)] = {
                    "wall_ms": wall, "launches_per_step": launches,
                }
                log(f"wall (median, ms)        : {wall:.3f}")
                log(f"kernel launches per step : {launches}")
                log("")
                log(table)
                log("")
            except Exception as e:
                import traceback
                log(f"FAILED: {type(e).__name__}: {e}")
                log(traceback.format_exc())
                summary[(cell_name, tag)] = {"error": str(e)}
            finally:
                if model is not None:
                    del model
                torch.cuda.empty_cache()

    log("=" * 100)
    log("=== TORCH-PROFILER SUMMARY (wall + launches)")
    log("=" * 100)
    log(f"{'cell':<22} {'mode':<7} {'wall (ms)':>10} {'launches':>10}")
    for (cn, tag), v in summary.items():
        if "wall_ms" in v:
            log(f"{cn:<22} {tag:<7} {v['wall_ms']:>10.3f} {v['launches_per_step']:>10}")
        else:
            log(f"{cn:<22} {tag:<7} ERR: {v.get('error', '?')}")

    log("")
    if all((c, "eager") in summary and "wall_ms" in summary[(c, "eager")]
            for c in ["mlp_h_4d", "rl_kv_wgmmaCUDA"]):
        mlp_e = summary[("mlp_h_4d", "eager")]["wall_ms"]
        for cn in RL_KV_CELLS:
            if (cn, "eager") in summary and "wall_ms" in summary[(cn, "eager")]:
                rl_e = summary[(cn, "eager")]["wall_ms"]
                log(f"eager : {cn} / MLP wall ratio  = {rl_e/mlp_e:.2f}×   "
                    f"delta = {rl_e - mlp_e:.2f} ms/step")
    if all((c, "graph") in summary and "wall_ms" in summary[(c, "graph")]
            for c in ["mlp_h_4d", "rl_kv_wgmmaCUDA"]):
        mlp_g = summary[("mlp_h_4d", "graph")]["wall_ms"]
        for cn in RL_KV_CELLS:
            if (cn, "graph") in summary and "wall_ms" in summary[(cn, "graph")]:
                rl_g = summary[(cn, "graph")]["wall_ms"]
                log(f"graph : {cn} / MLP wall ratio  = {rl_g/mlp_g:.2f}×   "
                    f"delta = {rl_g - mlp_g:.2f} ms/step")
    for cn in RL_KV_CELLS:
        if (cn, "eager") in summary and (cn, "graph") in summary \
                and "wall_ms" in summary[(cn, "eager")] \
                and "wall_ms" in summary[(cn, "graph")]:
            rl_e = summary[(cn, "eager")]["wall_ms"]
            rl_g = summary[(cn, "graph")]["wall_ms"]
            log(f"{cn}: graph saves {(rl_e-rl_g):.2f} ms/step "
                f"({100*(rl_e - rl_g)/rl_e:.1f}%)")

    return out.getvalue()


# ============================================================
# (B) NCU sectioned profile per cell
# ============================================================
@app.function(gpu="H100", timeout=1800)
def ncu_profile(
    cell: str,
    d: int = 768, b: int = 2, t: int = 1024,
    n_layers: int = 12, r: int = 32, l_grid: int = 22,
    launch_skip: int = 200,
    launch_count: int = 400,
) -> str:
    import os, subprocess, sys
    env = {
        **os.environ,
        "PYTHONPATH": "/repo/src:/repo/benchmarks",
    }

    # Confirm ncu is present
    which = subprocess.run(["which", "ncu"], capture_output=True, text=True)
    print(f"[ncu] which ncu: {which.stdout.strip()}  rc={which.returncode}",
          flush=True)
    ver = subprocess.run(["ncu", "--version"], capture_output=True, text=True)
    print(f"[ncu] version:\n{ver.stdout}", flush=True)

    # NCU `--launch-skip` and `--launch-count` operate on launch-ordinal, not Python
    # steps. Each training step launches ~120-200 kernels (12 layers × ~10 kernels
    # fwd+bwd, plus framework ops). With warmup=20 steps, the warmup phase issues
    # ~2400-4000 launches → set launch-skip=4000 to be safe. Then capture
    # launch-count=400 (~2 full steps) so per-kernel statistics are stable.
    sections = [
        "ComputeWorkloadAnalysis",
        "MemoryWorkloadAnalysis",
        "LaunchStats",
        "Occupancy",
        "SchedulerStats",
        "WarpStateStats",
        "InstructionStats",
        "SourceCounters",
    ]
    cmd = [
        "ncu",
        "--target-processes", "all",
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
        "--print-summary", "per-kernel",
        "--csv",
    ]
    for s in sections:
        cmd += ["--section", s]

    cmd += [
        sys.executable, "/repo/benchmarks/_one_ffn_step.py",
        "--cell", cell,
        "--d", str(d), "--b", str(b), "--t", str(t),
        "--n-layers", str(n_layers), "--r", str(r), "--l-grid", str(l_grid),
        "--warmup", "20",
        "--steps", "2",
    ]

    print(f"[ncu] running cell={cell}", flush=True)
    print(f"[ncu] cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            cwd="/repo")
    print(f"[ncu] rc={proc.returncode}", flush=True)
    out = f"=== NCU profile: cell={cell} ===\n"
    out += "stdout:\n" + proc.stdout + "\n"
    if proc.returncode != 0:
        out += "stderr:\n" + proc.stderr + "\n"
    print(out, flush=True)
    return out


# ============================================================
# Local entrypoint
# ============================================================
@app.local_entrypoint()
def main(d: int = 768, b: int = 2, t: int = 1024,
         n_layers: int = 12, r: int = 32, l_grid: int = 22,
         skip_ncu: bool = False) -> None:
    print("\n#### PART A: torch.profiler breakdown (eager + graph)\n")
    print(torch_profiler.remote(
        d=d, b=b, t=t, n_layers=n_layers, r=r, l_grid=l_grid))

    if skip_ncu:
        print("\n[skip_ncu=True; skipping NCU passes]\n")
        return

    print("\n#### PART B: NCU sectioned profile per cell\n")
    for cell in CELLS:
        print(f"\n---- ncu_profile cell={cell} ----")
        print(ncu_profile.remote(
            cell=cell,
            d=d, b=b, t=t, n_layers=n_layers, r=r, l_grid=l_grid))
