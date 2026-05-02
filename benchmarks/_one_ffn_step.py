"""Single-step FFN driver — invoked under NCU to capture per-kernel metrics.

Runs warmup + N profiled steps for ONE FFN cell. Used by
modal_h100_hypothesis_probe.py via:

    ncu --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
        --section LaunchStats --section SchedulerStats --section Occupancy \
        --section SourceCounters --target-processes all --csv \
        python benchmarks/_one_ffn_step.py --cell <name> --steps 1

Cells:
  mlp_h_4d         : baseline ReLU² MLP
  rl_kv_hopperCUDA : RL-KV B2 r=32 L=22 with hopper_cuda backward
  rl_kv_wgmmaCUDA  : RL-KV B2 r=32 L=22 with wgmma_cuda backward
  rl_kv_triton     : RL-KV B2 r=32 L=22 with Triton backward (control)
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--t", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--l-grid", type=int, default=22)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--steps", type=int, default=1,
                     help="profiled steps (NCU captures all of them)")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    from sparsespline_ffn import MLPFFN
    from sparsespline_ffn.rl_spline_kv_reference import RLSplineKVConfig
    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature,
    )

    class RLKVCell(nn.Module):
        def __init__(self, d, r, l_grid, bwd_kernel):
            super().__init__()
            self.d = d
            # L is derived from G + spline_order; for L=22 we use G=20 spline_order=2.
            G = max(1, l_grid - 2)
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
            )
            return self.W_out(f).reshape(shape)

    class Stack(nn.Module):
        def __init__(self, builder, n_layers):
            super().__init__()
            self.layers = nn.ModuleList([builder() for _ in range(n_layers)])

        def forward(self, x):
            for l in self.layers:
                x = x + l(x)
            return x

    if args.cell == "mlp_h_4d":
        model = Stack(lambda: MLPFFN(d=args.d, mlp_ratio=4), args.n_layers)
    elif args.cell == "rl_kv_hopperCUDA":
        model = Stack(lambda: RLKVCell(args.d, args.r, args.l_grid,
                                          "hopper_cuda"), args.n_layers)
    elif args.cell == "rl_kv_wgmmaCUDA":
        model = Stack(lambda: RLKVCell(args.d, args.r, args.l_grid,
                                          "wgmma_cuda"), args.n_layers)
    elif args.cell == "rl_kv_triton":
        model = Stack(lambda: RLKVCell(args.d, args.r, args.l_grid,
                                          "triton"), args.n_layers)
    else:
        raise ValueError(f"unknown cell: {args.cell}")

    model = model.to(device=device, dtype=dtype).train()
    x_const = torch.randn(args.b, args.t, args.d, device=device, dtype=dtype)
    target = torch.randn(args.b, args.t, args.d, device=device, dtype=dtype)

    # warmup (out of NCU range — NCU's --launch-skip is honored)
    for _ in range(args.warmup):
        model.zero_grad(set_to_none=True)
        x = x_const.detach().requires_grad_(True)
        y = model(x)
        ((y - target) ** 2).sum().backward()
    torch.cuda.synchronize()

    # profiled steps — NCU captures all kernels launched here
    print(f"[probe] starting {args.steps} profiled step(s) for cell={args.cell}",
          flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        model.zero_grad(set_to_none=True)
        x = x_const.detach().requires_grad_(True)
        y = model(x)
        ((y - target) ** 2).sum().backward()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0 / args.steps
    print(f"[probe] cell={args.cell}  wall (mean over {args.steps} step) "
          f"= {wall_ms:.3f} ms/step", flush=True)


if __name__ == "__main__":
    main()
