"""Comprehensive FFN comparison: speed + VRAM + params.

Compares every FFN variant we have at d=768 (nanochat default),
B=2, T=1024 (matching Stage-3 training shapes).  Reports forward,
backward, and forward+backward wall, plus peak VRAM during training,
plus parameter count.

Variants:
  mlp_h_4d            : nanochat baseline (4× hidden ReLU²)
  mlp_h_2d            : narrow_relu2_h_2d (2× hidden)
  mlp_h_d             : narrow_relu2_h_d (1× hidden)
  ss_h_d_half         : SimpleSpline B2 h=d/2 (= ss_pa6 architecture)
  ss_h_d              : SimpleSpline B2 h=d (= ss_pa6_h_d architecture)
  swiglu_h_d          : SwiGLU h=d
  spline_glu_h_d      : SplineGLU B2 h=d
  rl_kv_r32_ref       : RL-Spline-KV r=32, pure-PyTorch reference
  rl_kv_r32_kernel    : RL-Spline-KV r=32, FlashSplineFeature kernel fwd
  rl_kv_r64_kernel    : RL-Spline-KV r=64, FlashSplineFeature kernel fwd

Run:
  python benchmarks/ffn_full_compare.py
  python benchmarks/ffn_full_compare.py --d 1024 --bt 4096
  python benchmarks/ffn_full_compare.py --json-out out.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from sparsespline_ffn import MLPFFN
from sparsespline_ffn.simple_spline_mlp import SimpleSplineMLP, SimpleSplineConfig
from sparsespline_ffn.glu_ffn import SwiGLU, GLUConfig, SplineGLU, SplineGLUConfig
from sparsespline_ffn.rl_spline_kv_reference import (
    RLSplineKVReference, RLSplineKVConfig,
)


def _make_modules(d: int) -> dict[str, Callable[[], nn.Module]]:
    """Builders (lazy) for each FFN variant.  All take no args."""
    return {
        "mlp_h_4d":          lambda: MLPFFN(d=d, mlp_ratio=4),
        "mlp_h_2d":          lambda: MLPFFN(d=d, mlp_ratio=2),
        "mlp_h_d":           lambda: MLPFFN(d=d, mlp_ratio=1),
        "ss_h_d_half":       lambda: SimpleSplineMLP(SimpleSplineConfig(d=d, h_ratio=0.5, G=20, use_kernel=True)),
        "ss_h_d":            lambda: SimpleSplineMLP(SimpleSplineConfig(d=d, h_ratio=1.0, G=20, use_kernel=True)),
        "swiglu_h_d":        lambda: SwiGLU(GLUConfig(d=d, mlp_ratio=1.0)),
        "spline_glu_h_d":    lambda: SplineGLU(SplineGLUConfig(d=d, mlp_ratio=1.0, G=20, use_kernel=True)),
        "rl_kv_r32_ref":     lambda: _RLKVWrap(d=d, r=32, use_kernel=False),
        "rl_kv_r32_kernel":  lambda: _RLKVWrap(d=d, r=32, use_kernel=True),
        "rl_kv_r64_kernel":  lambda: _RLKVWrap(d=d, r=64, use_kernel=True),
    }


class _RLKVWrap(nn.Module):
    """Thin wrapper that uses the FlashSplineFeature autograd Function
    (kernel fwd + ref bwd) when use_kernel=True, else the full PyTorch
    reference module.
    """

    def __init__(self, d: int, r: int, use_kernel: bool):
        super().__init__()
        self.use_kernel = use_kernel
        if use_kernel:
            self.K = nn.Linear(d, d, bias=False)
            self.cfg = RLSplineKVConfig(d=d, h_ratio=1.0, r=r, G=20)
            self.h = d
            self.C = nn.Parameter(torch.zeros(d, 22, r))
            self.W_out = nn.Linear(d + r, d, bias=False)
            with torch.no_grad():
                s_in = (3.0 / d) ** 0.5
                s_h  = (3.0 / (d + r)) ** 0.5
                nn.init.uniform_(self.K.weight, -s_in, s_in)
                nn.init.uniform_(self.W_out.weight, -s_h, s_h)
        else:
            self.ref = RLSplineKVReference(RLSplineKVConfig(
                d=d, h_ratio=1.0, r=r, G=20))

    def forward(self, x):
        if self.use_kernel:
            from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
                flash_spline_feature,
            )
            d = self.cfg.d
            shape = x.shape
            x_flat = x.reshape(-1, d)
            z = self.K(x_flat)
            f = flash_spline_feature(
                z, self.C,
                grid_lo=float(self.cfg.grid_lo), grid_hi=float(self.cfg.grid_hi),
                G=int(self.cfg.G),
                activation=self.cfg.activation,
                lambda_scale=float(self.cfg.lambda_scale),
                use_kernel=True,
            )
            y = self.W_out(f)
            return y.reshape(shape)
        return self.ref(x)


def _params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _peak_mb_after(fn: Callable[[], None]) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def _median_ms(fn: Callable[[], None], iters: int = 30, warmup: int = 8) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def bench_module(name: str, builder, *,
                 B: int, T: int, d: int,
                 device: torch.device, dtype: torch.dtype) -> dict:
    torch.cuda.empty_cache(); gc.collect()
    torch.manual_seed(0)

    m = builder().to(device=device, dtype=dtype).train()
    n_params = _params(m)

    x_const = torch.randn(B, T, d, device=device, dtype=dtype)
    target  = torch.randn(B, T, d, device=device, dtype=dtype)

    def step_fwd():
        with torch.no_grad():
            y = m(x_const)
        return y

    def step_fwd_with_grad():
        x = x_const.detach().requires_grad_(True)
        y = m(x)
        return y

    def step_bwd():
        x = x_const.detach().requires_grad_(True)
        y = m(x)
        loss = (y - target).pow(2).sum()
        loss.backward()
        m.zero_grad(set_to_none=True)
        return loss

    # forward only (no grad context for fairness)
    ms_fwd = _median_ms(step_fwd)

    # full training step (fwd + bwd + zero_grad)
    ms_total = _median_ms(step_bwd)

    # Peak memory during training step (after warmup so autotune cache is settled)
    for _ in range(5):  # warmup to settle autotune
        step_bwd()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        step_bwd()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    alloc_mb = torch.cuda.memory_allocated() / 1024**2

    del m, x_const, target
    torch.cuda.empty_cache(); gc.collect()

    return {
        "name":    name,
        "params":  n_params,
        "params_M": n_params / 1e6,
        "ms_fwd":  ms_fwd,
        "ms_total": ms_total,
        "ms_bwd":  ms_total - ms_fwd,
        "peak_mb": peak_mb,
        "alloc_mb": alloc_mb,
    }


def render_table(rows: list[dict], baseline_name: str = "mlp_h_4d") -> str:
    base = next((r for r in rows if r["name"] == baseline_name), None)
    if base is None:
        base = rows[0]

    lines = [
        f"{'name':<18} {'params(M)':>10} {'ms_fwd':>9} {'ms_bwd':>9} "
        f"{'ms_tot':>9} {'peak_MB':>9} {'p/base':>7} {'fwd/base':>9} "
        f"{'tot/base':>9} {'mem/base':>9}",
        "-" * 120,
    ]
    for r in rows:
        ratio_p = r["params"] / max(1, base["params"])
        ratio_f = r["ms_fwd"] / max(1e-9, base["ms_fwd"])
        ratio_t = r["ms_total"] / max(1e-9, base["ms_total"])
        ratio_m = r["peak_mb"] / max(1e-3, base["peak_mb"])
        lines.append(
            f"{r['name']:<18} {r['params_M']:>10.2f} "
            f"{r['ms_fwd']:>9.3f} {r['ms_bwd']:>9.3f} {r['ms_total']:>9.3f} "
            f"{r['peak_mb']:>9.1f} {ratio_p:>6.2f}x {ratio_f:>8.2f}x "
            f"{ratio_t:>8.2f}x {ratio_m:>8.2f}x"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="if set, only bench these names")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; abort.")
        return
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    builders = _make_modules(args.d)
    if args.variants:
        builders = {k: v for k, v in builders.items() if k in args.variants}

    print(f"\n=== Full-FFN comparison ===")
    print(f"  d={args.d}  B={args.B}  T={args.T}  dtype={args.dtype}  device={device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    rows = []
    for name, builder in builders.items():
        print(f"  benching {name} ...", flush=True)
        try:
            r = bench_module(name, builder, B=args.B, T=args.T, d=args.d,
                              device=device, dtype=dtype)
            rows.append(r)
        except Exception as e:
            print(f"    [skip] {name}: {e}")

    print()
    print(render_table(rows))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\n[json written to {args.json_out}]")


if __name__ == "__main__":
    main()
