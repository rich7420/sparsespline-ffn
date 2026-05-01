"""Benchmark: V+C fusion (A) vs CUDA graphs (B) vs baseline.

Five configs measured at production-shape (d=768, R_o=R_i=96, R_b=16, G=20)
on whatever GPU is present (3080, H100, ...):

  1. mlp                       MLPFFN (mlp_ratio=4) — baseline
  2. fm_formB                  FullMix-Tucker, no kernel, no fusion (slow path)
  3. fm_kernel                 + B1Lookup Triton kernel (current best, "Tier 3")
  4. fm_kernel_fusedVC         + Approach A (use_fused_vc=True)
  5. fm_kernel_cudagraph       + Approach B (CUDA graph capture/replay)
  6. fm_kernel_fusedVC_graph   + A AND B combined

Each config: warmup, then 100 fwd+bwd iters, report median ms + peak VRAM.

Output: prints a markdown table.  Optionally dumps JSON for cross-GPU
comparison.

Usage:
    python benchmarks/v_c_fusion_bench.py
    python benchmarks/v_c_fusion_bench.py --B 4 --T 1024 --iters 200
    python benchmarks/v_c_fusion_bench.py --out-json /tmp/bench.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN, MLPFFN
from sparsespline_ffn.simple_spline_mlp import SimpleSplineConfig, SimpleSplineMLP


# ---------------------------------------------------------------------------
# CUDA-graphs wrapper: capture once, replay on each call (static shape only).
# ---------------------------------------------------------------------------


class CudaGraphFFN:
    """Wraps any nn.Module(x: (B,T,d)) -> (B,T,d) for graph capture.

    Captures one fwd+bwd of ``ref_ffn`` against a static input/grad buffer.
    Subsequent ``step()`` calls copy fresh input into the static buffer and
    replay the graph -- no Python-side autograd, no kernel-launch overhead.

    Static contract: same B, T, d, dtype across all calls.
    """

    def __init__(self, ref_ffn: torch.nn.Module, B: int, T: int, d: int,
                 dtype: torch.dtype, device: torch.device,
                 warmup_iters: int = 5) -> None:
        self.ffn = ref_ffn.to(device=device, dtype=dtype).train()
        self.device = device
        self.B, self.T, self.d = B, T, d

        # Static buffers
        self.static_x = torch.randn(B, T, d, device=device, dtype=dtype,
                                     requires_grad=True)
        # Loss target (we time MSE-like loss the same way the comparison
        # bench does to match per-iter work)
        self._setup(warmup_iters)

    def _setup(self, warmup_iters: int) -> None:
        # Warmup must run on a separate stream prior to capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                self._forward_backward(self.static_x)
                self.ffn.zero_grad(set_to_none=False)
                if self.static_x.grad is not None:
                    self.static_x.grad.zero_()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        self.ffn.zero_grad(set_to_none=False)
        if self.static_x.grad is not None:
            self.static_x.grad.zero_()
        with torch.cuda.graph(self.graph):
            self.static_loss = self._forward_backward(self.static_x)

    def _forward_backward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ffn(x)
        loss = y.pow(2).sum()
        loss.backward()
        return loss

    def step(self, x_new: torch.Tensor) -> None:
        """Replay one fwd+bwd with a fresh input.  Caller still has access
        to gradients via self.ffn.parameters() and self.static_x.grad."""
        # Reset gradient buffers BEFORE replay.
        self.ffn.zero_grad(set_to_none=False)
        if self.static_x.grad is not None:
            self.static_x.grad.zero_()
        # NB: use .data.copy_() (not .copy_()) to avoid the autograd
        # "in-place op on leaf with requires_grad" error.  We are
        # deliberately mutating the *storage* of the static input.
        with torch.no_grad():
            self.static_x.data.copy_(x_new)
        self.graph.replay()


# ---------------------------------------------------------------------------
# Bench helpers
# ---------------------------------------------------------------------------


def time_ffn(name: str, ffn: torch.nn.Module, B: int, T: int, d: int,
             dtype: torch.dtype, device: torch.device,
             warmup: int, iters: int) -> dict:
    """Eager fwd+bwd timing."""
    ffn = ffn.to(device=device, dtype=dtype).train()
    xf = lambda: torch.randn(B, T, d, device=device, dtype=dtype,
                              requires_grad=True)  # noqa: E731

    for _ in range(warmup):
        x = xf()
        loss = ffn(x).pow(2).sum()
        loss.backward()
        ffn.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        x = xf()
        loss = ffn(x).pow(2).sum()
        loss.backward()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ffn.zero_grad(set_to_none=True)

    samples.sort()
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
               if device.type == "cuda" else float("nan"))
    return {
        "name": name,
        "median_ms": samples[len(samples) // 2],
        "min_ms": min(samples),
        "p10_ms": samples[max(0, int(0.1 * len(samples)))],
        "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        "peak_mb": peak_mb,
    }


def time_cudagraph_ffn(name: str, ffn: torch.nn.Module, B: int, T: int, d: int,
                        dtype: torch.dtype, device: torch.device,
                        warmup: int, iters: int) -> dict:
    """CUDA-graph captured fwd+bwd timing."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    g = CudaGraphFFN(ffn, B, T, d, dtype, device, warmup_iters=warmup)
    fresh = lambda: torch.randn(B, T, d, device=device, dtype=dtype)  # noqa: E731

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        x = fresh()
        g.step(x)
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1000)

    samples.sort()
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
               if device.type == "cuda" else float("nan"))
    return {
        "name": name,
        "median_ms": samples[len(samples) // 2],
        "min_ms": min(samples),
        "p10_ms": samples[max(0, int(0.1 * len(samples)))],
        "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        "peak_mb": peak_mb,
    }


def _bench_wider_with_AB(
    name: str, B: int, T: int, d: int, dtype: torch.dtype,
    device: torch.device, warmup: int, iters: int,
    *, R_o: int, R_i: int, R_b: int,
) -> dict:
    """Time a wider-bandwidth FullMix-Tucker via the AB path (CUDA graph +
    V+C fusion).  Falls back to eager if graph capture fails."""
    fm = make_fm(d, use_kernel=True, use_fused_vc=True,
                 R_o=R_o, R_i=R_i, R_b=R_b)
    try:
        return time_cudagraph_ffn(
            name, fm, B, T, d, dtype, device, warmup, iters,
        )
    except Exception as e:
        print(f"  [warn] graph capture failed for {name}: {e}; eager fallback")
        # Re-init since graph capture may have left state.
        fm = make_fm(d, use_kernel=True, use_fused_vc=True,
                     R_o=R_o, R_i=R_i, R_b=R_b)
        return time_ffn(name, fm, B, T, d, dtype, device, warmup, iters)


def _bench_fp8_forward_path(
    name: str, B: int, T: int, d: int, dtype: torch.dtype,
    device: torch.device, warmup: int, iters: int,
) -> dict:
    """Forward-only timing of a 'what-if' fp8 path.

    We DON'T ship fp8 in production yet -- this measures the speed ceiling
    if we replaced V/C/U bf16 GEMMs with ``torch._scaled_mm`` fp8 GEMMs.

    Per-tensor scale uses a static amax (calibrated on a single warmup batch),
    not a dynamic accumulator -- realistic for inference, lower bound for
    training (training would have small additional amax-update overhead).
    """
    fm = make_fm(d, use_kernel=True, use_fused_vc=False).to(device=device, dtype=dtype)
    fm.eval()

    # Precompute scale factors from a warmup pass (per-tensor amax).
    F8 = torch.float8_e4m3fn
    F8_MAX = 448.0  # e4m3 max representable absolute value

    # Warmup x to calibrate scales.
    x_calib = torch.randn(B, T, d, device=device, dtype=dtype)

    @torch.no_grad()
    def _calibrate_scales():
        # Run normal fwd to capture activations going into V, C, U matmuls.
        captures: dict[str, torch.Tensor] = {}
        # Patch the einsums to capture inputs
        orig_einsum = torch.einsum
        nm_calls = [0]
        original_einsum_args = []

        def patched(spec, *tensors):
            r = orig_einsum(spec, *tensors)
            if spec == "nmc, mb -> nbc":
                captures["beta_for_V"] = tensors[0].clone()
            elif spec == "nbc, abc -> na":
                captures["xi_for_C"] = tensors[0].clone()
            return r

        torch.einsum = patched
        try:
            y = fm(x_calib)
            captures["eta_for_U"] = (
                # Re-derive eta from captured xi via a forward eval.
                torch.einsum("nbc, abc -> na", captures["xi_for_C"], fm.C)
            )
        finally:
            torch.einsum = orig_einsum

        scales = {}
        for k, t_ in captures.items():
            amax = t_.float().abs().max()
            scales[k] = (amax / F8_MAX).clamp(min=1e-6)
        # And per-weight scales
        for name_w in ("V", "C", "U"):
            w = getattr(fm, name_w)
            amax = w.float().abs().max()
            scales[f"{name_w}_w"] = (amax / F8_MAX).clamp(min=1e-6)
        return scales

    scales = _calibrate_scales()

    # Pre-quantize the param weights to fp8 + their scales.
    s_V_w = scales["V_w"]
    s_U_w = scales["U_w"]
    V_fp8 = (fm.V.float() / s_V_w).clamp(-F8_MAX, F8_MAX).to(F8)
    # _scaled_mm requires the second arg to be column-major
    V_fp8_t = V_fp8.t().contiguous().t()  # ensure layout

    # For C, fp8 contraction over (R_i, R_b) is awkward (3D); fall back to
    # bf16 for C contraction in this fwd-only what-if and only fp8 the
    # V and U matmuls.  This is the realistic shape: tiny C is launch-bound,
    # not compute-bound, so fp8'ing it doesn't save time anyway.

    def fp8_forward(x: torch.Tensor) -> torch.Tensor:
        # Stage 1: mixer A (kept bf16 for now)
        z = fm.A(x.reshape(-1, d)) if fm.A is not None else x.reshape(-1, d)

        # Stage 2: B1 lookup (Triton kernel; no fp8 here)
        bin_idx, t_e = fm._bin_and_frac(z)
        from sparsespline_ffn.kernels import B1Lookup
        beta = B1Lookup.apply(fm.Q, bin_idx, t_e)

        # Stage 3 — fp8 V contraction.
        # beta: (N, m, R_b)  V: (m, R_i)
        # Need to call _scaled_mm with 2D tensors.  Reshape:
        #   (N*R_b, m) @ (m, R_i)  -> (N*R_b, R_i)  then reshape -> (N, R_b, R_i) -> permute (N, R_i, R_b)
        N = beta.shape[0]
        beta_2d = beta.permute(0, 2, 1).reshape(N * fm.cfg.R_b, d)
        # Compute fresh activation scale from this batch (calibrated baseline)
        s_beta = scales["beta_for_V"]
        beta_fp8 = (beta_2d.float() / s_beta).clamp(-F8_MAX, F8_MAX).to(F8)
        # _scaled_mm: out = (a*scale_a) @ (b*scale_b), needs b col-major.
        xi_2d = torch._scaled_mm(
            beta_fp8, V_fp8_t,
            scale_a=s_beta.reshape(1, 1).float(),
            scale_b=s_V_w.reshape(1, 1).float(),
            out_dtype=dtype,
        )
        xi = xi_2d.view(N, fm.cfg.R_b, fm.cfg.R_i).permute(0, 2, 1).contiguous()

        # Stage 4: C core contraction (kept bf16; tiny tensor)
        eta = torch.einsum("nbc, abc -> na", xi, fm.C)

        # Stage 5 — fp8 U readout: eta @ U.T -> (N, d).
        # eta is (N, R_o), U is (d, R_o).  For (N, R_o) @ (R_o, d) we need
        # b = U.T which is (R_o, d).  _scaled_mm wants b in col-major; pass
        # U directly (its strides expose (d, R_o) row-major == (R_o, d)
        # col-major when reinterpreted), via a contiguous transpose.
        s_eta = scales["eta_for_U"]
        eta_fp8 = (eta.float() / s_eta).clamp(-F8_MAX, F8_MAX).to(F8)
        # U is (d, R_o); we want shape (R_o, d) col-major.  ``.t()`` returns
        # a view with swapped strides, satisfying _scaled_mm's col-major req.
        U_fp8 = (fm.U.float() / s_U_w).clamp(-F8_MAX, F8_MAX).to(F8)  # (d, R_o)
        U_fp8_b = U_fp8.t()  # view (R_o, d) col-major
        y_2d = torch._scaled_mm(
            eta_fp8, U_fp8_b,
            scale_a=s_eta.reshape(1, 1).float(),
            scale_b=s_U_w.reshape(1, 1).float(),
            out_dtype=dtype,
        )
        return (y_2d * fm.gamma).reshape(*x.shape[:-1], -1)

    # Warmup
    xf = lambda: torch.randn(B, T, d, device=device, dtype=dtype)  # noqa: E731
    for _ in range(warmup):
        with torch.no_grad():
            _ = fp8_forward(xf())
    torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    samples = []
    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = fp8_forward(xf())
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - t0) * 1000)

    samples.sort()
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    return {
        "name": name,
        "median_ms": samples[len(samples) // 2],
        "min_ms": min(samples),
        "p10_ms": samples[max(0, int(0.1 * len(samples)))],
        "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        "peak_mb": peak_mb,
        "note": "fwd-only (no backward); compare to fwd-portion of other rows",
    }


def make_fm(d: int, *, use_kernel: bool, use_fused_vc: bool,
            use_mixer: bool = True,
            R_o: int = 96, R_i: int = 96, R_b: int = 16) -> FullMixTuckerFFN:
    cfg = FullMixTuckerConfig(
        d=d, m=d, R_o=R_o, R_i=R_i, R_b=R_b, G=20,
        use_kernel=use_kernel, use_fused_vc=use_fused_vc, use_mixer=use_mixer,
    )
    torch.manual_seed(0)
    return FullMixTuckerFFN(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for this bench.")
        return 1
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    gpu_name = torch.cuda.get_device_name(device)
    print("=" * 78)
    print(f"V+C fusion / CUDA graphs benchmark")
    print(f"  GPU         : {gpu_name}")
    print(f"  shape       : d={args.d}, B={args.B}, T={args.T} "
          f"(N={args.B*args.T} tokens)")
    print(f"  dtype       : {dtype}")
    print(f"  warmup={args.warmup} + iters={args.iters}")
    print("=" * 78)

    rows = []

    # 1. MLP baseline
    rows.append(time_ffn(
        "MLP_baseline",
        MLPFFN(d=args.d, mlp_ratio=4),
        args.B, args.T, args.d, dtype, device, args.warmup, args.iters,
    ))

    # 2. form-B (no kernel, no fusion) -- the slow reference
    rows.append(time_ffn(
        "fm_formB",
        make_fm(args.d, use_kernel=False, use_fused_vc=False),
        args.B, args.T, args.d, dtype, device, args.warmup, args.iters,
    ))

    # 3. form-B + Triton kernel (current Tier 3)
    rows.append(time_ffn(
        "fm_kernel",
        make_fm(args.d, use_kernel=True, use_fused_vc=False),
        args.B, args.T, args.d, dtype, device, args.warmup, args.iters,
    ))

    # 4. + Approach A (V+C fused)
    rows.append(time_ffn(
        "fm_kernel_A",
        make_fm(args.d, use_kernel=True, use_fused_vc=True),
        args.B, args.T, args.d, dtype, device, args.warmup, args.iters,
    ))

    # 5. + Approach B (CUDA graphs over kernel-only path)
    try:
        rows.append(time_cudagraph_ffn(
            "fm_kernel_B",
            make_fm(args.d, use_kernel=True, use_fused_vc=False),
            args.B, args.T, args.d, dtype, device, args.warmup, args.iters,
        ))
    except Exception as e:
        print(f"  [warn] CUDA graph (kernel only) failed: {e}")
        rows.append({"name": "fm_kernel_B", "median_ms": float("nan"),
                     "min_ms": float("nan"), "p10_ms": float("nan"),
                     "p90_ms": float("nan"), "peak_mb": float("nan"),
                     "error": str(e)})

    # 6. + A AND B combined
    try:
        rows.append(time_cudagraph_ffn(
            "fm_kernel_AB",
            make_fm(args.d, use_kernel=True, use_fused_vc=True),
            args.B, args.T, args.d, dtype, device, args.warmup, args.iters,
        ))
    except Exception as e:
        print(f"  [warn] CUDA graph (kernel + V+C fusion) failed: {e}")
        rows.append({"name": "fm_kernel_AB", "median_ms": float("nan"),
                     "min_ms": float("nan"), "p10_ms": float("nan"),
                     "p90_ms": float("nan"), "peak_mb": float("nan"),
                     "error": str(e)})

    # 7. + use_mixer=False (drop mixer A) — saves one matmul + ~5MB params
    rows.append(time_ffn(
        "fm_kernel_A_noMixer",
        make_fm(args.d, use_kernel=True, use_fused_vc=True, use_mixer=False),
        args.B, args.T, args.d, dtype, device, args.warmup, args.iters,
    ))

    # 8. AB combined + use_mixer=False (the "everything for speed" config)
    try:
        rows.append(time_cudagraph_ffn(
            "fm_kernel_AB_noMixer",
            make_fm(args.d, use_kernel=True, use_fused_vc=True, use_mixer=False),
            args.B, args.T, args.d, dtype, device, args.warmup, args.iters,
        ))
    except Exception as e:
        print(f"  [warn] CUDA graph (no mixer) failed: {e}")
        rows.append({"name": "fm_kernel_AB_noMixer", "median_ms": float("nan"),
                     "min_ms": float("nan"), "p10_ms": float("nan"),
                     "p90_ms": float("nan"), "peak_mb": float("nan"),
                     "error": str(e)})

    # 9-11. Wider-bandwidth configs to test quality-vs-cost trade-off.
    # Following the "three bandwidth axes" diagnosis: current pa6 is
    # narrow on output (R_o=96 = d/8) and hidden (R_i*R_b=1536 = d/2);
    # MLP has output=d=768 and hidden=4d=3072.
    bandwidth_configs = [
        ("wide_output", dict(R_o=256, R_i=96,  R_b=16)),  # bigger output, cheap
        ("wide_hidden", dict(R_o=96,  R_i=192, R_b=32)),  # bigger hidden, doubles beta
        ("wide_all",    dict(R_o=256, R_i=192, R_b=24)),  # both
    ]
    for name, knobs in bandwidth_configs:
        rows.append(_bench_wider_with_AB(
            f"fm_AB_{name}",
            args.B, args.T, args.d, dtype, device, args.warmup, args.iters,
            **knobs,
        ))

    # 12-13. SimpleSpline-MLP — radical simplification.  Per-channel B2
    # spline activation replaces relu², h=d/2 instead of 4d.  Predicted
    # to win all three (speed/VRAM/quality) by leaning on B2 expressivity
    # rather than rank-bounded Tucker decomposition.
    for h_ratio_label, h_ratio in [("h_d_half", 0.5), ("h_d_full", 1.0)]:
        ss_cfg = SimpleSplineConfig(d=args.d, h_ratio=h_ratio, G=20,
                                     use_kernel=True)
        ss = SimpleSplineMLP(ss_cfg)
        # Avoid zero-init W_d (default) so we exercise real backward path
        with torch.no_grad():
            ss.W_d.weight.normal_(0.0, (3.0 / args.d) ** 0.5 * 0.4)
        rows.append(time_ffn(
            f"SimpleSpline_{h_ratio_label}",
            ss, args.B, args.T, args.d, dtype, device,
            args.warmup, args.iters,
        ))

    # 14. + CUDA graphs over SimpleSpline (the production path)
    for h_ratio_label, h_ratio in [("h_d_half", 0.5)]:
        ss_cfg = SimpleSplineConfig(d=args.d, h_ratio=h_ratio, G=20,
                                     use_kernel=True)
        ss = SimpleSplineMLP(ss_cfg)
        with torch.no_grad():
            ss.W_d.weight.normal_(0.0, (3.0 / args.d) ** 0.5 * 0.4)
        try:
            rows.append(time_cudagraph_ffn(
                f"SimpleSpline_{h_ratio_label}_B",
                ss, args.B, args.T, args.d, dtype, device,
                args.warmup, args.iters,
            ))
        except Exception as e:
            rows.append({"name": f"SimpleSpline_{h_ratio_label}_B",
                         "median_ms": float("nan"),
                         "min_ms": float("nan"), "p10_ms": float("nan"),
                         "p90_ms": float("nan"), "peak_mb": float("nan"),
                         "error": str(e)})

    # 15. fp8 V/C/U matmul (Hopper sm90+ only) — forward-only measurement.
    # We don't ship the fp8 path in production yet; this is a "what-if"
    # bench to see how much fp8 V/C/U could save.
    if torch.cuda.get_device_capability()[0] >= 9:
        try:
            rows.append(_bench_fp8_forward_path(
                "fm_kernel_fp8_fwd_only",
                args.B, args.T, args.d, dtype, device,
                args.warmup, args.iters,
            ))
        except Exception as e:
            print(f"  [warn] fp8 fwd bench failed: {e}")
            rows.append({"name": "fm_kernel_fp8_fwd_only",
                         "median_ms": float("nan"), "min_ms": float("nan"),
                         "p10_ms": float("nan"), "p90_ms": float("nan"),
                         "peak_mb": float("nan"), "error": str(e)})
    else:
        rows.append({"name": "fm_kernel_fp8_fwd_only",
                     "median_ms": float("nan"), "min_ms": float("nan"),
                     "p10_ms": float("nan"), "p90_ms": float("nan"),
                     "peak_mb": float("nan"),
                     "error": "fp8 requires Hopper sm90+; got "
                              f"sm{torch.cuda.get_device_capability()[0]}"
                              f"{torch.cuda.get_device_capability()[1]}"})

    # ---- Print ----
    base = next((r["median_ms"] for r in rows if r["name"] == "MLP_baseline"), 1.0)
    print(f"\n  {'config':<18} {'median(ms)':>11} {'min':>8} {'p10':>8} "
          f"{'p90':>8} {'peak(MB)':>10} {'vs MLP':>9}")
    print("  " + "-" * 78)
    for r in rows:
        m = r["median_ms"]
        ratio = m / base if (m == m and base > 0) else float("nan")
        print(
            f"  {r['name']:<18} {m:>11.3f} {r['min_ms']:>8.3f} "
            f"{r['p10_ms']:>8.3f} {r['p90_ms']:>8.3f} "
            f"{r['peak_mb']:>10.1f} {ratio:>8.2f}x"
        )

    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "gpu": gpu_name, "B": args.B, "T": args.T, "d": args.d,
            "dtype": str(dtype), "iters": args.iters, "rows": rows,
        }, indent=2))
        print(f"\n  wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
