"""d-scaling sweep: does FullMix win MLP at large d?

Theory: ``use_mixer=False`` removes the d² mixer term, leaving FullMix
params = O(d × (R_o + R_i)) instead of O(d²).  At d ≥ 1024+ this should
flip the speed and VRAM comparison vs MLP.

We bench five configs across d ∈ {768, 1024, 1536, 2048}:
  1. MLP_baseline                 (mlp_ratio=4, the canonical reference)
  2. fm_AB                        FullMix with kernel + V/C fused + CUDA graph
  3. fm_AB_noMixer                + use_mixer=False (the d-linear path)
  4. fm_AB_noMixer_wide_output    + R_o = d/3 (closes F.4.b)
  5. fm_AB_wide_output            mixer ON but R_o = d/3

The hypothesis: row 3 or 4 beats MLP on BOTH speed and VRAM at d ≥ 1536.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# Reuse the helpers from v_c_fusion_bench
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_c_fusion_bench import (  # noqa: E402
    CudaGraphFFN,
    make_fm,
    time_cudagraph_ffn,
    time_ffn,
)

from sparsespline_ffn import MLPFFN  # noqa: E402


def bench_one_d(
    d: int, B: int, T: int, dtype: torch.dtype, device: torch.device,
    warmup: int, iters: int,
) -> list[dict]:
    rows: list[dict] = []

    # 1. MLP baseline
    rows.append(time_ffn(
        f"MLP_d{d}",
        MLPFFN(d=d, mlp_ratio=4),
        B, T, d, dtype, device, warmup, iters,
    ))

    # 2. fm with kernel + A + B (full optimization, mixer ON, R_o=96 fixed)
    try:
        rows.append(time_cudagraph_ffn(
            f"fm_AB_d{d}",
            make_fm(d, use_kernel=True, use_fused_vc=True),
            B, T, d, dtype, device, warmup, iters,
        ))
    except Exception as e:
        rows.append({"name": f"fm_AB_d{d}", "error": str(e),
                     "median_ms": float("nan"), "peak_mb": float("nan")})

    # 3. fm AB + use_mixer=False (the d-linear-scaling test)
    try:
        rows.append(time_cudagraph_ffn(
            f"fm_AB_noMixer_d{d}",
            make_fm(d, use_kernel=True, use_fused_vc=True, use_mixer=False),
            B, T, d, dtype, device, warmup, iters,
        ))
    except Exception as e:
        rows.append({"name": f"fm_AB_noMixer_d{d}", "error": str(e),
                     "median_ms": float("nan"), "peak_mb": float("nan")})

    # 4. fm AB no-mixer + wide_output (R_o = d/3)
    R_o = max(96, d // 3)
    try:
        rows.append(time_cudagraph_ffn(
            f"fm_AB_noMixer_wideRo_d{d}_Ro{R_o}",
            make_fm(d, use_kernel=True, use_fused_vc=True,
                    use_mixer=False, R_o=R_o, R_i=96, R_b=16),
            B, T, d, dtype, device, warmup, iters,
        ))
    except Exception as e:
        rows.append({"name": f"fm_AB_noMixer_wideRo_d{d}", "error": str(e),
                     "median_ms": float("nan"), "peak_mb": float("nan")})

    # 5. fm AB with mixer + wide_output (control: mixer ON, R_o=d/3)
    try:
        rows.append(time_cudagraph_ffn(
            f"fm_AB_wideRo_d{d}_Ro{R_o}",
            make_fm(d, use_kernel=True, use_fused_vc=True,
                    use_mixer=True, R_o=R_o, R_i=96, R_b=16),
            B, T, d, dtype, device, warmup, iters,
        ))
    except Exception as e:
        rows.append({"name": f"fm_AB_wideRo_d{d}", "error": str(e),
                     "median_ms": float("nan"), "peak_mb": float("nan")})

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+",
                    default=[768, 1024, 1536, 2048])
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    gpu = torch.cuda.get_device_name(device)
    print("=" * 78)
    print(f"d-scaling sweep")
    print(f"  GPU={gpu}  dtype={dtype}  B={args.B} T={args.T}")
    print(f"  ds={args.ds}  warmup={args.warmup} iters={args.iters}")
    print("=" * 78)

    all_rows: list[dict] = []
    for d in args.ds:
        print(f"\n>>> d={d}")
        rows = bench_one_d(
            d, args.B, args.T, dtype, device,
            args.warmup, args.iters,
        )
        # Print table for this d
        mlp_med = next((r["median_ms"] for r in rows if r["name"].startswith("MLP_")), None)
        print(f"  {'config':<40} {'median(ms)':>11} {'peak(MB)':>10} {'vs MLP':>10}")
        for r in rows:
            m = r.get("median_ms", float("nan"))
            p = r.get("peak_mb", float("nan"))
            ratio = m / mlp_med if (mlp_med and m == m) else float("nan")
            note = f" [{r['error'][:40]}]" if "error" in r else ""
            print(f"  {r['name']:<40} {m:>11.3f} {p:>10.1f} {ratio:>9.2f}x{note}")
        all_rows.extend(rows)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "gpu": gpu, "B": args.B, "T": args.T, "dtype": str(dtype),
            "ds": args.ds, "iters": args.iters, "rows": all_rows,
        }, indent=2))
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
