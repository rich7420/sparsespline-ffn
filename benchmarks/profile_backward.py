"""Profile FullMix-Tucker form-B backward to identify the kernel target.

Stage 2's headline finding: form-B backward is ~109x slower than MLP at d=768
on RTX 3080 (bf16).  Before writing a kernel we want to know where the time
goes — fused full-stage kernel is a 1-2 week project, but a focused dQ
scatter-add kernel could be done in 2-3 days if dQ dominates.

What this script does:
  1. Build a FullMixTuckerFFN at production scale (d=768, R_o=R_i=96, R_b=16,
     G=20) on CUDA in bf16.
  2. Warm up + run 50 forward+backward iterations under torch.profiler.
  3. Aggregate ATEN-level CUDA self-time, sorted desc.  This shows which
     primitive ops own the backward cost.
  4. Bucket ops to the 5 stages (mixer A / B1 gather / V contract / C core /
     U readout) using a heuristic based on op name + tensor shape.
  5. Repeat for MLPFFN as a control baseline.
  6. Print a side-by-side cost decomposition.

Outputs: a JSON with the raw aggregates and a markdown-style summary table.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from sparsespline_ffn import MLPFFN, FullMixTuckerConfig, FullMixTuckerFFN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


PROD_CFG = dict(d=768, m=768, R_o=96, R_i=96, R_b=16, G=20)
DEFAULT_B = 4
DEFAULT_T = 512  # 2048 tokens, similar to nanochat micro-batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aggregate_self_cuda_time_us(events) -> dict[str, dict]:
    """Group profiler events by op name; sum self CUDA time and call count."""
    agg: dict[str, dict] = defaultdict(lambda: {"self_us": 0.0, "calls": 0})
    for ev in events:
        # Skip CPU-only events with no CUDA self time
        if ev.self_device_time_total <= 0:
            continue
        key = ev.key  # e.g., "aten::scatter_add_"
        agg[key]["self_us"] += ev.self_device_time_total
        agg[key]["calls"] += int(ev.count)
    return dict(agg)


def _bucket_op(name: str) -> str:
    """Heuristic bucket assignment by op name.

    Lookups / gather             -> 'B1_lookup'
    Scatter / index_put_         -> 'B1_lookup_bwd'
    addmm / mm                   -> '*_matmul' (mixer A or U readout)
    einsum / bmm                 -> 'V_or_C_contract'
    other                        -> 'other'
    """
    n = name.lower()
    if any(k in n for k in ("scatter", "index_put", "_index_put_impl",
                             "index_select_backward", "embedding_dense_backward")):
        return "B1_lookup_bwd"
    if any(k in n for k in ("index", "gather", "lerp")):
        return "B1_lookup_fwd"
    if any(k in n for k in ("bmm", "einsum")):
        return "V_or_C_contract"
    if any(k in n for k in ("addmm", "mm", "linear")):
        return "matmul"
    if any(k in n for k in ("add", "mul", "div", "to_copy", "_to_copy",
                             "transpose", "view", "contiguous", "reshape", "clone")):
        return "elementwise/copy"
    if "backward" in n:
        return "other_bwd"
    return "other"


def _profile_module(
    name: str, module: torch.nn.Module, x_factory, *,
    device: torch.device, warmup: int, iters: int, do_backward: bool,
) -> dict:
    module.train(do_backward)
    # Warmup
    for _ in range(warmup):
        x = x_factory()
        if do_backward:
            x.requires_grad_(True)
        y = module(x)
        if do_backward:
            y.pow(2).sum().backward()
            module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(iters):
            x = x_factory()
            if do_backward:
                x.requires_grad_(True)
            y = module(x)
            if do_backward:
                y.pow(2).sum().backward()
                module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    events = list(prof.key_averages())
    agg = _aggregate_self_cuda_time_us(events)
    total = sum(v["self_us"] for v in agg.values())

    # Per-bucket aggregation
    buckets: dict[str, float] = defaultdict(float)
    for op_name, v in agg.items():
        buckets[_bucket_op(op_name)] += v["self_us"]

    return {
        "name": name,
        "iters": iters,
        "total_self_cuda_us": total,
        "ms_per_iter": total / iters / 1000.0,
        "ops": agg,
        "buckets": dict(buckets),
    }


def _print_top_ops(result: dict, top_k: int = 15) -> None:
    print(f"\n  Top {top_k} ops by self CUDA time ({result['name']}):")
    print(f"    {'op':<48} {'self(ms)':>10} {'%':>6} {'calls':>8} {'us/call':>10}")
    ranked = sorted(result["ops"].items(), key=lambda kv: -kv[1]["self_us"])[:top_k]
    total = result["total_self_cuda_us"]
    for op_name, v in ranked:
        share = 100 * v["self_us"] / max(1.0, total)
        per_call = v["self_us"] / max(1, v["calls"])
        print(f"    {op_name:<48} {v['self_us']/1000:>10.2f} {share:>5.1f}% "
              f"{v['calls']:>8d} {per_call:>10.1f}")


def _print_bucket_table(fm_res: dict, mlp_res: dict | None) -> None:
    """Compare bucket totals between FullMix and MLP."""
    fm_total = fm_res["total_self_cuda_us"]
    mlp_total = mlp_res["total_self_cuda_us"] if mlp_res else None
    keys = sorted(set(fm_res["buckets"].keys()) | set(mlp_res["buckets"].keys() if mlp_res else []))

    print("\n  Bucket cost decomposition:")
    print(f"    {'bucket':<22} {'FM(ms)':>10} {'FM%':>6} "
          + (f"{'MLP(ms)':>10} {'MLP%':>6}" if mlp_res else "")
          + f" {'FM/iter':>10}")

    for k in keys:
        fm_us = fm_res["buckets"].get(k, 0.0)
        fm_pct = 100 * fm_us / max(1.0, fm_total)
        per_iter_ms = fm_us / fm_res["iters"] / 1000.0
        if mlp_res:
            mlp_us = mlp_res["buckets"].get(k, 0.0)
            mlp_pct = 100 * mlp_us / max(1.0, mlp_total)
            print(f"    {k:<22} {fm_us/1000:>10.2f} {fm_pct:>5.1f}% "
                  f"{mlp_us/1000:>10.2f} {mlp_pct:>5.1f}% {per_iter_ms:>10.3f}")
        else:
            print(f"    {k:<22} {fm_us/1000:>10.2f} {fm_pct:>5.1f}% "
                  f"{per_iter_ms:>10.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--T", type=int, default=DEFAULT_T)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--use-kernel", action="store_true",
                    help="route stage 2 through the Triton B1Lookup kernel")
    ap.add_argument("--out-json", default=None,
                    help="dump aggregates to this path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required for this profile; aborting.")
        return 1
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print("=" * 78)
    print("Backward profile: FullMix-Tucker (form B) vs MLPFFN")
    print(f"  device={device}, dtype={dtype}")
    print(f"  config: {PROD_CFG}")
    print(f"  shape : B={args.B}, T={args.T} (N={args.B*args.T} tokens)")
    print(f"  warmup={args.warmup} + iters={args.iters}")
    print("=" * 78)

    cfg = FullMixTuckerConfig(**PROD_CFG, use_kernel=args.use_kernel)
    fm = FullMixTuckerFFN(cfg).to(device=device, dtype=dtype)
    if args.use_kernel:
        print("  use_kernel=TRUE -- routing stage 2 through Triton B1Lookup")
    mlp = MLPFFN(d=PROD_CFG["d"], mlp_ratio=4).to(device=device, dtype=dtype)

    def x_factory() -> torch.Tensor:
        return torch.randn(args.B, args.T, PROD_CFG["d"], device=device, dtype=dtype)

    # FWD-only (for context: how much of the cost is bwd?)
    print("\n>>> FullMix-Tucker fwd-only profile")
    fm_fwd = _profile_module("FullMix_fwd", fm, x_factory,
                             device=device, warmup=args.warmup,
                             iters=args.iters, do_backward=False)
    print(f"  ms/iter (CUDA self-total): {fm_fwd['ms_per_iter']:.3f}")
    _print_top_ops(fm_fwd, top_k=10)

    print("\n>>> FullMix-Tucker fwd+bwd profile")
    fm_bwd = _profile_module("FullMix_fwd+bwd", fm, x_factory,
                             device=device, warmup=args.warmup,
                             iters=args.iters, do_backward=True)
    print(f"  ms/iter (CUDA self-total): {fm_bwd['ms_per_iter']:.3f}")
    _print_top_ops(fm_bwd, top_k=15)

    print("\n>>> MLPFFN fwd+bwd profile (control)")
    mlp_bwd = _profile_module("MLP_fwd+bwd", mlp, x_factory,
                              device=device, warmup=args.warmup,
                              iters=args.iters, do_backward=True)
    print(f"  ms/iter (CUDA self-total): {mlp_bwd['ms_per_iter']:.3f}")
    _print_top_ops(mlp_bwd, top_k=10)

    _print_bucket_table(fm_bwd, mlp_bwd)

    print("\nFwd-only vs fwd+bwd breakdown (FullMix):")
    fwd_ms = fm_fwd["ms_per_iter"]
    full_ms = fm_bwd["ms_per_iter"]
    bwd_ms = full_ms - fwd_ms
    print(f"  fwd     : {fwd_ms:7.3f} ms/iter  ({100*fwd_ms/max(1e-9, full_ms):.1f}%)")
    print(f"  bwd     : {bwd_ms:7.3f} ms/iter  ({100*bwd_ms/max(1e-9, full_ms):.1f}%)")
    print(f"  fwd+bwd : {full_ms:7.3f} ms/iter")
    print(f"\n  vs MLP fwd+bwd ({mlp_bwd['ms_per_iter']:.3f} ms): "
          f"{full_ms/max(1e-9, mlp_bwd['ms_per_iter']):.1f}x slower")

    # Identify the kernel target
    print("\nKernel-scope verdict:")
    fm_buckets = fm_bwd["buckets"]
    fm_total = sum(fm_buckets.values())
    sorted_b = sorted(fm_buckets.items(), key=lambda kv: -kv[1])
    top_b, top_us = sorted_b[0]
    top_pct = 100 * top_us / max(1.0, fm_total)
    second_b, second_us = sorted_b[1] if len(sorted_b) > 1 else ("", 0.0)
    second_pct = 100 * second_us / max(1.0, fm_total)
    print(f"  dominant bucket : {top_b} ({top_pct:.1f}% of fwd+bwd)")
    print(f"  next            : {second_b} ({second_pct:.1f}%)")
    if top_pct > 60:
        print(f"  >>> RECOMMEND: focused kernel on '{top_b}' alone "
              f"(MVP: 2-3 day project)")
    elif top_pct + second_pct > 75:
        print(f"  >>> RECOMMEND: kernel covers '{top_b}' + '{second_b}'")
    else:
        print("  >>> RECOMMEND: full 5-stage fused kernel (cost spread across stages)")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "config": PROD_CFG, "B": args.B, "T": args.T, "dtype": args.dtype,
            "fm_fwd": fm_fwd, "fm_bwd": fm_bwd, "mlp_bwd": mlp_bwd,
        }, indent=2))
        print(f"\n  wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
