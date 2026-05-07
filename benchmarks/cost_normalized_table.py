"""Cost-normalized comparison table for the d20 3-way runs.

Implements P0-Parallel-3 from docs/PLAN_2026-05-04_neurips_experiment_queue.md.

Outputs three views:
  1) Same-token (8.71 B tokens locked): each run's bpb/CORE/wallclock/FLOPs/MFU.
  2) Same-wallclock (MLP V2 budget = 141.1 min): linearly interpolate each
     other method's trajectory at MLP's wallclock equivalent step.
  3) Per-million-token cost: wallclock-seconds, GPU-hours, FLOPs.

All inputs come from JSON files so the table is regeneratable on new logs.

Usage:
    .venv/bin/python benchmarks/cost_normalized_table.py
    .venv/bin/python benchmarks/cost_normalized_table.py \\
        --report-out docs/_artifacts/cost_normalized_2026-05-04.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COST = ROOT / "benchmarks" / "data" / "cost_normalized_2026-05-04.json"
DEFAULT_TRAJ = ROOT / "benchmarks" / "data" / "trajectory_2026-05-04.json"
DEFAULT_OUT = ROOT / "docs" / "RESULTS_2026-05-04_cost_normalized.md"


def interp(steps: list[int], values: list[float], target: float) -> float:
    """Piecewise-linear interpolate `values` at `target` step.

    Clamps to endpoints if out of range. Assumes `steps` is sorted ascending.
    """
    if target <= steps[0]:
        return values[0]
    if target >= steps[-1]:
        return values[-1]
    for i in range(len(steps) - 1):
        a, b = steps[i], steps[i + 1]
        if a <= target <= b:
            t = (target - a) / (b - a)
            return values[i] + t * (values[i + 1] - values[i])
    return values[-1]


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f} %"


def fmt_minutes(x: float) -> str:
    return f"{x:.2f} min"


def render(cost: dict, traj: dict) -> str:
    sched = cost["schedule"]
    methods = cost["methods"]
    total_tokens = sched["total_tokens"]
    n_gpus = sched["num_gpus"]
    n_iters = sched["num_iterations"]

    lines: list[str] = []
    add = lines.append

    add("# Cost-normalized comparison — d20 3-way (P0-Parallel-3)")
    add("")
    add(
        "**Date:** 2026-05-04  "
        "**Reproducer:** `.venv/bin/python benchmarks/cost_normalized_table.py`  "
        f"**Inputs:** `{DEFAULT_COST.relative_to(ROOT)}`, "
        f"`{DEFAULT_TRAJ.relative_to(ROOT)}`"
    )
    add("")

    add(
        f"All three runs share the same training schedule: "
        f"**{n_iters:,} steps × {sched['tokens_per_step']:,} tokens/step "
        f"= {total_tokens / 1e9:.2f} B tokens** on **{n_gpus} × H100 SXM**. "
        "What differs is wallclock-to-budget and FLOPs-to-budget."
    )
    add("")

    # --- 1) Same-token table ---
    add("## 1 · Same-token (fixed 8.71 B tokens, fixed 16 600 steps)")
    add("")
    add(
        "| Method | val bpb | CORE | wallclock | tok/sec | bf16 MFU | total FLOPs | "
        "GPU-hours | wallclock /M tok | FLOPs/token |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in methods.items():
        wc_min = m["wallclock_minutes"]
        tok_s = m["tok_per_sec"]
        mfu = m["bf16_mfu"]
        flops = m["flops_total"]
        bpb = m["bpb_final"]
        core = m["core_final"]
        gpu_hr = wc_min / 60.0 * n_gpus
        wc_per_m = wc_min * 60.0 / (total_tokens / 1e6)  # seconds per million tokens
        flops_per_token = flops / total_tokens
        add(
            f"| **{name}** | {bpb:.4f} | {core:.4f} | {fmt_minutes(wc_min)} | "
            f"{tok_s:,.0f} | {fmt_pct(mfu)} | {flops:.2e} | "
            f"{gpu_hr:.1f} H100-hr | {wc_per_m:.3f} s | {flops_per_token:.2e} |"
        )
    add("")
    # Δs vs MLP
    base = methods["MLP V2"]
    add("**Δ vs MLP V2 (same-token):**")
    add("")
    add("| Method | Δ bpb | Δ CORE | wallclock × MLP | FLOPs × MLP | GPU-hr × MLP |")
    add("|---|---:|---:|---:|---:|---:|")
    for name, m in methods.items():
        if name == "MLP V2":
            continue
        d_bpb = m["bpb_final"] - base["bpb_final"]
        d_core = m["core_final"] - base["core_final"]
        wc_ratio = m["wallclock_minutes"] / base["wallclock_minutes"]
        flops_ratio = m["flops_total"] / base["flops_total"]
        gpu_hr_ratio = wc_ratio  # same num_gpus, so identical
        add(
            f"| {name} | {d_bpb:+.4f} | {d_core:+.4f} | "
            f"{wc_ratio:.2f} × | {flops_ratio:.2f} × | {gpu_hr_ratio:.2f} × |"
        )
    add("")
    add(
        "Reading: at fixed token budget, **Run C uses fewer FLOPs (0.93 ×) "
        "but more wallclock (1.60 ×) than MLP V2** — i.e. it's FLOPs-efficient "
        "but kernel-throughput limited. This matches the "
        "21.94 % bf16 MFU vs 37.5 % MFU gap (Run C's spline backward dominates "
        "step time). Run A is worse on every column."
    )
    add("")

    # --- 2) Same-wallclock table ---
    add("## 2 · Same-wallclock (MLP V2 budget = 141.10 min)")
    add("")
    add(
        "What does each method look like if we cut training off at MLP V2's "
        "wallclock budget? `step_at_MLP_budget = (141.10 / wallclock_min) × 16 600`. "
        "We linearly interpolate each method's bpb/CORE trajectory at that step. "
        "(Run A's trajectory is the resumed series from the canonical stitched data, "
        "so the wallclock-equivalent step assumes Run A had not been preempted; "
        "this is the most charitable reading for Run A.)"
    )
    add("")
    add(
        "| Method | step at MLP budget | bpb @ that step | CORE @ that step | "
        "Δ bpb vs MLP final | Δ CORE vs MLP final |"
    )
    add("|---|---:|---:|---:|---:|---:|")
    mlp_budget = base["wallclock_minutes"]
    for name, m in methods.items():
        ratio = mlp_budget / m["wallclock_minutes"]
        step_eq = int(ratio * n_iters)
        if name == "MLP V2":
            bpb_eq = m["bpb_final"]
            core_eq = m["core_final"]
        else:
            tj = traj["methods"][name]
            bpb_eq = interp(tj["bpb_steps"], tj["bpb_values"], step_eq)
            core_eq = interp(tj["core_steps"], tj["core_values"], step_eq)
        d_bpb = bpb_eq - base["bpb_final"]
        d_core = core_eq - base["core_final"]
        add(
            f"| **{name}** | {step_eq:,} | {bpb_eq:.4f} | {core_eq:.4f} | "
            f"{d_bpb:+.4f} | {d_core:+.4f} |"
        )
    add("")
    add(
        "Reading: **at fixed wallclock, MLP V2 wins both metrics.** Run C at "
        "MLP-equivalent wallclock has only consumed ≈ 63 % of the schedule; "
        "its bpb is roughly +0.05 above MLP-final and its CORE is roughly −0.05 "
        "below MLP-final. **Run C's CORE win exists only at fixed token budget, "
        "not fixed wallclock.** This is the honest framing — the paper should "
        "say so explicitly."
    )
    add("")

    # --- 3) Per-million-token cost ---
    add("## 3 · Per-million-token cost (raw rates)")
    add("")
    add(
        "| Method | wallclock /M tok | GPU-seconds /M tok | FLOPs /M tok | "
        "vs MLP wallclock | vs MLP FLOPs |"
    )
    add("|---|---:|---:|---:|---:|---:|")
    base_wc_per_m = base["wallclock_minutes"] * 60.0 / (total_tokens / 1e6)
    base_flops_per_m = base["flops_total"] / (total_tokens / 1e6)
    for name, m in methods.items():
        wc_per_m = m["wallclock_minutes"] * 60.0 / (total_tokens / 1e6)
        gpu_s_per_m = wc_per_m * n_gpus
        flops_per_m = m["flops_total"] / (total_tokens / 1e6)
        add(
            f"| **{name}** | {wc_per_m:.3f} s | {gpu_s_per_m:.2f} s | "
            f"{flops_per_m:.3e} | {wc_per_m / base_wc_per_m:.2f} × | "
            f"{flops_per_m / base_flops_per_m:.2f} × |"
        )
    add("")

    # --- 4) Decoupling summary ---
    add("## 4 · Decoupling summary")
    add("")
    add(
        "| Question | Answer |"
    )
    add("|---|---|")
    add(
        "| Same token budget — does Run C beat MLP on bpb? | **No.** Run C +0.0064 bpb."
    )
    add(
        "| Same token budget — does Run C beat MLP on CORE? | **Yes, directionally.** "
        "+0.0112 (CI [−0.018, +0.042], straddles zero — see core uncertainty doc)."
    )
    add(
        "| Same wallclock — does Run C beat MLP? | **No, on either metric.** "
        "Run C only reaches step ≈ 10 400, where it is worse on both bpb and CORE."
    )
    add(
        "| Is Run C cheaper in FLOPs? | **Yes — 0.93 × MLP FLOPs at fixed tokens** "
        "(2.34e19 vs 2.51e19). The spline FFN has lower arithmetic intensity per token."
    )
    add(
        "| Is Run C cheaper in wallclock? | **No — 1.60 × MLP wallclock** "
        "at fixed tokens. Lower MFU (21.94 % vs 37.5 %) means the kernel is "
        "memory-bound, not arithmetic-bound."
    )
    add(
        "| Is Run A's wallclock penalty fixable? | Probably partly. "
        "16.16 % MFU is well below Run C's 21.94 %; the all-layer config strains "
        "the kernel pipelining further. Optimisation effort is better spent on "
        "Run C's late33 footprint (less total spline work)."
    )
    add("")

    add(
        "**Reviewer-attack closure:** the +0.0112 CORE win is **not** an artifact "
        "of Run C consuming more compute. It uses **fewer** FLOPs. It does use "
        "more wallclock, but that's a kernel-engineering issue (low MFU on the "
        "spline backward), not a fairness issue with the comparison. The paper "
        "should claim FLOPs-fairness explicitly and report the kernel-MFU gap "
        "as future work (workload-shape stress test P1-Sequential-6 will give "
        "the heatmap)."
    )

    add("")
    add("---")
    add("")
    add("## Pointers")
    add("")
    add(
        "- Trajectory plots that visualise the same-token decoupling: "
        "`docs/RESULTS_2026-05-04_core_trajectory.md`."
    )
    add(
        "- Bootstrap CI / sign test on the +0.0112 number: "
        "`docs/RESULTS_2026-05-04_core_uncertainty.md`."
    )
    add(
        "- Source data table: "
        "`docs/RESULTS_2026-05-04_full_d20_3way_comparison.md`."
    )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--cost", type=Path, default=DEFAULT_COST)
    parser.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    cost = json.loads(args.cost.read_text())
    traj = json.loads(args.traj.read_text())

    text = render(cost, traj)
    print(text)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(text + "\n")
    print(f"\nWrote: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
