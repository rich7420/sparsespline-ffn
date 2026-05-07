#!/usr/bin/env python3
"""Recompute val_loss scaling tables from training logs (no manual math).

For each training run, extracts the `[val] step N val_loss=X` trajectory
and reports three metrics:
  * final_eval        — last `[val]` line in the run
  * last_5_eval_mean  — arithmetic mean of the last 5 evals
  * best_so_far       — minimum val_loss observed across the run

Sources are matched explicitly so we can not confuse seeds / token budgets.
Output: markdown tables + a JSON dump for downstream automation.

Usage:
    python benchmarks/recompute_scaling_tables.py
    python benchmarks/recompute_scaling_tables.py --json out/scaling.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev


# Regex for the periodic eval line: `[val] step 50000  val_loss=4.9977  grad_finite=True`
VAL_RE = re.compile(r'\[val\] step (\d+)\s+val_loss=([\d.]+)')

# Regex for the JSON dump field: `"final_val_loss": X` (one-shot eval at end-of-train)
FINAL_RE = re.compile(r'"final_val_loss":\s*([\d.]+)')

# Regex for wall_s
WALL_RE = re.compile(r'"wall_s":\s*([\d.]+)')


def parse_log(path: Path) -> dict:
    """Pull val trajectory + final + wall from a training log file."""
    if not path.exists():
        return {"missing": True, "path": str(path)}
    text = path.read_text()
    traj = [(int(m.group(1)), float(m.group(2))) for m in VAL_RE.finditer(text)]
    final_dump_m = FINAL_RE.search(text)
    wall_m = WALL_RE.search(text)
    return {
        "path": str(path),
        "n_evals": len(traj),
        "trajectory": traj,
        "final_eval": traj[-1][1] if traj else None,
        "last_5_eval_mean": mean(v for _, v in traj[-5:]) if len(traj) >= 5 else None,
        "best_so_far": min(v for _, v in traj) if traj else None,
        "final_val_loss_dump": float(final_dump_m.group(1)) if final_dump_m else None,
        "wall_s": float(wall_m.group(1)) if wall_m else None,
    }


# Run registry — explicit mapping so seeds / sizes never get confused.
RUNS = {
    # 100M, seed 0 (existing — captured both as .output AND fallback /tmp logs)
    "MLP_100M_s0":       Path("/tmp/claude-1000/-home-anon-pal-kan/3e7acc8b-38f7-4a29-a12e-e03141de1bb3/tasks/b2epdddku.output"),
    "RLKV_100M_v11v5_s0": Path("/tmp/claude-1000/-home-anon-pal-kan/3e7acc8b-38f7-4a29-a12e-e03141de1bb3/tasks/bghsqfo3y.output"),
    "RLKV_100M_v1v1_s0":   Path("/tmp/claude-1000/-home-anon-pal-kan/3e7acc8b-38f7-4a29-a12e-e03141de1bb3/tasks/b8nj6dcdg.output"),
    "RLKV_100M_v11v1_s0":  Path("/tmp/claude-1000/-home-anon-pal-kan/3e7acc8b-38f7-4a29-a12e-e03141de1bb3/tasks/bxecn21zo.output"),
    # 100M, seed 1, 2 — redirected to /tmp because of the > /tmp/X.log mistake
    "MLP_100M_s1":         Path("/tmp/mlp_s1.log"),
    "MLP_100M_s2":         Path("/tmp/mlp_s2.log"),
    "RLKV_100M_v11v5_s1":  Path("/tmp/v11v5_s1.log"),
    "RLKV_100M_v11v5_s2":  Path("/tmp/v11v5_s2.log"),
    # 200M
    "MLP_200M_s0":         Path("/tmp/mlp_200M.log"),
    "RLKV_200M_v11v5_s0":  Path("/tmp/v11v5_200M.log"),
    # 800M
    "MLP_800M_s0":         Path("/tmp/claude-1000/-home-anon-pal-kan/3e7acc8b-38f7-4a29-a12e-e03141de1bb3/tasks/blvnu9tnz.output"),
    "RLKV_800M_v11v5_s0":  Path("/tmp/claude-1000/-home-anon-pal-kan/3e7acc8b-38f7-4a29-a12e-e03141de1bb3/tasks/b8588qbjt.output"),
}


def fmt(x: float | None, prec: int = 4) -> str:
    if x is None: return "N/A"
    return f"{x:.{prec}f}"


def render_run_metrics(parsed: dict[str, dict]) -> list[str]:
    """One row per run with the 3 metrics + final_dump for cross-check."""
    rows = []
    rows.append("| run | n_evals | final_eval | last_5_eval_mean | best_so_far | final_val_loss (dump) | wall (s) |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, info in parsed.items():
        if info.get("missing"):
            rows.append(f"| {name} | MISSING | — | — | — | — | — |")
            continue
        rows.append(f"| {name} | {info['n_evals']} | {fmt(info['final_eval'])} | "
                    f"{fmt(info['last_5_eval_mean'])} | {fmt(info['best_so_far'])} | "
                    f"{fmt(info['final_val_loss_dump'])} | "
                    f"{fmt(info['wall_s'], 1) if info['wall_s'] else '—'} |")
    return rows


def render_gap_table(parsed: dict[str, dict],
                       pairs: list[tuple[str, str, str]]) -> list[str]:
    """One row per (MLP, RL-KV) pair showing each metric and the resulting gap."""
    rows = []
    rows.append("| pair (MLP / RL-KV) | metric | MLP | RL-KV | gap (RL-KV − MLP) |")
    rows.append("|---|---|---:|---:|---:|")
    for label, mlp_key, rlkv_key in pairs:
        mlp = parsed.get(mlp_key, {})
        rlkv = parsed.get(rlkv_key, {})
        if mlp.get("missing") or rlkv.get("missing"):
            rows.append(f"| {label} | — | MISSING | MISSING | MISSING |")
            continue
        for metric in ("final_eval", "last_5_eval_mean", "best_so_far"):
            m = mlp.get(metric); r = rlkv.get(metric)
            if m is None or r is None: continue
            rows.append(f"| {label} | {metric} | {fmt(m)} | {fmt(r)} | {fmt(r - m, 4)} |")
    return rows


def render_seed_summary(parsed: dict[str, dict],
                          arch_label: str,
                          seed_keys: list[str]) -> list[str]:
    """Multi-seed aggregate (mean / stdev) per metric."""
    rows = []
    rows.append(f"| metric | seed values for {arch_label} | mean | stdev |")
    rows.append("|---|---|---:|---:|")
    for metric in ("final_eval", "last_5_eval_mean", "best_so_far"):
        vals = []
        for k in seed_keys:
            info = parsed.get(k, {})
            if info.get("missing") or info.get(metric) is None:
                continue
            vals.append(info[metric])
        if not vals:
            rows.append(f"| {metric} | (no data) | — | — |")
            continue
        s = stdev(vals) if len(vals) >= 2 else 0.0
        rows.append(f"| {metric} | {', '.join(fmt(v) for v in vals)} | "
                    f"{fmt(mean(vals))} | {fmt(s)} |")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None,
                     help="optional: dump machine-readable JSON to this path")
    args = ap.parse_args()

    parsed = {name: parse_log(path) for name, path in RUNS.items()}

    print("# Recomputed scaling-table metrics (auto-generated)\n")
    print("All values extracted programmatically from `[val] step N val_loss=X` "
           "trajectory lines.  No hand math.\n")
    print("Three metrics reported per run:")
    print("* `final_eval`         — value of the last `[val]` line")
    print("* `last_5_eval_mean`   — arithmetic mean of the last 5 `[val]` values")
    print("* `best_so_far`        — minimum val_loss observed during training\n")
    print("`final_val_loss (dump)` is the JSON dump's `final_val_loss` field "
           "(a separate eval at end-of-train; differs from `final_eval` because "
           "it samples different val batches).\n")

    print("---\n")
    print("## Per-run table (raw)\n")
    for line in render_run_metrics(parsed):
        print(line)

    print("\n---\n")
    print("## Multi-seed aggregate at 100M\n")
    print("### MLP h_4d (n=3 seeds)\n")
    for line in render_seed_summary(parsed, "MLP h_4d (s0/s1/s2)",
                                       ["MLP_100M_s0", "MLP_100M_s1", "MLP_100M_s2"]):
        print(line)
    print("\n### RL-KV v11+v5 (n=3 seeds)\n")
    for line in render_seed_summary(parsed, "RL-KV v11+v5 (s0/s1/s2)",
                                       ["RLKV_100M_v11v5_s0", "RLKV_100M_v11v5_s1",
                                        "RLKV_100M_v11v5_s2"]):
        print(line)

    print("\n---\n")
    print("## Pairwise gap (per metric)\n")
    pairs = [
        ("100M s0",   "MLP_100M_s0",   "RLKV_100M_v11v5_s0"),
        ("100M s1",   "MLP_100M_s1",   "RLKV_100M_v11v5_s1"),
        ("100M s2",   "MLP_100M_s2",   "RLKV_100M_v11v5_s2"),
        ("200M s0",   "MLP_200M_s0",   "RLKV_200M_v11v5_s0"),
        ("800M s0",   "MLP_800M_s0",   "RLKV_800M_v11v5_s0"),
    ]
    for line in render_gap_table(parsed, pairs):
        print(line)

    print("\n---\n")
    print("## Stack comparison at 100M s0 (kernel-only ablation)\n")
    for line in render_gap_table(parsed, [
        ("RL-KV stable v1+v1 vs new v11+v5",
         "RLKV_100M_v1v1_s0", "RLKV_100M_v11v5_s0"),
        ("RL-KV new v11+v5 vs partial v11+v1",
         "RLKV_100M_v11v5_s0", "RLKV_100M_v11v1_s0"),
    ]):
        print(line)

    if args.json:
        out = {name: {k: v for k, v in info.items() if k != "trajectory"}
               for name, info in parsed.items()}
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nJSON dumped to {args.json}", flush=True)


if __name__ == "__main__":
    main()
