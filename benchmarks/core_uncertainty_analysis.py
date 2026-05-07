"""CORE uncertainty analysis for d20 / NeurIPS 2026 placement-sweep evidence.

Implements P0-Parallel-1 from docs/PLAN_2026-05-04_neurips_experiment_queue.md:

    1) Task-level paired bootstrap CI for Δ CORE between two methods.
    2) Sign test on per-task wins/losses (binomial p-value).
    3) Checkpoint-window stability: mean ± std over the last K eval points.

Designed to be reproducible AND re-runnable on new data:
  * Inputs are read from a JSON file (default: benchmarks/data/core_per_task_2026-05-04.json).
  * Adding a new method = append a key under `methods` and a `comparisons` entry.
  * Adding a new task = append to `tasks` and to every method's `centered` list.
  * Bootstrap RNG is seeded; the same --seed always reproduces the same CI.

Usage:
    .venv/bin/python benchmarks/core_uncertainty_analysis.py
    .venv/bin/python benchmarks/core_uncertainty_analysis.py \\
        --data benchmarks/data/core_per_task_2026-05-04.json \\
        --json-out docs/_artifacts/core_uncertainty_2026-05-04.json \\
        --csv-out  docs/_artifacts/core_per_task_2026-05-04.csv \\
        --n-resamples 10000 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

DEFAULT_DATA = Path(__file__).parent / "data" / "core_per_task_2026-05-04.json"


@dataclass
class CoreData:
    tasks: list[str]
    methods: dict[str, np.ndarray]
    method_labels: dict[str, str]
    trajectory_steps: list[int]
    trajectory: dict[str, np.ndarray]
    comparisons: list[tuple[str, str]]
    step: int
    problems_per_task: int
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "CoreData":
        with path.open() as f:
            raw = json.load(f)

        tasks = list(raw["tasks"])
        methods: dict[str, np.ndarray] = {}
        method_labels: dict[str, str] = {}
        for name, entry in raw["methods"].items():
            arr = np.asarray(entry["centered"], dtype=np.float64)
            if arr.size != len(tasks):
                raise ValueError(
                    f"method {name!r}: centered len={arr.size} != tasks len={len(tasks)}"
                )
            methods[name] = arr
            method_labels[name] = entry.get("label", name)

        traj = raw.get("trajectory_core", {})
        traj_steps = list(traj.get("steps", []))
        trajectory = {
            name: np.asarray(traj[name], dtype=np.float64)
            for name in methods
            if name in traj
        }

        comparisons = [
            (c["treat"], c["control"]) for c in raw.get("comparisons", [])
        ]
        for t, c in comparisons:
            if t not in methods or c not in methods:
                raise ValueError(f"comparison ({t!r}, {c!r}) references unknown method")

        return cls(
            tasks=tasks,
            methods=methods,
            method_labels=method_labels,
            trajectory_steps=traj_steps,
            trajectory=trajectory,
            comparisons=comparisons,
            step=int(raw.get("step", -1)),
            problems_per_task=int(raw.get("problems_per_task", -1)),
            source_path=path,
        )


@dataclass
class BootstrapResult:
    treat: str
    control: str
    delta_observed: float
    ci_lo: float
    ci_hi: float
    p_two_sided: float
    n_resamples: int
    seed: int


@dataclass
class SignTestResult:
    treat: str
    control: str
    wins: int
    losses: int
    ties: int
    p_two_sided: float


def bootstrap_delta(
    treat: np.ndarray,
    control: np.ndarray,
    n_resamples: int,
    alpha: float,
    seed: int,
    treat_name: str,
    control_name: str,
) -> BootstrapResult:
    """Task-level paired bootstrap of (treat − control).

    Resamples task indices with replacement and recomputes the mean per-task
    difference. Returns a percentile CI and a two-sided bootstrap p-value
    (twice the fraction of resamples whose sign disagrees with the observed
    mean, capped at 1.0).
    """
    diffs = treat - control
    rng = np.random.default_rng(seed)
    n = diffs.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[idx].mean(axis=1)
    delta_obs = float(diffs.mean())
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    if delta_obs >= 0.0:
        p_one = float((boot_means <= 0.0).mean())
    else:
        p_one = float((boot_means >= 0.0).mean())
    p_two = min(1.0, 2.0 * p_one)
    return BootstrapResult(
        treat=treat_name,
        control=control_name,
        delta_observed=delta_obs,
        ci_lo=float(lo),
        ci_hi=float(hi),
        p_two_sided=p_two,
        n_resamples=n_resamples,
        seed=seed,
    )


def sign_test(
    treat: np.ndarray,
    control: np.ndarray,
    treat_name: str,
    control_name: str,
) -> SignTestResult:
    diffs = treat - control
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())
    n = wins + losses
    p = (
        float(binomtest(wins, n=n, p=0.5, alternative="two-sided").pvalue)
        if n > 0
        else 1.0
    )
    return SignTestResult(
        treat=treat_name,
        control=control_name,
        wins=wins,
        losses=losses,
        ties=ties,
        p_two_sided=p,
    )


def window_stability(data: CoreData) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method, vals in data.trajectory.items():
        out[method] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": int(vals.size),
        }
    return out


def write_per_task_csv(data: CoreData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    method_names = list(data.methods.keys())
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", *method_names])
        for i, t in enumerate(data.tasks):
            w.writerow([t] + [f"{data.methods[m][i]:.4f}" for m in method_names])


def render_report(
    data: CoreData,
    boot: list[BootstrapResult],
    signs: list[SignTestResult],
    win: dict[str, dict[str, float]],
    n_resamples: int,
    seed: int,
) -> str:
    lines: list[str] = []
    add = lines.append
    bar = "=" * 72
    sub = "-" * 72
    add(bar)
    add("CORE uncertainty analysis")
    add(f"  source       : {data.source_path}")
    add(f"  step         : {data.step}")
    add(f"  problems/task: {data.problems_per_task}")
    add(f"  num tasks    : {len(data.tasks)}")
    add(f"  bootstrap B  : {n_resamples}    seed: {seed}")
    add(bar)

    add("")
    add("Method CORE means (mean of per-task centered scores):")
    for m, arr in data.methods.items():
        add(f"  {m:8s}  CORE = {arr.mean():.4f}   ({data.method_labels[m]})")

    add("")
    add(sub)
    add("1) Task-level paired bootstrap (95% CI, two-sided p)")
    add(sub)
    for r in boot:
        add(
            f"  {r.treat:6s} − {r.control:6s}  Δ={r.delta_observed:+.4f}  "
            f"95% CI [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]  p≈{r.p_two_sided:.3f}"
        )

    add("")
    add(sub)
    add("2) Sign test on per-task wins/losses (binomial two-sided)")
    add(sub)
    for r in signs:
        add(
            f"  {r.treat:6s} vs {r.control:6s}  wins={r.wins:2d}  losses={r.losses:2d}  "
            f"ties={r.ties}  p={r.p_two_sided:.3f}"
        )

    add("")
    add(sub)
    add(f"3) Checkpoint-window stability  steps={data.trajectory_steps}")
    add(sub)
    for m, st in win.items():
        add(
            f"  {m:8s}  mean={st['mean']:.4f}  std={st['std']:.4f}  "
            f"range=[{st['min']:.4f}, {st['max']:.4f}]  n={st['n']}"
        )
    if data.trajectory_steps and len(data.methods) >= 2:
        add("")
        add("  Per-step Δ vs MLP V2 (if present):")
        if "MLP V2" in data.trajectory:
            base = data.trajectory["MLP V2"]
            for m, arr in data.trajectory.items():
                if m == "MLP V2":
                    continue
                d = arr - base
                add(
                    f"    {m:8s}  per-step={[round(v, 4) for v in d.tolist()]}  "
                    f"mean={d.mean():+.4f}  std={d.std(ddof=1):.4f}"
                )

    add("")
    add(sub)
    add("4) Per-task centered scores")
    add(sub)
    method_names = list(data.methods.keys())
    header = f"  {'task':36s}  " + "  ".join(f"{m:>8s}" for m in method_names)
    add(header)
    for i, t in enumerate(data.tasks):
        cells = "  ".join(f"{data.methods[m][i]:8.4f}" for m in method_names)
        add(f"  {t:36s}  {cells}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args(argv)

    data = CoreData.load(args.data)

    boot = [
        bootstrap_delta(
            data.methods[t],
            data.methods[c],
            n_resamples=args.n_resamples,
            alpha=args.alpha,
            seed=args.seed,
            treat_name=t,
            control_name=c,
        )
        for (t, c) in data.comparisons
    ]
    signs = [
        sign_test(data.methods[t], data.methods[c], treat_name=t, control_name=c)
        for (t, c) in data.comparisons
    ]
    win = window_stability(data)

    report = render_report(
        data, boot, signs, win, args.n_resamples, args.seed
    )
    print(report)

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report + "\n")
        print(f"\nWrote report : {args.report_out}")

    if args.csv_out:
        write_per_task_csv(data, args.csv_out)
        print(f"Wrote CSV    : {args.csv_out}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "source": str(data.source_path),
            "step": data.step,
            "problems_per_task": data.problems_per_task,
            "n_tasks": len(data.tasks),
            "bootstrap": {
                "n_resamples": args.n_resamples,
                "alpha": args.alpha,
                "seed": args.seed,
                "results": [asdict(r) for r in boot],
            },
            "sign_test": [asdict(r) for r in signs],
            "core_means": {m: float(arr.mean()) for m, arr in data.methods.items()},
            "window_stability": win,
            "trajectory_steps": data.trajectory_steps,
            "trajectory": {m: arr.tolist() for m, arr in data.trajectory.items()},
            "tasks": data.tasks,
            "per_task": {m: arr.tolist() for m, arr in data.methods.items()},
        }
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"Wrote JSON   : {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
