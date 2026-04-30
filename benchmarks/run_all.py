"""Master benchmark runner.

Runs every benchmark script in this directory, captures stdout to
benchmark_runs/<timestamp>/<name>.txt, and prints a summary table.

Usage:
    python benchmarks/run_all.py                # run all
    python benchmarks/run_all.py --only quality # only files matching 'quality'
    python benchmarks/run_all.py --skip latency # skip slow ones
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent

# Each entry is (name, args).  Order matters for readability of the report.
# Cheap analytical benchmarks first, then quality benchmarks (training),
# then system benchmarks (latency / memory).
BENCHMARKS = [
    # --- analytical / static ---
    ("param_count",            []),
    ("flops",                  []),
    ("activation_memory",      []),
    ("invariant_audit",        []),
    # --- quality / training ---
    ("quality_regression",     []),
    ("quality_high_freq",      []),
    ("quality_jacobian",       []),
    ("quality_distill",        []),
    ("quality_convergence",    []),
    ("quality_rank_sweep",     []),     # F.4.b
    ("quality_asymmetric_rank", []),    # F.4.c Strategy A
    ("quality_placement_K",    []),     # F.5.1
    ("quality_warmstart",      []),     # L.4 HOSVD
    ("quality_mixer_ablation", []),     # M.5
    ("quality_grid_resolution", []),    # E.2 / I.2 G sweep
    # --- diagnostics ---
    ("init_sensitivity",       []),     # L.4 sigma_c sweep
    ("subspace_diversity",     []),     # F.5.1 caveat
    # --- system ---
    ("latency",                ["--B", "4", "--T", "512",
                                "--warmup", "5", "--iters", "20"]),
    ("fwd_bwd_split",          ["--B", "4", "--T", "512",
                                "--warmup", "5", "--iters", "20"]),
]


def _filter(names: list[tuple[str, list[str]]],
            only: list[str], skip: list[str]) -> list[tuple[str, list[str]]]:
    out = []
    for name, args in names:
        if only and not any(p in name for p in only):
            continue
        if skip and any(p in name for p in skip):
            continue
        out.append((name, args))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[],
                    help="substrings; only run benchmarks whose name matches")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="substrings; skip benchmarks whose name matches")
    ap.add_argument("--out-dir", default=None,
                    help="override output dir (default: benchmark_runs/<ts>)")
    args = ap.parse_args()

    selected = _filter(BENCHMARKS, args.only, args.skip)
    if not selected:
        print("No benchmarks selected.")
        return 1

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "benchmark_runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(selected)} benchmark(s); output -> {out_dir}\n")
    summary: list[tuple[str, str, float]] = []
    for name, extra_args in selected:
        script = BENCH_DIR / f"{name}.py"
        if not script.exists():
            print(f"  [SKIP] {name}: script not found")
            continue
        cmd = [sys.executable, str(script), *extra_args]
        out_file = out_dir / f"{name}.txt"
        print(f"  [RUN ] {name}  -> {out_file.name}", flush=True)
        t0 = time.perf_counter()
        with out_file.open("w") as f:
            f.write(f"$ {' '.join(cmd)}\n\n")
            try:
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT,
                               cwd=REPO_ROOT, text=True)
                status = "OK"
            except subprocess.CalledProcessError as e:
                status = f"FAIL(exit={e.returncode})"
        dt_s = time.perf_counter() - t0
        summary.append((name, status, dt_s))
        print(f"         {status} in {dt_s:.1f}s")

    print("\n" + "=" * 60)
    print(f"  Summary  ({len(summary)} runs)")
    print("=" * 60)
    print(f"  {'name':<24} {'status':<10} {'wall(s)':>10}")
    for name, status, wall in summary:
        print(f"  {name:<24} {status:<10} {wall:>10.1f}")
    print(f"\n  Outputs in: {out_dir}")

    # Auto-print headlines from each benchmark's stdout
    print("\n" + "=" * 60)
    print("  Headlines  (last few summary lines per benchmark)")
    print("=" * 60)
    for name, status, _wall in summary:
        if status != "OK":
            continue
        path = out_dir / f"{name}.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # Look for the first short table or "Ratios" / "FullMix" lines.
        lines = text.splitlines()
        # Print between the last "===" separator and EOF, if present.
        last_sep = None
        for i in range(len(lines) - 1, -1, -1):
            if re.match(r"^=+\s*$", lines[i]):
                last_sep = i
                break
        if last_sep is not None:
            tail = lines[last_sep:last_sep + 12]
        else:
            tail = lines[-10:]
        print(f"\n[{name}]")
        for ln in tail:
            print("  " + ln)

    return 0 if all(s == "OK" for _n, s, _w in summary) else 2


if __name__ == "__main__":
    sys.exit(main())
