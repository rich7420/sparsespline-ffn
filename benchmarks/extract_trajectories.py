"""Extract bpb + CORE trajectories from dispatcher_runs/*.log into JSON.

Parses lines of the form:
    Step <step> | Validation bpb: <float>
    Step <step> | CORE metric: <float>

Outputs a JSON file consumed by `benchmarks/core_trajectory_plots.py`.

Usage:
    .venv/bin/python benchmarks/extract_trajectories.py \\
        --out benchmarks/data/trajectory_2026-05-04.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "benchmarks" / "data" / "trajectory_2026-05-04.json"

# (display name, list of (log path, step_min_inclusive, step_max_inclusive))
# Run A: take steps 0..12000 from the original log, then 12250..16600 from the
# resume log. The resume log re-evaluates step 12000 (identical, since it loads
# the ckpt) so we drop its 12000 row to avoid a duplicate.
SOURCES: list[tuple[str, list[tuple[Path, int, int]]]] = [
    (
        "MLP V2",
        [(ROOT / "dispatcher_runs" / "mlp_d20_reference_v2_seed0.log", 0, 16600)],
    ),
    (
        "Run C",
        [(ROOT / "dispatcher_runs" / "full_rlkv_late33_grid5_lwarmup.log", 0, 16600)],
    ),
    (
        "Run A",
        [
            (ROOT / "dispatcher_runs" / "2026-05-04_full_rlkv_b2_all_runA.log", 0, 12000),
            (
                ROOT / "dispatcher_runs" / "2026-05-04_full_rlkv_b2_all_runA_resume12000.log",
                12250,
                16600,
            ),
        ],
    ),
]

# Pre-preempt continuation from Run A (12250..13250) — kept as a separate
# series so the plot can overlay "what might have been" against the resumed
# trajectory.
PREEMPT_AUX: list[tuple[str, Path, int, int]] = [
    (
        "Run A (pre-preempt continuation)",
        ROOT / "dispatcher_runs" / "2026-05-04_full_rlkv_b2_all_runA.log",
        12250,
        13250,
    ),
]

BPB_RE = re.compile(r"Step\s+(\d+)\s*\|\s*Validation bpb:\s*([0-9.eE+-]+)")
CORE_RE = re.compile(r"Step\s+(\d+)\s*\|\s*CORE metric:\s*([0-9.eE+-]+)")


@dataclass
class Series:
    bpb: dict[int, float]
    core: dict[int, float]

    def to_jsonable(self) -> dict:
        return {
            "bpb_steps":  sorted(self.bpb.keys()),
            "bpb_values": [self.bpb[s] for s in sorted(self.bpb.keys())],
            "core_steps":  sorted(self.core.keys()),
            "core_values": [self.core[s] for s in sorted(self.core.keys())],
        }


def parse_log(path: Path, step_min: int, step_max: int) -> Series:
    bpb: dict[int, float] = {}
    core: dict[int, float] = {}
    with path.open() as f:
        for line in f:
            m = BPB_RE.search(line)
            if m:
                step = int(m.group(1))
                if step_min <= step <= step_max:
                    bpb[step] = float(m.group(2))
                continue
            m = CORE_RE.search(line)
            if m:
                step = int(m.group(1))
                if step_min <= step <= step_max:
                    core[step] = float(m.group(2))
    return Series(bpb=bpb, core=core)


def merge_series(parts: list[Series]) -> Series:
    bpb: dict[int, float] = {}
    core: dict[int, float] = {}
    for s in parts:
        # Later sources win on collision.
        bpb.update(s.bpb)
        core.update(s.core)
    return Series(bpb=bpb, core=core)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    methods: dict[str, Series] = {}
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for method, sources in SOURCES:
        parts = [parse_log(p, lo, hi) for (p, lo, hi) in sources]
        merged = merge_series(parts)
        methods[method] = merged
        counts[method] = {"bpb_n": len(merged.bpb), "core_n": len(merged.core)}
        print(
            f"  {method:8s}  bpb={len(merged.bpb):3d} pts  core={len(merged.core):3d} pts  "
            f"steps {min(merged.bpb)}..{max(merged.bpb)}"
        )

    aux: dict[str, Series] = {}
    for name, path, lo, hi in PREEMPT_AUX:
        s = parse_log(path, lo, hi)
        aux[name] = s
        print(f"  AUX {name}: bpb={len(s.bpb)} core={len(s.core)} steps {lo}..{hi}")

    out = {
        "_doc": (
            "bpb (every 250 steps) + CORE (every 2000 steps) trajectories for the d20 "
            "3-way comparison. Extracted from dispatcher_runs/*.log by extract_trajectories.py."
        ),
        "annotations": {
            "warmdown_begin_step": 13280,
            "runA_resume_step": 12000,
            "final_step": 16600,
        },
        "methods": {m: s.to_jsonable() for m, s in methods.items()},
        "auxiliary": {m: s.to_jsonable() for m, s in aux.items()},
        "source_logs": {
            method: [str(p.relative_to(ROOT)) for (p, _, _) in sources]
            for method, sources in SOURCES
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
