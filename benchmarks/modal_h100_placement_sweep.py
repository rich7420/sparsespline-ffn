"""Placement sweep dispatcher — P0-Sequential-1.

Fires 5 cells of the d20 200M-param / 1B-token pilot (mode=rlkv_pilot_1B,
~30 min each on 8 × H100 SXM) — one per RL-KV placement variant. All cells
share the same architecture, kernels, and λ-warmup; only the placement
window varies, so any CORE delta is attributable to placement alone.

Cells (d20):

    early33   layers 0..5    (6 layers)   `rl_kv_b2_early33`
    middle33  layers 7..12   (6 layers)   `rl_kv_b2_middle33`
    late33    layers 13..19  (7 layers)   `rl_kv_b2_late33`   [matches Run C]
    late20    layers 16..19  (4 layers)   `rl_kv_b2_late20`
    late50    layers 10..19  (10 layers)  `rl_kv_b2_late50`

The `late33` cell intentionally uses 7 layers (ceil(20/3)) so the result is
directly comparable to the d20 Run C 8.71 B-token training; the other cells
follow the user-specified PLAN layer indices verbatim. The launcher reports
layer counts alongside CORE so a reviewer can normalise per-spline-layer.

Usage:
    # Dry-run (default): just print the modal commands.
    .venv/bin/python benchmarks/modal_h100_placement_sweep.py

    # Execute sequentially (~3 hr total).
    .venv/bin/python benchmarks/modal_h100_placement_sweep.py --execute

    # Subset of cells.
    .venv/bin/python benchmarks/modal_h100_placement_sweep.py --execute \\
        --cells early33,middle33

    # Override the run-tag prefix.
    .venv/bin/python benchmarks/modal_h100_placement_sweep.py --execute \\
        --run-tag-prefix placement_sweep_200M_v2
"""

from __future__ import annotations

import argparse
import dataclasses
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "dispatcher_runs"
MODAL_LAUNCHER = "benchmarks/modal_h100_nanochat_d20.py"


@dataclasses.dataclass(frozen=True)
class Cell:
    name: str           # short cell label, e.g. "early33"
    ffn_type: str       # base_train.py --ffn-type value
    layer_count: int    # # spline layers at d20 (informational)
    layer_indices: str  # informational

CELLS: list[Cell] = [
    Cell("early33",  "rl_kv_b2_early33",  6, "0..5"),
    Cell("middle33", "rl_kv_b2_middle33", 6, "7..12"),
    Cell("late33",   "rl_kv_b2_late33",   7, "13..19"),
    Cell("late20",   "rl_kv_b2_late20",   4, "16..19"),
    Cell("late50",   "rl_kv_b2_late50",  10, "10..19"),
]


def build_modal_cmd(cell: Cell, run_tag_prefix: str) -> list[str]:
    """Return the `modal run` argv for one placement cell.

    Mirrors the d20 Run C training recipe (mode rlkv_pilot_1B = 1900 steps
    @ device_batch=16, total_batch=524288 → ~1 B tokens / ~30 min on 8×H100,
    same RL-KV hyperparameters) so quality numbers are directly comparable.
    """
    run_tag = f"{run_tag_prefix}_{cell.name}"
    return [
        "modal", "run", MODAL_LAUNCHER + "::main",
        "--mode", "rlkv_pilot_1B",
        "--ffn-type", cell.ffn_type,
        "--run-tag", run_tag,
        # Match Run C's RL-KV hyperparameters bit-for-bit.
        "--rlkv-h-ratio", "2.0",
        "--rlkv-r", "32",
        "--rlkv-l", "22",
        "--rlkv-fwd-kernel", "v11_cuda",
        "--rlkv-bwd-kernel", "triton",
        "--rlkv-grid-lo", "-5.0",
        "--rlkv-grid-hi", "5.0",
        "--rlkv-lambda-warmup-steps", "1000",
        "--rlkv-lambda-warmup-lo", "0.25",
        "--c-lr", "0.02",
        "--c-weight-decay", "0.0",
    ]


def run_cell(cell: Cell, run_tag_prefix: str, dry_run: bool, log_dir: Path) -> int:
    cmd = build_modal_cmd(cell, run_tag_prefix)
    pretty = " ".join(shlex.quote(p) for p in cmd)
    print(f"\n[{cell.name}]  layers={cell.layer_count} ({cell.layer_indices})")
    print(f"[{cell.name}]  CMD: {pretty}")

    if dry_run:
        print(f"[{cell.name}]  (dry-run; not executed)")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_tag_prefix}_{cell.name}.log"
    print(f"[{cell.name}]  log -> {log_path}")
    t0 = time.time()
    with log_path.open("w") as logf:
        logf.write(f"# CMD: {pretty}\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"[{cell.name}]  rc={proc.returncode}  dt={dt/60:.1f} min")
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--cells", type=str, default=",".join(c.name for c in CELLS),
        help="Comma-separated cell names to run (default: all 5).",
    )
    parser.add_argument(
        "--run-tag-prefix", type=str, default="placement_sweep_200M",
        help="Run-tag prefix; per-cell tag is <prefix>_<cell> (e.g. "
             "placement_sweep_200M_early33).",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually invoke `modal run` for each cell. Without this flag the "
             "script just prints the commands.",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=LOG_DIR,
        help="Directory for per-cell logs (only used with --execute).",
    )
    args = parser.parse_args(argv)

    requested = [c.strip() for c in args.cells.split(",") if c.strip()]
    name_to_cell = {c.name: c for c in CELLS}
    unknown = [r for r in requested if r not in name_to_cell]
    if unknown:
        print(f"unknown cells: {unknown}; valid: {list(name_to_cell)}", file=sys.stderr)
        return 2
    cells = [name_to_cell[r] for r in requested]

    print(f"Placement sweep — {len(cells)} cells")
    print(f"  prefix    : {args.run_tag_prefix}")
    print(f"  execute   : {args.execute}")
    print(f"  cells     : {[c.name for c in cells]}")

    rcs: dict[str, int] = {}
    t_total = time.time()
    for cell in cells:
        rcs[cell.name] = run_cell(
            cell, args.run_tag_prefix, dry_run=not args.execute, log_dir=args.log_dir
        )
        if args.execute and rcs[cell.name] != 0:
            print(f"[{cell.name}] FAILED (rc={rcs[cell.name]}); aborting sweep.",
                  file=sys.stderr)
            break
    dt_total = time.time() - t_total

    print("\n--- Sweep summary ---")
    for cell in cells:
        rc = rcs.get(cell.name, "skipped")
        print(f"  {cell.name:9s}  layers={cell.layer_count:2d} ({cell.layer_indices})  rc={rc}")
    if args.execute:
        print(f"  total wallclock: {dt_total/60:.1f} min")
    return 0 if all(rc == 0 for rc in rcs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
