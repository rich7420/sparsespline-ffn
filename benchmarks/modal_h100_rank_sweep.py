"""Run C rank sweep dispatcher — P0-Sequential-2.

Sweeps the spline residual rank r ∈ {16, 32, 64, 128} on the late33 placement
at 200M / 1B-token scale. Adds an MLP V2 baseline at the same step budget so
the bpb-vs-rank curve has its zero-rank reference. Total: 5 cells × ~30 min
on 8 × H100.

Goal: prove (or refute) that the +0.0064 bpb residual of Run C vs MLP at full
d20 is rank-bottlenecked. If bpb gap shrinks with r → 64 / 128 while CORE
keeps winning, the paper claims "structured FFN at adequate rank" cleanly.

Usage:
    .venv/bin/python benchmarks/modal_h100_rank_sweep.py             # dry-run
    .venv/bin/python benchmarks/modal_h100_rank_sweep.py --execute   # run
    .venv/bin/python benchmarks/modal_h100_rank_sweep.py --execute --cells r64,r128
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
    name: str
    ffn_type: str
    rlkv_r: int | None  # None for MLP baseline
    note: str

CELLS: list[Cell] = [
    Cell("mlp",  "mlp",              None, "MLP V2 baseline (zero-rank reference)"),
    Cell("r16",  "rl_kv_b2_late33",  16,   "late33, r=16 (constrained)"),
    Cell("r32",  "rl_kv_b2_late33",  32,   "late33, r=32 (matches Run C)"),
    Cell("r64",  "rl_kv_b2_late33",  64,   "late33, r=64 (uplift?)"),
    Cell("r128", "rl_kv_b2_late33",  128,  "late33, r=128 (capacity test)"),
]


def build_modal_cmd(cell: Cell, run_tag_prefix: str) -> list[str]:
    """Modal CLI args for one rank-sweep cell.

    Both MLP baseline and RL-KV cells run the SAME pilot mode (rlkv_pilot_1B
    = 1900 steps × 524 288 tokens/step = 1 B tokens) so token budgets are
    identical and the bpb axis is directly comparable.
    """
    run_tag = f"{run_tag_prefix}_{cell.name}"
    base = [
        "modal", "run", MODAL_LAUNCHER + "::main",
        "--mode", "rlkv_pilot_1B",
        "--ffn-type", cell.ffn_type,
        "--run-tag", run_tag,
    ]
    if cell.ffn_type == "mlp":
        # MLP: don't pass any RL-KV flags. The d20 launcher's build_args
        # short-circuits the RL-KV branch when ffn_type == "mlp", but the
        # local_entrypoint requires a uniform set of CLI args, so explicit
        # defaults keep it predictable.
        return base
    # RL-KV cells share Run-C hyperparameters except for `r`.
    return base + [
        "--rlkv-h-ratio", "2.0",
        "--rlkv-r", str(cell.rlkv_r),
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
    print(f"\n[{cell.name}]  {cell.note}")
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
        help="Comma-separated cell names (default: all 5).",
    )
    parser.add_argument("--run-tag-prefix", type=str, default="rank_sweep_200M")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    args = parser.parse_args(argv)

    requested = [c.strip() for c in args.cells.split(",") if c.strip()]
    name_to_cell = {c.name: c for c in CELLS}
    unknown = [r for r in requested if r not in name_to_cell]
    if unknown:
        print(f"unknown cells: {unknown}; valid: {list(name_to_cell)}", file=sys.stderr)
        return 2
    cells = [name_to_cell[r] for r in requested]

    print(f"Rank sweep — {len(cells)} cells")
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
        rstr = f"r={cell.rlkv_r}" if cell.rlkv_r else "MLP baseline"
        print(f"  {cell.name:5s}  {rstr:18s}  rc={rc}")
    if args.execute:
        print(f"  total wallclock: {dt_total/60:.1f} min")
    return 0 if all(rc == 0 for rc in rcs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
