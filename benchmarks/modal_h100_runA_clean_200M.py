"""Clean Run A 200M rerun dispatcher — P0-Sequential-4.

Reruns the all-layers RL-KV negative control at 200M / 1B-token scale,
no preempt confound this time. Used to confirm that the d20 Run A
regression (val bpb +0.0205, CORE −0.0233 vs MLP) is real and not just an
artifact of Modal worker preemption + dataloader random-state reset that
hit the 8.71B run.

Single cell, ~30 min on 8 × H100. Same hyperparameters as Run A's full d20
training; only the schedule shrinks (1900 steps instead of 16 600).

Usage:
    .venv/bin/python benchmarks/modal_h100_runA_clean_200M.py             # dry-run
    .venv/bin/python benchmarks/modal_h100_runA_clean_200M.py --execute   # run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "dispatcher_runs"
MODAL_LAUNCHER = "benchmarks/modal_h100_nanochat_d20.py"


def build_modal_cmd(run_tag: str) -> list[str]:
    """All-layers RL-KV @ 1 B tokens, Run-A hyperparameters."""
    return [
        "modal", "run", MODAL_LAUNCHER + "::main",
        "--mode", "rlkv_pilot_1B",
        "--ffn-type", "rl_kv_b2",          # all 20 layers
        "--run-tag", run_tag,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run-tag", type=str, default="runA_clean_200M_seed0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    args = parser.parse_args(argv)

    cmd = build_modal_cmd(args.run_tag)
    pretty = " ".join(shlex.quote(p) for p in cmd)
    print(f"Run A clean 200M rerun")
    print(f"  run_tag : {args.run_tag}")
    print(f"  execute : {args.execute}")
    print(f"  CMD     : {pretty}")

    if not args.execute:
        print("\n(dry-run; not executed)")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{args.run_tag}.log"
    print(f"  log     : {log_path}")
    t0 = time.time()
    with log_path.open("w") as logf:
        logf.write(f"# CMD: {pretty}\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"\nrc={proc.returncode}  dt={dt/60:.1f} min")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
