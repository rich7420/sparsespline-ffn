"""Parse all training-result artifacts and emit CSV + markdown.

Sources:
  - 3080 (Phase 0 / Stage-3): JSON files in
    nanochat/.nanochat-runtime/runs/v41_stage3/
  - H100 (Phase 1, Modal): JSON files in
    nanochat/.nanochat-runtime/runs/v41_h100/  OR
    JSON blobs embedded in Modal subprocess logs (/tmp/*_h100.txt).

Outputs:
  - CSV (machine-readable): docs/results_summary.csv
  - Markdown (human-readable): printed to stdout, optionally saved.

Columns: cell, source, n_steps, params_M, wall_s, peak_MB,
         val_final, val_best, best_step, gap_final, gap_best,
         degradation, late_pct
where:
  gap_X        = val_X − MLP_baseline_X      (positive = worse)
  degradation  = val_final − val_best         (positive = run got worse near end)
  late_pct     = (final_step − best_step) / final_step  (how far past best)

Run:
  python benchmarks/results_summary.py
  python benchmarks/results_summary.py --csv-out my.csv --md-out my.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parent.parent

# Default search locations
LOCAL_3080_DIR = REPO / "nanochat" / ".nanochat-runtime" / "runs" / "v41_stage3"
LOCAL_H100_DIR = REPO / "nanochat" / ".nanochat-runtime" / "runs" / "v41_h100"
MODAL_LOG_DIR = Path("/tmp")  # /tmp/{cell}_h100.txt blobs from Modal stdouts


def _parse_run_json(path: Path, source: str) -> dict | None:
    """Read a Stage-3 JSON dump and pull out the fields we display."""
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        print(f"[warn] could not parse {path}: {e}", file=sys.stderr)
        return None
    v = d.get("val_history", [])
    if not v:
        return None
    final = d.get("final_val_loss", v[-1]["val_loss"])
    best = min(v, key=lambda e: e["val_loss"])
    last_step = v[-1]["step"]
    return {
        "cell":       d.get("mode", path.stem),
        "source":     source,
        "n_steps":    int(d.get("n_steps", last_step)),
        "params_M":   float(d.get("params_m", float("nan"))),
        "wall_s":     float(d.get("wall_s", float("nan"))),
        "peak_MB":    float(d.get("peak_mb", float("nan"))),
        "val_final":  float(final),
        "val_best":   float(best["val_loss"]),
        "best_step":  int(best["step"]),
        "last_step":  int(last_step),
        "_path":      str(path),
    }


def _parse_modal_log(path: Path) -> dict | None:
    """Extract embedded Stage-3 JSON from a Modal subprocess.run() output.

    The Modal launcher (`benchmarks/modal_h100_train.py`) calls the
    nanochat training script with capture_output=True, then prints the
    captured stdout — which contains the dumped JSON.  We grep the JSON
    fields directly so we do not depend on exact print structure.
    """
    try:
        text = path.read_text()
    except Exception:
        return None
    if "val_history" not in text:
        return None
    # Pull all (step, val_loss) pairs robustly, regardless of formatting.
    pairs = re.findall(r'"step":\s*(\d+)\s*,\s*"val_loss":\s*([\d.]+)', text)
    if not pairs:
        return None
    by_step: dict[int, float] = {}
    for s, v in pairs:
        by_step[int(s)] = float(v)
    items = sorted(by_step.items())
    last_step, final_val = items[-1]
    best_step, best_val = min(items, key=lambda x: x[1])
    # Other fields
    def _pick(field: str) -> float:
        m = re.search(rf'"{field}":\s*([\d.]+)', text)
        return float(m.group(1)) if m else float("nan")
    cell_match = re.search(r'"mode":\s*"([^"]+)"', text)
    cell = cell_match.group(1) if cell_match else path.stem.replace("_h100", "")
    return {
        "cell":      cell,
        "source":    "h100",
        "n_steps":   last_step,
        "params_M":  _pick("params_m"),
        "wall_s":    _pick("wall_s"),
        "peak_MB":   _pick("peak_mb"),
        "val_final": final_val,
        "val_best":  best_val,
        "best_step": best_step,
        "last_step": last_step,
        "_path":     str(path),
    }


def collect(
    local_dirs: Iterable[Path] = (LOCAL_3080_DIR, LOCAL_H100_DIR),
    modal_log_glob: str = str(MODAL_LOG_DIR / "*_h100.txt"),
) -> list[dict]:
    rows: list[dict] = []
    seen_cells: set[tuple[str, str]] = set()  # (cell, source)
    # 1) Local JSON files (3080 + persisted H100 from volume)
    for d in local_dirs:
        if not d.exists():
            continue
        source = "3080" if "stage3" in d.name else "h100"
        for p in sorted(d.glob("*.json")):
            row = _parse_run_json(p, source)
            if row is None:
                continue
            key = (row["cell"], row["source"])
            if key in seen_cells:
                continue
            seen_cells.add(key)
            rows.append(row)
    # 2) Modal subprocess log blobs in /tmp (live H100 runs)
    for path_str in sorted(glob.glob(modal_log_glob)):
        p = Path(path_str)
        row = _parse_modal_log(p)
        if row is None:
            continue
        key = (row["cell"], row["source"])
        if key in seen_cells:
            continue
        seen_cells.add(key)
        rows.append(row)
    return rows


def annotate(rows: list[dict]) -> list[dict]:
    """Add gap-to-baseline and degradation columns."""
    # Find MLP baseline (use 3080 mlp_baseline if present)
    base = None
    for r in rows:
        if r["cell"] == "mlp_baseline":
            base = r
            break
    base_final = base["val_final"] if base else None
    base_best = base["val_best"] if base else None

    for r in rows:
        r["gap_final"]    = (r["val_final"] - base_final) if base_final is not None else float("nan")
        r["gap_best"]     = (r["val_best"]  - base_best)  if base_best  is not None else float("nan")
        r["degradation"]  = r["val_final"] - r["val_best"]
        r["late_pct"]     = (
            (r["last_step"] - r["best_step"]) / r["last_step"]
            if r["last_step"] > 0 else 0.0
        )
    return rows


def to_csv(rows: list[dict], path: Path) -> None:
    cols = ["cell", "source", "n_steps", "params_M", "wall_s", "peak_MB",
            "val_final", "val_best", "best_step", "last_step",
            "gap_final", "gap_best", "degradation", "late_pct"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def to_markdown(rows: list[dict]) -> str:
    cols = [
        ("cell",        "cell",          "{}"),
        ("source",      "src",           "{}"),
        ("params_M",    "params (M)",    "{:.1f}"),
        ("wall_s",      "wall (s)",      "{:.0f}"),
        ("peak_MB",     "peak (MB)",     "{:.0f}"),
        ("val_final",   "final",         "{:.4f}"),
        ("val_best",    "best",          "{:.4f}"),
        ("best_step",   "best step",     "{}"),
        ("gap_final",   "gap-final",     "{:+.3f}"),
        ("gap_best",    "gap-best",      "{:+.3f}"),
        ("degradation", "degr (best→final)", "{:+.3f}"),
        ("late_pct",    "best-at-pct",   "{:.0%}"),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep    = "|" + "|".join("---" for _ in cols) + "|"

    def fmt(row: dict) -> str:
        cells = []
        for key, _, fmt_s in cols:
            v = row.get(key, float("nan"))
            try:
                cells.append(fmt_s.format(v))
            except Exception:
                cells.append(str(v))
        return "| " + " | ".join(cells) + " |"

    # Sort: 3080 first by val_best ascending, then h100 by val_best ascending
    rows_sorted = sorted(
        rows,
        key=lambda r: (0 if r["source"] == "3080" else 1, r["val_best"]),
    )
    lines = [header, sep] + [fmt(r) for r in rows_sorted]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-out",
                    default=str(REPO / "benchmark_runs" / "results_summary.csv"))
    ap.add_argument("--md-out", default=None,
                    help="if given, also write markdown to this file")
    ap.add_argument("--no-csv", action="store_true",
                    help="skip writing CSV (just print markdown)")
    args = ap.parse_args()

    rows = annotate(collect())
    md = to_markdown(rows)
    print(md)
    print(f"\n[parsed {len(rows)} runs from "
          f"3080 ({LOCAL_3080_DIR}) and h100 (volume + /tmp/*_h100.txt)]")

    if not args.no_csv:
        out = Path(args.csv_out)
        to_csv(rows, out)
        print(f"[csv written to {out}]")
    if args.md_out:
        Path(args.md_out).write_text(md + "\n")
        print(f"[md written to {args.md_out}]")


if __name__ == "__main__":
    main()
