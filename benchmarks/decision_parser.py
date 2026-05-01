"""Auto-read all H100 + 3080 results and emit a decision.md.

Per v7 §R.9 / RESULTS_2026-05-01.md §8 decision tree:

  1. Read every JSON / log file we have for Stage 3 100M cells.
  2. Group by (cell, schedule_tag).
  3. Compute apples-to-apples comparisons within same schedule.
  4. Emit a decision.md that walks the v7 §R.9 lookup table for the
     current state and recommends the next action.

Idempotent — safe to re-run any time new H100 results land.

Run:
  python benchmarks/decision_parser.py
  python benchmarks/decision_parser.py --out benchmark_runs/decision.md
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
LOCAL_3080_DIR = REPO / "nanochat" / ".nanochat-runtime" / "runs" / "v41_stage3"
LOCAL_H100_DIR = REPO / "nanochat" / ".nanochat-runtime" / "runs" / "v41_h100"

# Modal subprocess logs in /tmp; tag is captured from filename
MODAL_LOG_PATTERNS = ["/tmp/*_h100.txt", "/tmp/A1_*", "/tmp/A2_*", "/tmp/A2b_*"]


def _parse_json(path: Path, source: str, tag: str = "") -> dict | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    v = d.get("val_history", [])
    if not v:
        return None
    final = d.get("final_val_loss", v[-1]["val_loss"])
    best = min(v, key=lambda e: e["val_loss"])
    return {
        "cell":       d.get("mode", path.stem),
        "tag":        tag,
        "source":     source,
        "n_steps":    int(d.get("n_steps", v[-1]["step"])),
        "params_M":   float(d.get("params_m", float("nan"))),
        "wall_s":     float(d.get("wall_s", float("nan"))),
        "peak_MB":    float(d.get("peak_mb", float("nan"))),
        "val_final":  float(final),
        "val_best":   float(best["val_loss"]),
        "best_step":  int(best["step"]),
        "last_step":  int(v[-1]["step"]),
        "_path":      str(path),
    }


def _parse_modal_log(path: Path) -> dict | None:
    try:
        text = path.read_text()
    except Exception:
        return None
    if "val_history" not in text:
        return None
    pairs = re.findall(r'"step":\s*(\d+)\s*,\s*"val_loss":\s*([\d.]+)', text)
    if not pairs:
        return None
    by_step = {int(s): float(v) for s, v in pairs}
    items = sorted(by_step.items())
    cell_match = re.search(r'"mode":\s*"([^"]+)"', text)
    cell = cell_match.group(1) if cell_match else path.stem.replace("_h100", "")

    # Tag mapping (filename → semantic tag).
    # A1 + A2 were launched together with peak_lr=2e-4, warmup=1500.
    # A2b is the rerun with original schedule (peak_lr=3e-4, warmup=500).
    # Any other "_h100.txt" stem → no tag (default schedule = orig).
    tag = ""
    name = path.stem
    if name.startswith("A1_") or name.startswith("A2_"):
        tag = "cosine_lr2e4_w1500"
    elif name.startswith("A2b_"):
        tag = "orig_lr3e4_w500"
    elif name.startswith("A3_"):
        tag = "orig_lr3e4_w500"

    def _pick(field: str) -> float:
        m = re.search(rf'"{field}":\s*([\d.]+)', text)
        return float(m.group(1)) if m else float("nan")

    last_step, final_val = items[-1]
    best_step, best_val = min(items, key=lambda x: x[1])
    return {
        "cell":      cell,
        "tag":       tag,
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


def collect_all() -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()  # (cell, tag, source)

    # Stage 3 (3080) — no tag
    if LOCAL_3080_DIR.exists():
        for p in sorted(LOCAL_3080_DIR.glob("*.json")):
            r = _parse_json(p, "3080", tag="")
            if r is None:
                continue
            key = (r["cell"], r["tag"], r["source"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    # H100 volume snapshots (if mirrored locally) — tag from filename
    if LOCAL_H100_DIR.exists():
        for p in sorted(LOCAL_H100_DIR.glob("*.json")):
            # filename pattern: {cell}_train.json or {cell}_{tag}_train.json
            stem = p.stem.removesuffix("_train")
            r = _parse_json(p, "h100", tag="")
            if r is None:
                continue
            # try to extract tag if cell != stem
            if r["cell"] and stem.startswith(r["cell"]):
                r["tag"] = stem[len(r["cell"]):].lstrip("_")
            key = (r["cell"], r["tag"], r["source"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    # Modal subprocess logs in /tmp
    for pat in MODAL_LOG_PATTERNS:
        for path_str in sorted(glob.glob(pat)):
            r = _parse_modal_log(Path(path_str))
            if r is None:
                continue
            key = (r["cell"], r["tag"], r["source"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    return rows


def annotate(rows: list[dict]) -> list[dict]:
    base = next((r for r in rows if r["cell"] == "mlp_baseline"), None)
    base_final = base["val_final"] if base else None
    base_best = base["val_best"] if base else None
    for r in rows:
        r["gap_final"]   = (r["val_final"] - base_final) if base_final is not None else float("nan")
        r["gap_best"]    = (r["val_best"]  - base_best)  if base_best  is not None else float("nan")
        r["degradation"] = r["val_final"] - r["val_best"]
        r["late_pct"]    = ((r["last_step"] - r["best_step"]) / r["last_step"]
                            if r["last_step"] > 0 else 0.0)
    return rows


def _row_label(r: dict) -> str:
    return f"{r['cell']}" + (f" [{r['tag']}]" if r["tag"] else "")


def render_decision(rows: list[dict]) -> str:
    """Walk v7 §R.9 decision tree given current data."""
    # Group helpers
    by_label: dict[str, dict] = {}
    for r in rows:
        by_label[_row_label(r)] = r

    # Pivots
    mlp = by_label.get("mlp_baseline")
    # ss_pa6 ORIGINAL had no tag (it was launched before we added --tag)
    ss_orig = by_label.get("ss_pa6")
    ss_cosine = by_label.get("ss_pa6 [cosine_lr2e4_w1500]")
    narrow_cosine = by_label.get("narrow_relu2_h_d [cosine_lr2e4_w1500]")
    narrow_orig = (by_label.get("narrow_relu2_h_d [orig_lr3e4_w500]")
                   or by_label.get("narrow_relu2_h_d"))
    ss_pa6_h_d = by_label.get("ss_pa6_h_d")
    ss_pa6_h_d_tagged = next(
        (r for k, r in by_label.items() if r["cell"] == "ss_pa6_h_d"), None)

    md_lines = ["# RL-Spline-KV / SimpleSpline — auto-decision",
                "",
                "*Generated by `benchmarks/decision_parser.py`. Re-run any time data lands.*",
                "",
                "## Apples-to-apples scoreboard",
                "",
                "| cell [tag] | source | best | final | gap-best | degr | best-pct |",
                "|---|---|---:|---:|---:|---:|---:|"]
    rows_sorted = sorted(rows, key=lambda r: r["val_best"])
    for r in rows_sorted:
        label = _row_label(r)
        md_lines.append(
            f"| {label} | {r['source']} | {r['val_best']:.4f} | {r['val_final']:.4f} | "
            f"{r['gap_best']:+.3f} | {r['degradation']:+.3f} | "
            f"{r['late_pct']:.0%} |"
        )

    md_lines += ["", "## Decision tree (per v7 §R.9 / RESULTS §8)",
                 ""]

    # 1. Schedule-fix question
    if ss_orig and ss_cosine:
        delta = ss_cosine["val_best"] - ss_orig["val_best"]
        if delta > 0.05:
            md_lines += [
                f"**Schedule fix outcome.** Lower peak_lr (cosine variant)",
                f"made ss_pa6 *worse* by {delta:+.3f} nats best-vs-best.",
                "Conclusion: original schedule (peak_lr=3e-4, warmup=500) "
                "is closer to the SS optimum.  The 'best @ 30K then degrade' "
                "pattern is **architectural, not schedule-fixable**.",
                "",
            ]
        elif delta < -0.05:
            md_lines += [
                f"**Schedule fix outcome.** Cosine variant *helps* by "
                f"{-delta:+.3f}.  Adopt cosine schedule for SS family.",
                "",
            ]
        else:
            md_lines += ["**Schedule fix outcome.** Inconclusive — "
                         "schedules are within ±0.05.", ""]

    # 2. Apples-to-apples spline-vs-narrow comparison
    md_lines += ["### Spline-vs-narrow comparison (same schedule)", ""]

    cosine_pair = (ss_cosine, narrow_cosine)
    orig_pair   = (ss_orig,   narrow_orig)

    def _emit_pair(name: str, ss: dict | None, narrow: dict | None):
        if ss is None or narrow is None:
            md_lines.append(
                f"- **{name}**: pending — "
                f"{('ss missing' if ss is None else 'narrow missing')}.")
            return None
        diff = ss["val_best"] - narrow["val_best"]
        verdict = ("spline is the lever" if diff < -0.05
                   else "narrow alone is enough" if diff > 0.05
                   else "tied (within 0.05)")
        md_lines.append(
            f"- **{name}**: ss_pa6 best={ss['val_best']:.4f}  vs  "
            f"narrow_relu2_h_d best={narrow['val_best']:.4f}  "
            f"(diff {diff:+.3f}) → **{verdict}**.")
        return diff

    diff_cos = _emit_pair("cosine schedule", *cosine_pair)
    diff_orig = _emit_pair("original schedule", *orig_pair)
    md_lines.append("")

    # 3. Recommendation
    md_lines += ["## Recommendation", ""]

    if narrow_orig is None:
        md_lines += [
            "Phase A2b (narrow_relu2_h_d ORIGINAL schedule) **still in flight**.",
            "Wait for it before deciding next H100 launches.",
            "",
            "Decision rule when it lands:",
            "- narrow_orig.best ≥ 4.80 → spline is the lever; greenlight ss_pa6_h_d",
            "- narrow_orig.best ≤ 4.70 → narrow alone competitive; pause RL-Spline-KV "
            "kernel work, pivot to GLU / wider base ablations.",
            "- 4.70 < narrow_orig.best < 4.80 → marginal; one more datapoint "
            "(spline_glu_b2_h_d) before committing.",
        ]
    else:
        nb = narrow_orig["val_best"]
        ss_best = ss_orig["val_best"] if ss_orig else float("nan")
        spline_lift = nb - ss_best  # positive = spline helps
        md_lines.append(
            f"narrow_relu2_h_d original best = {nb:.4f}  vs  "
            f"ss_pa6 original best = {ss_best:.4f}  "
            f"(spline_lift = {spline_lift:+.3f}).")
        if spline_lift >= 0.10:
            md_lines += [
                "→ **Spline is a real lever.**  Run ss_pa6_h_d "
                "(h=d, original schedule) next.  This is the v7 RL-Spline-KV "
                "base path width.",
            ]
        elif spline_lift >= 0.03:
            md_lines += [
                "→ **Spline helps, but margin is thin.**  Run ss_pa6_h_d "
                "to see if h=d closes more of the gap, then evaluate.",
            ]
        else:
            md_lines += [
                "→ **Spline does not justify its complexity.**  Pause RL-Spline-KV "
                "kernel work.  Pivot to:",
                "  - swiglu_h_d (gating-only baseline) or",
                "  - narrow_relu2_h_2d (wider-narrow baseline).",
            ]

    return "\n".join(md_lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "benchmark_runs" / "decision.md"))
    args = ap.parse_args()
    rows = annotate(collect_all())
    md = render_decision(rows)
    print(md)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main()
