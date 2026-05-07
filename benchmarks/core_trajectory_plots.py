"""bpb-vs-step + CORE-vs-step trajectory plots for the d20 3-way comparison.

Implements P0-Parallel-2 from docs/PLAN_2026-05-04_neurips_experiment_queue.md.
Reads the JSON written by `benchmarks/extract_trajectories.py` and produces
side-by-side line plots with all 3 methods overlaid, annotated with the
warmdown moment (step 13280) and the Run A resume (step 12000).

Usage:
    .venv/bin/python benchmarks/core_trajectory_plots.py
    .venv/bin/python benchmarks/core_trajectory_plots.py \\
        --data benchmarks/data/trajectory_2026-05-04.json \\
        --out-png docs/_artifacts/core_trajectory_2026-05-04.png \\
        --out-pdf docs/_artifacts/core_trajectory_2026-05-04.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "benchmarks" / "data" / "trajectory_2026-05-04.json"
DEFAULT_PNG = ROOT / "docs" / "_artifacts" / "core_trajectory_2026-05-04.png"
DEFAULT_PDF = ROOT / "docs" / "_artifacts" / "core_trajectory_2026-05-04.pdf"

# Visual identity per method. Stable across plots and reports.
STYLES: dict[str, dict] = {
    "MLP V2": {"color": "#1f77b4", "marker": "o", "lw": 1.6},
    "Run C":  {"color": "#2ca02c", "marker": "s", "lw": 1.8},
    "Run A":  {"color": "#d62728", "marker": "^", "lw": 1.6},
    "Run A (pre-preempt continuation)": {
        "color": "#d62728", "marker": "x", "lw": 1.0, "ls": ":",
    },
}


def _style(name: str) -> dict:
    return STYLES.get(name, {"color": "0.4", "marker": ".", "lw": 1.2})


def annotate_phases(ax, warmdown_step: int, resume_step: int) -> None:
    """Vertical guides for warmdown begin and Run A resume."""
    y0, y1 = ax.get_ylim()
    ax.axvline(
        warmdown_step, color="0.5", lw=1.0, ls="--", alpha=0.7, zorder=1,
    )
    ax.text(
        warmdown_step, y1, " warmdown\n begins",
        color="0.35", fontsize=8, va="top", ha="left",
    )
    ax.axvline(
        resume_step, color="#d62728", lw=0.9, ls=":", alpha=0.55, zorder=1,
    )
    ax.text(
        resume_step, y0, " Run A resume\n (ckpt 12000)",
        color="#d62728", fontsize=8, va="bottom", ha="left", alpha=0.85,
    )
    ax.set_ylim(y0, y1)


def plot_bpb(ax, methods: dict, aux: dict, ann: dict) -> None:
    for name, s in methods.items():
        st = _style(name)
        ax.plot(
            s["bpb_steps"], s["bpb_values"],
            label=name, color=st["color"],
            marker=st["marker"], markersize=3.0,
            lw=st["lw"], alpha=0.95,
        )
    for name, s in aux.items():
        if not s["bpb_steps"]:
            continue
        st = _style(name)
        ax.plot(
            s["bpb_steps"], s["bpb_values"],
            label=name, color=st["color"],
            marker=st.get("marker", "x"), markersize=4.0,
            lw=st.get("lw", 1.0), ls=st.get("ls", ":"), alpha=0.85,
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("validation bpb")
    ax.set_title("(a) bpb vs step  (every 250 steps)")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    annotate_phases(ax, ann["warmdown_begin_step"], ann["runA_resume_step"])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)


def plot_bpb_zoom(ax, methods: dict, aux: dict, ann: dict, x_lo: int = 8000) -> None:
    """Linear-scale zoom on the second half of training (where the gap shows)."""
    for name, s in methods.items():
        st = _style(name)
        steps = [x for x in s["bpb_steps"] if x >= x_lo]
        vals = [v for x, v in zip(s["bpb_steps"], s["bpb_values"]) if x >= x_lo]
        ax.plot(
            steps, vals, label=name, color=st["color"],
            marker=st["marker"], markersize=3.5,
            lw=st["lw"], alpha=0.95,
        )
    for name, s in aux.items():
        steps = [x for x in s["bpb_steps"] if x >= x_lo]
        if not steps:
            continue
        vals = [v for x, v in zip(s["bpb_steps"], s["bpb_values"]) if x >= x_lo]
        st = _style(name)
        ax.plot(
            steps, vals, label=name, color=st["color"],
            marker=st.get("marker", "x"), markersize=5.0,
            lw=st.get("lw", 1.0), ls=st.get("ls", ":"), alpha=0.85,
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("validation bpb")
    ax.set_title(f"(b) bpb zoom  (steps {x_lo}–16600, linear)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_lo, 16700)
    annotate_phases(ax, ann["warmdown_begin_step"], ann["runA_resume_step"])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)


def plot_core(ax, methods: dict, ann: dict) -> None:
    for name, s in methods.items():
        st = _style(name)
        ax.plot(
            s["core_steps"], s["core_values"],
            label=name, color=st["color"],
            marker=st["marker"], markersize=6.0,
            lw=st["lw"] + 0.4, alpha=0.95,
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("CORE (mean centered acc., 22 base tasks)")
    ax.set_title("(c) CORE vs step  (every 2000 steps; 500/task at final)")
    ax.grid(True, alpha=0.3)
    annotate_phases(ax, ann["warmdown_begin_step"], ann["runA_resume_step"])
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--zoom-from", type=int, default=8000)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)

    raw = json.loads(args.data.read_text())
    methods = raw["methods"]
    aux = raw.get("auxiliary", {})
    ann = raw["annotations"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    plot_bpb(axes[0], methods, aux, ann)
    plot_bpb_zoom(axes[1], methods, aux, ann, x_lo=args.zoom_from)
    plot_core(axes[2], methods, ann)

    fig.suptitle(
        "d20 3-way: validation bpb (left/middle) and CORE (right) — "
        "Run C overtakes MLP on CORE during warmdown while bpb stays slightly worse",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=args.dpi)
    fig.savefig(args.out_pdf)
    print(f"Wrote: {args.out_png}")
    print(f"Wrote: {args.out_pdf}")

    # Print a short numerical summary used by the companion markdown.
    print()
    print("Decoupling summary (final step 16600):")
    bpb_final = {
        m: methods[m]["bpb_values"][-1] for m in methods if methods[m]["bpb_values"]
    }
    core_final = {
        m: methods[m]["core_values"][-1] for m in methods if methods[m]["core_values"]
    }
    for m in methods:
        print(
            f"  {m:8s}  bpb={bpb_final.get(m, float('nan')):.4f}  "
            f"CORE={core_final.get(m, float('nan')):.4f}"
        )
    if "MLP V2" in bpb_final and "Run C" in bpb_final:
        print(
            f"  Δ Run C − MLP   bpb={bpb_final['Run C']-bpb_final['MLP V2']:+.4f}  "
            f"CORE={core_final['Run C']-core_final['MLP V2']:+.4f}  "
            "(decoupling: bpb worse, CORE better)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
