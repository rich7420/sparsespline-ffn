"""Modal H100 training entrypoint for SimpleSpline / FullMix cells.

Mounts the existing ``sparsefuse-phase3-data`` volume which has:
  /nanochat/base_data_climbmix/  -- 5 ClimbMix parquet shards
  /nanochat/tokenizer/           -- pretrained nanochat tokenizer (vocab=64K)
  /nanochat/eval_bundle/         -- eval data

Each H100 function runs one nanochat training cell to a target step count
(default 50K = 100M tokens at B=2 T=1024) and dumps a JSON result back.

Usage:
    modal run benchmarks/modal_h100_train.py --cell ss_pa6 --steps 50000
    modal run benchmarks/modal_h100_train.py --cell ss_full
    modal run benchmarks/modal_h100_train.py::run_train --cell ss_pa6 --steps 50000
"""
from __future__ import annotations

import modal

# Mount the existing data volume (created earlier; already has shards + tokenizer).
DATA_VOLUME = modal.Volume.from_name("sparsefuse-phase3-data", create_if_missing=False)

# Image: same recipe as the bench app, plus pyarrow + huggingface tokenizers
# for the dataloader.
IMAGE = (
    # Need full CUDA toolchain (nvcc) to JIT-compile our .cu kernels.
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                                add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.9.1",
        "triton",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "numpy",
        "pytest",
        "pyarrow",
        "tokenizers",
        "tiktoken",
        "regex",
        "huggingface-hub",
        "ninja",  # required by torch.utils.cpp_extension.load
    )
    # Mount the local repo (ours; not nanochat).  The nanochat code lives in
    # nanochat/ subdir of the repo (vendored copy).
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[
            ".venv/**",
            ".git/**",
            "nanochat/.nanochat-runtime/**",
            "nanochat/.venv/**",
            "benchmark_runs/**",
            "**/__pycache__/**",
            "**/*.pyc",
        ],
        copy=True,
    )
    .run_commands(
        "cd /repo && pip install -e .",
        # nanochat itself isn't pip-installable; we run scripts from /repo/nanochat
        # with PYTHONPATH=/repo/nanochat.
    )
)

app = modal.App("sparsespline-h100-train", image=IMAGE)


def _train_cell(
    cell: str, steps: int, mb: int, seq_len: int, peak_lr: float,
    warmup_steps: int, eval_every: int, eval_batches: int,
    checkpoint_every: int, diag_every: int, use_kernel: bool,
    cuda_graph: bool, tag: str = "", seed: int = 0,
) -> str:
    import json
    import os
    import subprocess

    base_dir = "/data/nanochat"  # volume is mounted at /data
    cell_tag = f"{cell}_{tag}" if tag else cell
    out_json = f"/tmp/{cell_tag}_train.json"
    cmd = [
        "python", "/repo/nanochat/nanochat_integration/nanochat_v41_redesign.py",
        "--mode", cell,
        "--num-steps", str(steps),
        "--warmup-steps", str(warmup_steps),
        "--peak-lr", str(peak_lr),
        "--mb", str(mb),
        "--seq-len", str(seq_len),
        "--eval-every", str(eval_every),
        "--eval-batches", str(eval_batches),
        "--checkpoint-every", str(checkpoint_every),
        "--diag-every", str(diag_every),
        "--dump-json", out_json,
        "--seed", str(seed),
    ]
    if use_kernel:
        cmd.append("--use-kernel")
    if cuda_graph:
        cmd.append("--cuda-graph")

    env = {
        **os.environ,
        "PYTHONPATH": "/repo/nanochat:/repo/src",
        "NANOCHAT_BASE_DIR": base_dir,
    }
    proc = subprocess.run(
        cmd, cwd="/repo/nanochat", env=env,
        capture_output=True, text=True, check=False,
    )
    output = proc.stdout
    if proc.returncode != 0:
        output += "\n[STDERR]\n" + proc.stderr

    if os.path.exists(out_json):
        with open(out_json) as f:
            blob = f.read()
        # Save the JSON to the mounted volume too so we have persistent record
        os.makedirs("/data/nanochat/runs/v41_h100", exist_ok=True)
        persist_path = f"/data/nanochat/runs/v41_h100/{cell_tag}_train.json"
        with open(persist_path, "w") as f:
            f.write(blob)
        DATA_VOLUME.commit()
        output += f"\n[JSON saved to volume: {persist_path}]\n"
        output += "\n[JSON]\n" + blob

    return output


@app.function(
    gpu="H100",
    timeout=3600 * 4,  # 4 hr safety margin (800M at h_ratio=2 needs >2h with stable kernels)
    volumes={"/data": DATA_VOLUME},
)
def run_train(
    cell: str = "ss_pa6",
    steps: int = 50000,
    mb: int = 2,
    seq_len: int = 1024,
    peak_lr: float = 3e-4,
    warmup_steps: int = 500,
    eval_every: int = 2500,
    eval_batches: int = 20,
    checkpoint_every: int = 5000,
    diag_every: int = 100,
    use_kernel: bool = True,
    cuda_graph: bool = False,
    tag: str = "",
    seed: int = 0,
) -> str:
    return _train_cell(
        cell, steps, mb, seq_len, peak_lr, warmup_steps,
        eval_every, eval_batches, checkpoint_every, diag_every, use_kernel,
        cuda_graph=cuda_graph, tag=tag, seed=seed,
    )


@app.function(
    gpu="H100",
    timeout=3600 * 4,
    volumes={"/data": DATA_VOLUME},
)
def run_smoke_suite(
    steps: int = 500,
    mb: int = 2,
    seq_len: int = 1024,
    peak_lr: float = 3e-4,
    warmup_steps: int = 50,
    eval_every: int = 0,
    eval_batches: int = 20,
    checkpoint_every: int = 0,
    diag_every: int = 100,
    use_kernel: bool = True,
    cuda_graph: bool = True,
    tag: str = "graph500",
    all12: bool = False,
) -> str:
    import json
    import math
    import re

    def _finite(xs):
        return [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]

    def _series_stats(xs):
        vals = _finite(xs or [])
        if not vals:
            return None
        return {
            "first": vals[0],
            "last": vals[-1],
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "nonfinite_count": len(xs or []) - len(vals),
        }

    KEEP_KEYS = [
        "step", "grad_finite", "mean_C_norm", "mean_C_grad_norm",
        "mean_base_rms", "mean_delta_rms", "delta_over_base",
        "mean_y_base_rms", "mean_y_delta_rms", "mean_rho_delta",
        "mean_W_a_grad_norm", "mean_W_d_grad_norm",
        "mean_bin_entropy_norm", "mean_edge_bin_frac", "mean_active_frac",
    ]

    def _diag_snapshot(d: dict | None) -> dict | None:
        if not d:
            return None
        return {k: d.get(k) for k in KEEP_KEYS if k in d}

    def _tail_diag(row: dict) -> dict | None:
        hist = row.get("rl_kv_diag_history") or []
        if not hist:
            return None
        return _diag_snapshot(hist[-1])

    def _diag_trajectory(row: dict) -> dict | None:
        """Capture diag at first / 1Q / mid / 3Q / last steps so we can see C
        learning evolution, not just the last state. Each entry is a snapshot.
        """
        hist = row.get("rl_kv_diag_history") or []
        n = len(hist)
        if n == 0:
            return None
        idx = {
            "first": 0,
            "q1": max(0, n // 4),
            "mid": max(0, n // 2),
            "q3": max(0, (3 * n) // 4),
            "last": n - 1,
        }
        return {label: _diag_snapshot(hist[i]) for label, i in idx.items()}

    def _per_layer_last(row: dict) -> dict | None:
        """Per-layer values from the last diag entry. Tells us if any single
        layer's C is dead while others learn."""
        hist = row.get("rl_kv_diag_history") or []
        if not hist:
            return None
        d = hist[-1]
        keys = [
            "per_layer_C_norm", "per_layer_C_grad_norm",
            "per_layer_rho_delta", "per_layer_delta_rms",
            "per_layer_W_a_grad_norm", "per_layer_W_d_grad_norm",
            "per_layer_bin_entropy_norm", "per_layer_edge_bin_frac",
            "per_layer_active_frac",
        ]
        out: dict = {}
        for k in keys:
            v = d.get(k)
            if isinstance(v, list):
                vals = [float(x) for x in v
                        if isinstance(x, (int, float)) and math.isfinite(float(x))]
                if vals:
                    out[k] = {
                        "n": len(vals),
                        "min": min(vals),
                        "max": max(vals),
                        "mean": sum(vals) / len(vals),
                        "values": vals,
                    }
        return out or None

    def _spline_alive(diag: dict | None) -> dict:
        """Headline check: is the spline branch actually doing anything?
        We need ALL of these > 0 (not just the W_out_delta cols in step 0):
          - mean_C_norm                  : C parameter has moved off zero
          - mean_C_grad_norm              : C is currently receiving gradient
          - mean_delta_rms                : the spline output has magnitude
          - mean_rho_delta                : v7 paper threshold ≥ 0.20 for "spline contributing"
          - mean_W_d_grad_norm            : W_out delta cols learning too
        """
        if not diag:
            return {"verdict": "no_diag", "all_alive": False}
        checks = {
            "C_norm_moved":          diag.get("mean_C_norm", 0) > 1e-6,
            "C_grad_active":         diag.get("mean_C_grad_norm", 0) > 1e-6,
            "delta_has_signal":      diag.get("mean_delta_rms", 0) > 1e-6,
            "rho_delta_above_paper": diag.get("mean_rho_delta", 0) >= 0.20,
            "rho_delta_nonzero":     diag.get("mean_rho_delta", 0) > 1e-6,
            "W_delta_cols_learning": diag.get("mean_W_d_grad_norm", 0) > 1e-6,
        }
        return {
            "verdict": "ALIVE" if all(checks.values()) else (
                "PARTIAL" if any(checks.values()) else "DEAD"),
            "checks": checks,
        }

    def _loss_quarters(losses: list) -> dict | None:
        vals = _finite(losses or [])
        if len(vals) < 4:
            return None
        n = len(vals)
        q = {
            "q1_mean": sum(vals[: n // 4]) / max(1, n // 4),
            "q2_mean": sum(vals[n // 4 : n // 2]) / max(1, n // 4),
            "q3_mean": sum(vals[n // 2 : (3 * n) // 4]) / max(1, n // 4),
            "q4_mean": sum(vals[(3 * n) // 4 :]) / max(1, n - (3 * n) // 4),
        }
        q["drop_q1_to_q4"] = q["q1_mean"] - q["q4_mean"]
        return q

    def _summarize_run(row: dict, fallback_cell: str) -> dict:
        loss = row.get("loss") or {}
        losses = loss.get("all") or []
        grad_norm = row.get("grad_norm") or []
        update_rms = row.get("update_rms") or []
        vals = row.get("val_history") or []
        rank_post = row.get("rank_post")
        summary = {
            "mode": row.get("mode", fallback_cell),
            "steps": row.get("n_steps"),
            "cuda_graph": row.get("cuda_graph"),
            "use_kernel": row.get("use_kernel"),
            "wall_s": row.get("wall_s"),
            "peak_mb": row.get("peak_mb"),
            "params_m": row.get("params_m"),
            "loss": {
                "start": loss.get("start"),
                "end": loss.get("end"),
                "early_q": loss.get("early_q"),
                "late_q": loss.get("late_q"),
                "drop": loss.get("drop"),
                "series": _series_stats(losses),
            },
            "grad_norm": _series_stats(grad_norm),
            "update_rms": _series_stats(update_rms),
            "val": {
                "count": len(vals),
                "first": vals[0] if vals else None,
                "last": vals[-1] if vals else None,
                "best": min(vals, key=lambda x: x.get("val_loss", float("inf"))) if vals else None,
                "final_val_loss": row.get("final_val_loss"),
            },
            "rank_post": rank_post if rank_post and rank_post.get("K", 0) > 0 else None,
            "rl_kv_diag_last": _tail_diag(row),
            "rl_kv_diag_trajectory": _diag_trajectory(row),
            "rl_kv_per_layer_last": _per_layer_last(row),
            "rl_kv_alive": _spline_alive(_tail_diag(row)),
            "loss_quarters": _loss_quarters(losses),
        }
        if summary["wall_s"]:
            summary["tokens_per_s"] = steps * mb * seq_len / summary["wall_s"]
        return summary

    if all12:
        cells = [
            "mlp_baseline",
            "rl_kv_B2_r32_L22_wgmmaCUDA_all12",
            "rl_kv_B2_r32_L22_hopperCUDA_all12",
        ]
    else:
        cells = [
            "mlp_baseline",
            "rl_kv_B2_r32_L22_wgmmaCUDA",
            "rl_kv_B2_r32_L22_hopperCUDA",
        ]
    outputs = []
    rows = []
    for cell in cells:
        cell_tag = f"{tag}_{cell}"
        out = _train_cell(
            cell, steps, mb, seq_len, peak_lr, warmup_steps, eval_every,
            eval_batches, checkpoint_every, diag_every, use_kernel,
            cuda_graph=cuda_graph, tag=cell_tag,
        )
        shown = out.split("\n[JSON]\n", 1)[0]
        outputs.append(f"\n\n===== {cell} =====\n{shown}")
        match = re.search(r"\[JSON\]\n(\{.*\})", out, flags=re.S)
        if match:
            try:
                rows.append(_summarize_run(json.loads(match.group(1)), cell))
            except Exception as exc:
                rows.append({"mode": cell, "parse_error": str(exc)})
        else:
            rows.append({
                "mode": cell,
                "parse_error": "missing JSON block",
                "output_tail": shown[-4000:],
            })

    mlp_wall = next((r.get("wall_s") for r in rows
                     if r.get("mode") == "mlp_baseline"), None)
    mlp_tps = next((r.get("tokens_per_s") for r in rows
                    if r.get("mode") == "mlp_baseline"), None)
    mlp_peak = next((r.get("peak_mb") for r in rows
                     if r.get("mode") == "mlp_baseline"), None)
    mlp_finalval = next((r.get("val", {}).get("final_val_loss") for r in rows
                         if r.get("mode") == "mlp_baseline"), None)
    for row in rows:
        if mlp_wall and row.get("wall_s"):
            row["wall_vs_mlp"] = row["wall_s"] / mlp_wall
        if mlp_tps and row.get("tokens_per_s"):
            row["tokens_per_s_vs_mlp"] = row["tokens_per_s"] / mlp_tps
        if mlp_peak and row.get("peak_mb"):
            row["peak_mb_vs_mlp"] = row["peak_mb"] / mlp_peak
        fv = (row.get("val", {}) or {}).get("final_val_loss")
        if mlp_finalval is not None and fv is not None:
            row["val_loss_vs_mlp"] = fv - mlp_finalval

    # ============================================================
    # Headline table — easy human read of the 3 cells on key axes
    # ============================================================
    headline_lines = []
    headline_lines.append("=" * 110)
    headline_lines.append("HEADLINE  (priority 1 = `alive` ALIVE; priority 2 = vs MLP on speed/VRAM/quality)")
    headline_lines.append("=" * 110)
    fmt = ("{mode:<40} {alive:<8} {wall:>10} {wall_r:>9} "
           "{peak:>9} {peak_r:>9} {fval:>10} {fval_d:>10}")
    headline_lines.append(fmt.format(
        mode="cell", alive="alive?", wall="wall(s)", wall_r="vs MLP",
        peak="peak(MB)", peak_r="vs MLP", fval="finalVal", fval_d="vs MLP",
    ))
    headline_lines.append("-" * 110)
    for row in rows:
        alive = (row.get("rl_kv_alive") or {}).get("verdict", "—")
        wall = row.get("wall_s")
        wall_r = row.get("wall_vs_mlp")
        peak = row.get("peak_mb")
        peak_r = row.get("peak_mb_vs_mlp")
        fval = (row.get("val", {}) or {}).get("final_val_loss")
        fval_d = row.get("val_loss_vs_mlp")
        headline_lines.append(fmt.format(
            mode=str(row.get("mode", "?"))[:40],
            alive=str(alive),
            wall=f"{wall:.2f}" if wall else "—",
            wall_r=f"{wall_r:.3f}×" if wall_r else "—",
            peak=f"{peak:.0f}" if peak else "—",
            peak_r=f"{peak_r:.3f}×" if peak_r else "—",
            fval=f"{fval:.4f}" if fval is not None else "—",
            fval_d=f"{fval_d:+.4f}" if fval_d is not None else "—",
        ))
    headline_lines.append("=" * 110)

    # Diag trajectory table — see C_norm / rho_delta evolve
    headline_lines.append("")
    headline_lines.append("RL-KV DIAG TRAJECTORY  (mean across all RL-KV layers; want C_norm > 0 + rho_delta > 0)")
    headline_lines.append("=" * 110)
    diag_fmt = ("{cell:<38} {tag:<6} {step:>6} {Cn:>10} {Cg:>10} "
                "{drms:>10} {rho:>8} {Wdg:>10} {ent:>8}")
    headline_lines.append(diag_fmt.format(
        cell="cell", tag="when", step="step",
        Cn="C_norm", Cg="C_grad",
        drms="delta_rms", rho="rho_d",
        Wdg="W_d.grad", ent="ent_n",
    ))
    headline_lines.append("-" * 110)
    for row in rows:
        traj = row.get("rl_kv_diag_trajectory")
        if not traj:
            continue
        for tag in ["first", "mid", "last"]:
            d = traj.get(tag)
            if not d:
                continue
            headline_lines.append(diag_fmt.format(
                cell=str(row.get("mode", "?"))[:38],
                tag=tag, step=str(d.get("step", "—")),
                Cn=f"{d.get('mean_C_norm', 0):.4f}",
                Cg=f"{d.get('mean_C_grad_norm', 0):.4f}",
                drms=f"{d.get('mean_delta_rms', 0):.4e}",
                rho=f"{d.get('mean_rho_delta', 0):.4f}",
                Wdg=f"{d.get('mean_W_d_grad_norm', 0):.4f}",
                ent=f"{d.get('mean_bin_entropy_norm', 0):.3f}",
            ))
    headline_lines.append("=" * 110)

    summary = ("\n\n" + "\n".join(headline_lines)
               + "\n\n===== FULL JSON =====\n" + json.dumps(rows, indent=2))
    return "".join(outputs) + summary


@app.local_entrypoint()
def main(
    cell: str = "ss_pa6",
    steps: int = 50000,
    mb: int = 2,
    seq_len: int = 1024,
    peak_lr: float = 3e-4,
    warmup_steps: int = 500,
    eval_every: int = 2500,
    eval_batches: int = 20,
    checkpoint_every: int = 5000,
    diag_every: int = 100,
    use_kernel: bool = True,
    cuda_graph: bool = False,
    tag: str = "",
    suite: bool = False,
    all12: bool = False,
    seed: int = 0,
) -> None:
    if suite:
        print(f"Launching 500-step suite on H100  ({steps} steps, B={mb} T={seq_len}, "
              f"peak_lr={peak_lr}, warmup={warmup_steps}, cuda_graph={cuda_graph})")
        print(run_smoke_suite.remote(
            steps=steps, mb=mb, seq_len=seq_len, peak_lr=peak_lr,
            warmup_steps=warmup_steps, eval_every=eval_every,
            eval_batches=eval_batches, checkpoint_every=checkpoint_every,
            diag_every=diag_every, use_kernel=use_kernel,
            cuda_graph=cuda_graph, tag=tag or "graph500", all12=all12,
        ))
        return

    label = f"{cell} (tag={tag})" if tag else cell
    print(f"Launching {label} training on H100  ({steps} steps, B={mb} T={seq_len}, "
          f"peak_lr={peak_lr}, warmup={warmup_steps}, cuda_graph={cuda_graph})")
    out = run_train.remote(
        cell=cell, steps=steps, mb=mb, seq_len=seq_len, peak_lr=peak_lr,
        warmup_steps=warmup_steps, eval_every=eval_every,
        eval_batches=eval_batches, checkpoint_every=checkpoint_every,
        diag_every=diag_every, use_kernel=use_kernel,
        cuda_graph=cuda_graph, tag=tag, seed=seed,
    )
    print(out)
