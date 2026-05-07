"""Fetch full metrics from saved 100M train JSONs on Modal volume."""
from __future__ import annotations

import modal


IMAGE = modal.Image.debian_slim(python_version="3.12").pip_install("numpy")
app = modal.App("sparsespline-fetch-results", image=IMAGE)
volume = modal.Volume.from_name("sparsefuse-phase3-data", create_if_missing=False)


@app.function(volumes={"/data": volume}, timeout=120)
def fetch() -> str:
    import json, os
    base = "/data/nanochat/runs/v41_h100"
    if not os.path.isdir(base):
        return f"NO RUNS DIR at {base}"

    # Filter to current sweep + relevant baselines
    keep_substrings = [
        "100M_h2_lr",            # LR sweep
        "100M_h25",              # h_ratio=2.5
        "100M_h3",               # h_ratio=3
        "100M_withbase_h2",      # baseline h_ratio=2 lr=3e-4
        "100M_withbase_r32",     # baseline h_ratio=1
        "100M_nobase_h2",        # NOBASE h_ratio=2
        "100M_paper",            # NOBASE r=64 paper run
        "narrow_relu2",          # MLP baselines (if present)
        "ss_pa6",                # SimpleSpline MLP
    ]

    rows = []
    for fn in sorted(os.listdir(base)):
        if not fn.endswith("_train.json"):
            continue
        if not any(s in fn for s in keep_substrings):
            continue
        path = os.path.join(base, fn)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            rows.append({"file": fn, "error": str(e)})
            continue

        # Top-level fields (try several possible key names)
        wall = data.get("wall") or data.get("wall_seconds") or data.get("elapsed_s")
        peak = data.get("peak_mb") or data.get("peak_memory_mb") or data.get("max_memory_mb")
        n_steps = data.get("num_steps") or data.get("steps") or data.get("n_steps")
        peak_lr = data.get("peak_lr") or data.get("lr")
        warmup = data.get("warmup_steps") or data.get("warmup")
        cell_name = data.get("cell") or data.get("cell_name")
        tag = data.get("tag")

        # Derived: tokens trained = n_steps * mb * seq_len
        mb = data.get("mb") or 2
        seq_len = data.get("seq_len") or 1024
        tokens = (n_steps * mb * seq_len) if n_steps else None

        # Val history — first/middle/last few snapshots
        val_history = data.get("val_history", [])
        loss_history = data.get("losses", [])

        # Last diag (full row)
        diag_history = data.get("rl_kv_diag_history", [])
        last_diag = diag_history[-1] if diag_history else {}

        rows.append({
            "file": fn,
            "cell": cell_name,
            "tag": tag,
            "n_steps": n_steps,
            "tokens_trained": tokens,
            "peak_lr": peak_lr,
            "warmup": warmup,
            "wall_sec": wall,
            "peak_mb": peak,
            "tok_per_sec": (tokens / wall) if (tokens and wall) else None,
            "ms_per_step": (wall * 1000.0 / n_steps) if (wall and n_steps) else None,
            "final_val_loss": data.get("final_val_loss"),
            "val_at_5k":  next((v["val_loss"] for v in val_history if v.get("step") == 5000),  None),
            "val_at_25k": next((v["val_loss"] for v in val_history if v.get("step") == 25000), None),
            "val_at_45k": next((v["val_loss"] for v in val_history if v.get("step") == 45000), None),
            # Diag at end of training (45k or 47.5k)
            "diag_step": last_diag.get("step"),
            "mean_C_norm":           last_diag.get("mean_C_norm"),
            "mean_C_grad_norm":      last_diag.get("mean_C_grad_norm"),
            "delta_over_base":       last_diag.get("delta_over_base"),
            "mean_y_delta_rms":      last_diag.get("mean_y_delta_rms"),
            "mean_y_base_rms":       last_diag.get("mean_y_base_rms"),
            "mean_W_a_grad_norm":    last_diag.get("mean_W_a_grad_norm"),
            "mean_W_d_grad_norm":    last_diag.get("mean_W_d_grad_norm"),
            "mean_rho_delta":        last_diag.get("mean_rho_delta"),
            "mean_bin_entropy_norm": last_diag.get("mean_bin_entropy_norm"),
            "mean_edge_bin_frac":    last_diag.get("mean_edge_bin_frac"),
            "mean_active_frac":      last_diag.get("mean_active_frac"),
        })

    return json.dumps(rows, indent=2, default=str)


@app.local_entrypoint()
def main():
    print(fetch.remote())
