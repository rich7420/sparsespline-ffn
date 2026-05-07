"""Run the V+C fusion / CUDA graphs benchmark on a Modal H100.

Usage (from the repo root, with modal in PATH):
    modal run benchmarks/modal_h100_bench.py
    modal run benchmarks/modal_h100_bench.py::run_h100 --B 4 --T 1024 --iters 200
    modal run benchmarks/modal_h100_bench.py::run_a100 --B 4 --T 512

The function picks up the local repo, installs it inside the Modal image,
runs ``benchmarks/v_c_fusion_bench.py`` on the requested GPU, and prints
the markdown table back to your terminal.

Output also dumped to /tmp/v_c_fusion_<GPU>.json on the remote, then
returned to the caller as a string.
"""
from __future__ import annotations

import modal

# Pick a relatively recent CUDA + torch.  The repo's local 3080 venv uses
# torch 2.9/2.11 + cu126; we mirror that for behavioural parity.  Triton
# 3.x ships with torch 2.5+ for sm90 support.
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.9.1",
        "triton",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install("numpy", "pytest")
    # Mount the local repo as /repo in the image and install editable.
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
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
    .run_commands("cd /repo && pip install -e .")
)

app = modal.App("sparsespline-vc-fusion-bench", image=IMAGE)


def _run_bench(b: int, t: int, iters: int, warmup: int, dtype: str,
               script: str = "v_c_fusion_bench.py") -> str:
    B, T = b, t
    import json
    import os
    import subprocess

    out_json = f"/tmp/{script.replace('.py', '')}.json"
    cmd = [
        "python", f"/repo/benchmarks/{script}",
        "--B", str(B), "--T", str(T),
        "--warmup", str(warmup), "--iters", str(iters),
        "--dtype", dtype,
        "--out-json", out_json,
    ]
    if script == "d_scaling_sweep.py":
        # Different arg name for the d sweep
        cmd[1] = f"/repo/benchmarks/{script}"
        # remove --B/--T inappropriate names; this script takes them too. OK.
    proc = subprocess.run(
        cmd, cwd="/repo", capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": "/repo/src"},
    )
    output = proc.stdout
    if proc.returncode != 0:
        output += "\n[STDERR]\n" + proc.stderr
    if os.path.exists(out_json):
        with open(out_json) as f:
            output += "\n[JSON]\n" + json.dumps(json.load(f), indent=2)
    return output


@app.function(gpu="H100", timeout=900)
def run_h100(b: int = 4, t: int = 512, iters: int = 100, warmup: int = 15,
             dtype: str = "bf16") -> str:
    return _run_bench(b, t, iters, warmup, dtype)


@app.function(gpu="H100", timeout=1800)
def run_h100_dsweep(b: int = 4, t: int = 512, iters: int = 50, warmup: int = 10,
                    dtype: str = "bf16") -> str:
    return _run_bench(b, t, iters, warmup, dtype, script="d_scaling_sweep.py")


@app.function(gpu="A100", timeout=900)
def run_a100(b: int = 4, t: int = 512, iters: int = 100, warmup: int = 15,
             dtype: str = "bf16") -> str:
    return _run_bench(b, t, iters, warmup, dtype)


@app.function(gpu="L40S", timeout=900)
def run_l40s(b: int = 4, t: int = 512, iters: int = 100, warmup: int = 15,
             dtype: str = "bf16") -> str:
    return _run_bench(b, t, iters, warmup, dtype)


@app.local_entrypoint()
def main(b: int = 4, t: int = 512, iters: int = 100, warmup: int = 15,
         dtype: str = "bf16", gpu: str = "H100", script: str = "vc") -> None:
    """Default: run on H100.

    --script vc       (default) v_c_fusion_bench.py
    --script dsweep   d_scaling_sweep.py
    --gpu H100 / A100 / L40S
    Note: modal CLI lowercases option names — use --b / --t.
    """
    if script == "dsweep" and gpu.upper() == "H100":
        fn = run_h100_dsweep
    else:
        fn = {"H100": run_h100, "A100": run_a100, "L40S": run_l40s}[gpu.upper()]
    print(f"Launching {script} bench on Modal {gpu.upper()} (b={b}, t={t}, iters={iters})...")
    output = fn.remote(b=b, t=t, iters=iters, warmup=warmup, dtype=dtype)
    print(output)
