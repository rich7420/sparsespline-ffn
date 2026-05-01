"""Run benchmarks/ffn_full_compare.py on Modal H100.

Same shape, dtype, variants as the local 3080 bench.  No training data
needed — all variants take random inputs.

Usage:
  modal run benchmarks/modal_h100_ffn_compare.py
  modal run benchmarks/modal_h100_ffn_compare.py --d 768 --bt 4096
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.9.1", "triton",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install("numpy")
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[
            ".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
            "**/__pycache__/**", "**/*.pyc",
        ],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-ffn-compare-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_compare(d: int = 768, b: int = 2, t: int = 1024,
                 dtype: str = "bf16") -> str:
    import sys, os, json, subprocess
    sys.path.insert(0, "/repo/src")
    out_json = "/tmp/ffn_compare_h100.json"
    cmd = [
        "python", "/repo/benchmarks/ffn_full_compare.py",
        "--d", str(d), "--B", str(b), "--T", str(t),
        "--dtype", dtype, "--device", "cuda",
        "--json-out", out_json,
    ]
    env = {**os.environ, "PYTHONPATH": "/repo/src"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    blob = ""
    if os.path.exists(out_json):
        blob = "\n[JSON]\n" + open(out_json).read()
    return proc.stdout + ("\n[STDERR]\n" + proc.stderr if proc.returncode else "") + blob


@app.local_entrypoint()
def main(d: int = 768, b: int = 2, t: int = 1024, dtype: str = "bf16") -> None:
    print(f"H100 FFN compare: d={d} B={b} T={t} dtype={dtype}")
    out = run_compare.remote(d=d, b=b, t=t, dtype=dtype)
    print(out)
