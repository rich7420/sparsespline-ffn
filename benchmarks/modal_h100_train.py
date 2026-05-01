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
    tag: str = "",
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
    ]
    if use_kernel:
        cmd.append("--use-kernel")

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
    timeout=3600 * 2,  # 2 hr safety margin
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
    tag: str = "",
) -> str:
    return _train_cell(
        cell, steps, mb, seq_len, peak_lr, warmup_steps,
        eval_every, eval_batches, checkpoint_every, diag_every, use_kernel,
        tag=tag,
    )


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
    tag: str = "",
) -> None:
    label = f"{cell} (tag={tag})" if tag else cell
    print(f"Launching {label} training on H100  ({steps} steps, B={mb} T={seq_len}, "
          f"peak_lr={peak_lr}, warmup={warmup_steps})")
    out = run_train.remote(
        cell=cell, steps=steps, mb=mb, seq_len=seq_len, peak_lr=peak_lr,
        warmup_steps=warmup_steps, eval_every=eval_every,
        eval_batches=eval_batches, checkpoint_every=checkpoint_every,
        diag_every=diag_every, use_kernel=use_kernel, tag=tag,
    )
    print(out)
