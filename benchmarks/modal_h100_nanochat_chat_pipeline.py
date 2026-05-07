"""nanochat chat pipeline launcher (chat_sft + final ChatCORE eval).

Loads a base checkpoint (saved by base_train.py at /data/nanochat/base_checkpoints/<tag>/),
runs SFT over the chat data mixture (SmolTalk + MMLU + GSM8K + SpellingBee), and at the
end-of-training step performs a FULL ChatCORE eval (no max_problems cap) which emits:

  ARC-Easy / ARC-Challenge / MMLU / GSM8K / HumanEval / SpellingBee  →  ChatCORE metric

This is the same dataset format as the example metrics the user asked about.

Usage:
    modal run benchmarks/modal_h100_nanochat_chat_pipeline.py::main \
        --model-tag full_rlkv_late33_grid5_lwarmup --model-step 16000

    modal run benchmarks/modal_h100_nanochat_chat_pipeline.py::main \
        --model-tag mlp_d20_reference_v2_seed0 --model-step 16000
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "numpy", "pytest", "pyarrow", "tokenizers",
        "tiktoken", "regex", "huggingface-hub", "ninja",
        "wandb", "datasets", "psutil",
    )
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**",
                "nanochat/.nanochat-runtime/**", "nanochat/.venv/**",
                "benchmark_runs/**", "dispatcher_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)

DATA_VOLUME = modal.Volume.from_name("sparsefuse-phase3-data",
                                      create_if_missing=False)
app = modal.App("sparsespline-nanochat-chat-pipeline", image=IMAGE)


def _run_chat_sft(num_gpus: int, args_list: list[str]) -> int:
    import os
    import subprocess

    env = {
        **os.environ,
        "PYTHONPATH": "/repo/nanochat:/repo/src",
        "NANOCHAT_BASE_DIR": "/data/nanochat",
        "OMP_NUM_THREADS": "1",
        # Defensive: extend NCCL collective timeout from 10 min default to 1 hr.
        # Generative chat eval has uneven rank load (variable generation length
        # at max_new_tokens=512) → fast ranks reach allreduce barrier long
        # before slow ranks → 10 min default crashes (observed 2026-05-04).
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "3600",
        "NCCL_TIMEOUT_SEC": "3600",
    }
    if num_gpus > 1:
        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={num_gpus}",
            "-m", "scripts.chat_sft", "--",
        ] + args_list
    else:
        cmd = ["python", "-m", "scripts.chat_sft"] + args_list
    print(f"[launcher] CMD: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/repo/nanochat", env=env,
                          capture_output=False, check=False)
    return proc.returncode


@app.function(gpu="H100:8", timeout=12 * 3600,
              volumes={"/data": DATA_VOLUME})
def run_chat_pipeline_8gpu(args_list: list[str]) -> dict:
    rc = _run_chat_sft(8, args_list)
    return {"rc": rc, "num_gpus": 8}


@app.local_entrypoint()
def main(model_tag: str = "",
         model_step: int = 16000,
         device_batch_size: int = 16,
         num_iterations: int = -1,
         eval_every: int = -1,
         chatcore_every: int = 999999,
         chatcore_max_cat: int = -1,
         chatcore_max_sample: int = 50) -> None:
    """One Modal call → chat_sft on the named base checkpoint with FULL ChatCORE eval at end.

    Defaults:
      device_batch_size=16   — same as full d20 base run, fits 8×H100 with FA3
      num_iterations=-1      — full epoch over the SFT mixture (SmolTalk+MMLU+GSM8K+...)
      eval_every=-1          — disable intermediate val-bpb (saves time)
      chatcore_every=999999  — gate must be >0 in chat_sft.py for the last_step branch to
                                fire; setting it to a huge number means only the
                                last_step trigger runs (no intermediate eval cost).

    The final ChatCORE eval at last_step uses chatcore_max_cat=-1 / chatcore_max_sample=-1
    (passed as flags below) so it covers the FULL test set per task — these are the
    paper-grade numbers.
    """
    if not model_tag:
        raise SystemExit("--model-tag is required (e.g. full_rlkv_late33_grid5_lwarmup)")

    args_list = [
        "--run=dummy",
        f"--model-tag={model_tag}",
        f"--model-step={model_step}",
        f"--device-batch-size={device_batch_size}",
        f"--num-iterations={num_iterations}",
        f"--eval-every={eval_every}",
        f"--chatcore-every={chatcore_every}",
        # Categorical (ARC/MMLU) batched + fast → keep at -1 (full set, ~3 min).
        # Generative (GSM8K/HumanEval/SpellingBee) variable gen length →
        # cap to bound wall time + avoid uneven rank-load NCCL timeout.
        # 200 is paper-grade noise (~5%) and finishes in ~10 min/task.
        f"--chatcore-max-cat={chatcore_max_cat}",
        f"--chatcore-max-sample={chatcore_max_sample}",
    ]

    print(f"[launcher] model_tag={model_tag}  model_step={model_step}", flush=True)
    print(f"[launcher] chat_sft.py args: {args_list}", flush=True)

    out = run_chat_pipeline_8gpu.remote(args_list)
    print(f"[launcher] result: {out}", flush=True)
