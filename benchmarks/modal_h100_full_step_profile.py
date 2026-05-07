"""Kernel-by-kernel profile of one full nanochat training step.

Goal: identify which kernel(s) actually consume the wall in real nanochat
training (vs FFN-only microbench).  Builds a 12-layer GPT with RLKVAdapter
in NOBASE_all12 mode + cuda_graph + 1 captured step.  Reports top-30 CUDA
ops by total time.

This isolates: how much is FFN spline vs attention vs LM head vs framework.

Run:
  modal run benchmarks/modal_h100_full_step_profile.py
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                                add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "ninja", "pyarrow", "tokenizers", "tiktoken",
                  "regex", "huggingface-hub")
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/.venv/**",
                "nanochat/.nanochat-runtime/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("rlkv-full-step-profile", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
def run() -> str:
    import sys, io, time
    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo/nanochat")
    import torch, torch.nn as nn
    from torch.profiler import profile, ProfilerActivity, record_function

    out = io.StringIO()
    log = lambda s="": (out.write(s + "\n"), print(s, flush=True))

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"torch: {torch.__version__}")
    log("")

    # Build a full nanochat-style 12-layer GPT with NOBASE_all12 RL-KV
    from nanochat.gpt import GPT, GPTConfig
    from nanochat_integration.nanochat_v41_redesign import (
        replace_mlp_with_rl_kv, RL_KV_CELLS,
    )

    seq_len = 1024
    B = 2
    cfg = GPTConfig(
        sequence_len=seq_len, vocab_size=65536,
        n_layer=12, n_head=6, n_kv_head=6, n_embd=768,
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16

    with torch.device("meta"):
        model = GPT(cfg)
    model = model.to_empty(device=device)
    model.init_weights()
    model = model.to(dtype=dtype)

    # Replace all 12 layers with NOBASE RL-KV
    cell = RL_KV_CELLS["rl_kv_B2_r32_L22_NOBASE_all12"]
    selected = list(range(12))
    replace_mlp_with_rl_kv(
        model, selected_layers=selected,
        h_ratio=cell.h_ratio, r=cell.r, G=cell.G,
        spline_order=cell.spline_order,
        activation=cell.activation, lambda_scale=cell.lambda_scale,
        use_checkpoint=cell.use_checkpoint, use_kernel=True,
        bwd_kernel=cell.bwd_kernel, fwd_kernel=cell.fwd_kernel,
        no_base=cell.no_base,
    )
    model.train()

    # Static buffers for graph capture
    static_idx = torch.randint(0, 65536, (B, seq_len), device=device,
                                 dtype=torch.long)
    static_targets = torch.randint(0, 65536, (B, seq_len), device=device,
                                     dtype=torch.long)
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    # Warmup on side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(11):
            for p in model.parameters():
                p.grad.zero_()
            loss = model(static_idx, targets=static_targets)
            loss.backward()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Capture graph
    g = torch.cuda.CUDAGraph()
    for p in model.parameters():
        p.grad.zero_()
    with torch.cuda.graph(g):
        static_loss = model(static_idx, targets=static_targets)
        static_loss.backward()

    # Time the captured step (median of 30)
    import statistics
    samples = []
    for _ in range(30):
        for p in model.parameters():
            p.grad.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    log(f"Step wall (median over 30 graph replays): {statistics.median(samples):.3f} ms")
    log(f"Step wall (min): {min(samples):.3f} ms,  (max): {max(samples):.3f} ms")
    log("")

    # Profile 10 replays
    log("=" * 100)
    log("Per-op breakdown (over 10 graph replays)")
    log("=" * 100)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                  record_shapes=False) as prof:
        for _ in range(10):
            for p in model.parameters():
                p.grad.zero_()
            with record_function("STEP"):
                g.replay()
    log(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    return out.getvalue()


@app.local_entrypoint()
def main():
    print(run.remote())
