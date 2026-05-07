"""H100 kernel timing breakdown via torch.profiler.

Replaces nsys (which has GPU UUID issues in Modal containers).  Captures
per-kernel time for one fwd+bwd+optim step under three configs:

  1.  v1 fwd + v1 bwd  (production)
  2.  v11 fwd + v5 bwd (new fast path)
  3.  MLP h_4d         (cuBLAS only, reference)

Output: top-N kernels by self-CUDA-time, suitable for paper systems chapter.
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
        "numpy", "ninja", "pyarrow", "tokenizers",
        "tiktoken", "regex", "huggingface-hub",
    )
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
DATA_VOLUME = modal.Volume.from_name("sparsefuse-phase3-data",
                                       create_if_missing=False)
app = modal.App("sparsespline-torch-profile-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900,
              volumes={"/data": DATA_VOLUME})
def run_profile(cell_name: str) -> dict:
    import os, sys, json, time
    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo/nanochat")
    os.environ["NANOCHAT_BASE_DIR"] = "/data/nanochat"
    import torch
    from torch.profiler import profile, ProfilerActivity, schedule
    from nanochat_integration.nanochat_v41_redesign import build_model

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, T = 2, 1024
    n_layer, n_embd, n_head = 12, 768, 6
    vocab_size = 50304

    model, cell, selected = build_model(
        cell_name=cell_name, n_layer=n_layer, n_embd=n_embd, n_head=n_head,
        seq_len=T, vocab_size=vocab_size,
        use_kernel=True, device=device, dtype=torch.bfloat16,
    )
    idx = torch.randint(0, vocab_size, (B, T), device=device)
    targets = idx.clone()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    # Warmup (5 steps)
    for _ in range(5):
        optim.zero_grad()
        loss = model(idx, targets=targets)
        loss.backward()
        optim.step()
    torch.cuda.synchronize()

    # Wall-time measurement (median of 50)
    ts = []
    for _ in range(50):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        optim.zero_grad()
        loss = model(idx, targets=targets)
        loss.backward()
        optim.step()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    wall_ms = ts[len(ts) // 2]

    # Profiler — capture 3 measurement steps after schedule warmup.
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=2, active=3, repeat=1),
        record_shapes=False,
    ) as prof:
        for _ in range(7):  # 2 wait + 2 warmup + 3 active
            optim.zero_grad()
            loss = model(idx, targets=targets)
            loss.backward()
            optim.step()
            prof.step()
        torch.cuda.synchronize()

    # Top kernels by self-CUDA-time
    table = prof.key_averages().table(sort_by="self_cuda_time_total",
                                        row_limit=30)
    print(f"\n=== {cell_name} ===", flush=True)
    print(f"median step wall: {wall_ms:.4f} ms", flush=True)
    print(table, flush=True)

    # Extract structured top-15 for return JSON.
    # newer PyTorch (>=2.4) renamed self_cuda_time_total → self_device_time_total.
    avgs = prof.key_averages()
    rows = []
    for evt in avgs:
        if hasattr(evt, "self_device_time_total"):
            self_us = float(evt.self_device_time_total)
            total_us = float(evt.device_time_total)
        else:
            self_us = float(evt.self_cuda_time_total)
            total_us = float(evt.cuda_time_total)
        rows.append({
            "name": evt.key,
            "self_cuda_us": self_us,
            "total_cuda_us": total_us,
            "count": int(evt.count),
        })
    rows.sort(key=lambda r: r["self_cuda_us"], reverse=True)

    return {
        "cell": cell_name,
        "median_step_ms": wall_ms,
        "top_kernels": rows[:15],
    }


@app.local_entrypoint()
def main():
    cells = [
        "rl_kv_B2_r32_L22_wgmmaCUDA_h2_all12",     # v1 + v1
        "rl_kv_B2_r32_L22_v11fwd_v5bwd_h2_all12",  # v11 + v5
        "mlp_baseline",
    ]
    summary = {}
    for cell in cells:
        print(f"\n========== {cell} ==========\n", flush=True)
        out = run_profile.remote(cell)
        summary[cell] = {
            "median_step_ms": out["median_step_ms"],
            "top_kernels": out["top_kernels"],
        }
    import json as _json
    print("\n========== SUMMARY ==========\n", flush=True)
    print(_json.dumps(summary, indent=2), flush=True)
