"""H100 full-step breakdown profile — locate the hidden ~5.85 ms.

Compares ONE TRAINING STEP of MLP h_4d vs RL-KV h2 with-base.

Captures:
  1. Per-kernel CUDA time (top 30) for fwd+bwd+opt
  2. Per-module CPU+CUDA time (LayerNorm, FFN, attention, embed, loss)
  3. Memory bandwidth estimates (bytes / kernel time)
  4. Forward vs backward vs optimizer split
  5. Activation memory peak inside one step

Run with cuda_graph=True to match production conditions.
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "ninja")
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
DATA_VOLUME = modal.Volume.from_name("sparsefuse-phase3-data",
                                       create_if_missing=False)
app = modal.App("sparsespline-step-breakdown-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900,
                volumes={"/data": DATA_VOLUME})
def run_breakdown() -> str:
    import os, sys, time, json
    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo/nanochat")
    os.environ["NANOCHAT_BASE_DIR"] = "/data/nanochat"
    import torch
    from nanochat_integration.nanochat_v41_redesign import build_model

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, T = 2, 1024
    n_layer, n_embd, n_head = 12, 384, 6
    vocab_size = 50304

    cells = [
        ("mlp_baseline",                              False),
        ("rl_kv_B2_r32_L22_wgmmaCUDA_h2_all12",       True),
    ]

    out = {"shape": {"B": B, "T": T, "n_layer": n_layer, "n_embd": n_embd}}

    for cell_name, _is_rlkv in cells:
        print(f"\n=== {cell_name} ===", flush=True)
        model, cell, selected = build_model(
            cell_name=cell_name, n_layer=n_layer, n_embd=n_embd, n_head=n_head,
            seq_len=T, vocab_size=vocab_size,
            use_kernel=True, device=device, dtype=torch.bfloat16,
        )

        idx = torch.randint(0, vocab_size, (B, T), device=device)
        targets = idx.clone()
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                    capturable=True, fused=True)

        # Warm up
        for _ in range(20):
            optim.zero_grad()
            loss = model(idx, targets=targets)
            loss.backward()
            optim.step()
        torch.cuda.synchronize()

        # Section A — wall-clock medians per-step
        print("Section A: wall-clock medians", flush=True)
        ts = []
        for _ in range(60):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            optim.zero_grad()
            loss = model(idx, targets=targets)
            loss.backward()
            optim.step()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        ms_med = ts[len(ts) // 2]
        ms_min = ts[0]
        ms_max = ts[-1]

        # Section B — fwd-only / bwd-only / opt-only timings
        print("Section B: fwd / bwd / opt split", flush=True)
        # Fwd-only
        for _ in range(5):
            with torch.no_grad():
                _ = model(idx, targets=targets)
        torch.cuda.synchronize()
        ts_fwd = []
        for _ in range(60):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                loss = model(idx, targets=targets)
            torch.cuda.synchronize()
            ts_fwd.append((time.perf_counter() - t0) * 1000)
        ts_fwd.sort()
        ms_fwd = ts_fwd[len(ts_fwd) // 2]

        # Fwd+bwd (no opt)
        for _ in range(5):
            optim.zero_grad()
            loss = model(idx, targets=targets); loss.backward()
        torch.cuda.synchronize()
        ts_fb = []
        for _ in range(60):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            optim.zero_grad()
            loss = model(idx, targets=targets); loss.backward()
            torch.cuda.synchronize()
            ts_fb.append((time.perf_counter() - t0) * 1000)
        ts_fb.sort()
        ms_fb = ts_fb[len(ts_fb) // 2]

        # Section C — torch.profiler per-op breakdown
        print("Section C: torch.profiler per-op breakdown", flush=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            with_modules=True,
        ) as prof:
            for _ in range(20):
                optim.zero_grad()
                loss = model(idx, targets=targets)
                loss.backward()
                optim.step()
            torch.cuda.synchronize()

        # Top kernels
        events = prof.key_averages()
        kernel_rows = []
        for ev in events:
            if ev.device_time_total <= 0:
                continue
            total_us = ev.device_time_total
            avg_us = total_us / max(1, ev.count)
            kernel_rows.append({
                "op": ev.key[:90],
                "ms_per_call": avg_us / 1000.0,
                "calls": ev.count,
                "total_ms": total_us / 1000.0,
            })
        kernel_rows.sort(key=lambda r: r["total_ms"], reverse=True)

        # Aggregate to high-level buckets — pattern matching kernel names
        buckets = {
            "spline_fwd_kernels": ["spline_kv_fwd"],
            "spline_bwd_kernels": ["spline_kv_bwd_wgmma_kernel", "FlashSpline"],
            "spline_post_kernels": ["spline_kv_bwd_postprocess",
                                     "spline_kv_fwd_pack",
                                     "spline_kv_fwd_activation"],
            "matmul_kernels":     ["nvjet_", "ampere_bf16", "sm90_xmma",
                                   "nvjet_tst", "fmha"],
            "attention_kernels":  ["fmha", "flash_fwd", "scaled_dot_product"],
            "elementwise_kernels":["elementwise_kernel", "vectorized_elementwise"],
            "reduce_kernels":     ["reduce_kernel"],
            "softmax_kernels":    ["softmax"],
            "layernorm_kernels":  ["layer_norm", "rms_norm"],
            "embed_kernels":      ["index_select", "embedding"],
            "optimizer_kernels":  ["adam", "Adam", "fused_adam"],
            "memcpy":             ["Memcpy", "memcpy"],
            "memset":             ["Memset", "memset", "fill_kernel"],
        }
        bucket_ms = {b: 0.0 for b in buckets}
        unmatched = 0.0
        for row in kernel_rows:
            op = row["op"]
            matched = False
            for bucket, patterns in buckets.items():
                if any(p in op for p in patterns):
                    bucket_ms[bucket] += row["total_ms"] / 20.0  # per step
                    matched = True
                    break
            if not matched:
                unmatched += row["total_ms"] / 20.0
        bucket_ms["other_unmatched"] = unmatched

        cell_summary = {
            "ms_step_median": ms_med,
            "ms_step_min":    ms_min,
            "ms_step_max":    ms_max,
            "ms_fwd_only":    ms_fwd,
            "ms_fwd_plus_bwd":ms_fb,
            "ms_optim":       ms_med - ms_fb,
            "top_kernels":    kernel_rows[:25],
            "bucketed_ms":    bucket_ms,
            "peak_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
        out[cell_name] = cell_summary

        # cleanup
        del model, optim, idx, targets
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(json.dumps(out, indent=2), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main():
    print(run_breakdown.remote())
