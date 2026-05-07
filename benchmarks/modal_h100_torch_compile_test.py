"""H100 torch.compile A/B test — does Inductor speed up RL-KV training?

Compares wall time of fwd+bwd+optim step with/without torch.compile.
Tests both architectures (RL-KV h2 + MLP h_4d).

Custom autograd Functions (FlashSplineFeature, FlashSplineDelta) typically
fall back to eager under compile, so we expect modest gains from compile
on RL-KV (mainly fusing elementwise ops between custom kernels) vs larger
gains on MLP (Inductor-fused matmul + activation fusion).
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
app = modal.App("sparsespline-torch-compile-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900,
                volumes={"/data": DATA_VOLUME})
def run_compile_test() -> str:
    import os, sys, time, json
    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo/nanochat")
    os.environ["NANOCHAT_BASE_DIR"] = "/data/nanochat"
    import torch
    from nanochat_integration.nanochat_v41_redesign import build_model

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, T = 2, 1024
    n_layer, n_embd, n_head = 12, 768, 6
    vocab_size = 50304

    cells = [
        ("rl_kv_B2_r32_L22_v10fwd_h2_all12", "rl_kv"),
        ("mlp_baseline",                       "mlp"),
    ]

    def time_step(model, optim, idx, targets, steps=50):
        # Warmup (also triggers torch.compile JIT compile)
        for _ in range(5):
            optim.zero_grad()
            loss = model(idx, targets=targets)
            loss.backward()
            optim.step()
        torch.cuda.synchronize()
        # Measure
        ts = []
        for _ in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            optim.zero_grad()
            loss = model(idx, targets=targets)
            loss.backward()
            optim.step()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        return {"median_ms": ts[len(ts)//2], "min_ms": ts[0], "max_ms": ts[-1]}

    out = {}
    for cell_name, label in cells:
        out[label] = {}
        print(f"\n=== cell: {cell_name} ===", flush=True)

        # ----- Baseline: no torch.compile -----
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, _, _ = build_model(
            cell_name=cell_name, n_layer=n_layer, n_embd=n_embd, n_head=n_head,
            seq_len=T, vocab_size=vocab_size,
            use_kernel=True, device=device, dtype=torch.bfloat16,
        )
        idx = torch.randint(0, vocab_size, (B, T), device=device)
        targets = idx.clone()
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

        t_baseline = time_step(model, optim, idx, targets)
        baseline_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        out[label]["baseline"] = {**t_baseline, "peak_mb": baseline_peak_mb}
        print(f"  baseline: {t_baseline}, peak={baseline_peak_mb:.0f} MB", flush=True)

        # Free memory before compile build
        del model, optim
        torch.cuda.empty_cache()

        # ----- With torch.compile (default mode) -----
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        model, _, _ = build_model(
            cell_name=cell_name, n_layer=n_layer, n_embd=n_embd, n_head=n_head,
            seq_len=T, vocab_size=vocab_size,
            use_kernel=True, device=device, dtype=torch.bfloat16,
        )
        try:
            model = torch.compile(model)
            optim = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
            t_compile = time_step(model, optim, idx, targets)
            compile_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            out[label]["compile_default"] = {
                **t_compile, "peak_mb": compile_peak_mb,
                "speedup": t_baseline["median_ms"] / t_compile["median_ms"],
            }
            print(f"  compile (default): {t_compile}, peak={compile_peak_mb:.0f} MB", flush=True)
            print(f"    speedup: {t_baseline['median_ms']/t_compile['median_ms']:.3f}×", flush=True)
        except Exception as e:
            out[label]["compile_default"] = {"error": str(e)[:300]}
            print(f"  compile failed: {e}", flush=True)

        del model, optim
        torch.cuda.empty_cache()

        # ----- With torch.compile (reduce-overhead, uses cuda graph) -----
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        model, _, _ = build_model(
            cell_name=cell_name, n_layer=n_layer, n_embd=n_embd, n_head=n_head,
            seq_len=T, vocab_size=vocab_size,
            use_kernel=True, device=device, dtype=torch.bfloat16,
        )
        try:
            model = torch.compile(model, mode="reduce-overhead")
            optim = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
            t_ro = time_step(model, optim, idx, targets)
            ro_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            out[label]["compile_reduce_overhead"] = {
                **t_ro, "peak_mb": ro_peak_mb,
                "speedup": t_baseline["median_ms"] / t_ro["median_ms"],
            }
            print(f"  compile (reduce-overhead): {t_ro}, peak={ro_peak_mb:.0f} MB", flush=True)
            print(f"    speedup: {t_baseline['median_ms']/t_ro['median_ms']:.3f}×", flush=True)
        except Exception as e:
            out[label]["compile_reduce_overhead"] = {"error": str(e)[:300]}
            print(f"  compile (reduce-overhead) failed: {e}", flush=True)

        del model, optim
        torch.cuda.empty_cache()

    print(json.dumps(out, indent=2), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main():
    print(run_compile_test.remote())
