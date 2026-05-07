"""H100 FFN-block speed sweep across (batch, shape) — for paper systems chapter.

Sweeps:
  1. Batch sweep at production shape (d=768, h=2d, r=32, L=22) over B*T ∈
     {1024, 2048, 4096, 8192}.
  2. h_ratio sweep at B*T=2048 (h_ratio ∈ {1, 2}).  h_ratio=2 is production.
  3. r sweep at B*T=2048 (r ∈ {32, 64}).

For each config: time MLP h_4d, RL-KV v1+v1, RL-KV v11+v5 — one full
fwd+bwd+fused-AdamW step, median of 50 after 10 warmup.

Also reports peak VRAM per config to feed the memory table.
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
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-speed-sweep-h100", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
def run_sweep() -> dict:
    import sys, json, time
    sys.path.insert(0, "/repo/src")
    import torch
    from sparsespline_ffn import MLPFFN
    from sparsespline_ffn.rl_spline_kv_reference import (
        RLSplineKVConfig, RLSplineKVReference,
    )

    device = torch.device("cuda")

    def make_rl_kv(d, h_ratio, r, G, fwd_k, bwd_k):
        torch.manual_seed(42)
        cfg = RLSplineKVConfig(
            d=d, h_ratio=h_ratio, r=r, G=G,
            spline_order=2, lambda_scale=1.0,
            grid_lo=-3.0, grid_hi=3.0, activation="relu_sq",
            fwd_kernel=fwd_k, bwd_kernel=bwd_k, use_kernel=True,
        )
        return RLSplineKVReference(cfg).to(device).to(torch.bfloat16)

    def make_mlp(d, mlp_ratio=4):
        torch.manual_seed(42)
        return MLPFFN(d=d, mlp_ratio=mlp_ratio).to(device).to(torch.bfloat16)

    def step_ms_and_peak(ffn, x):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        opt = torch.optim.AdamW(ffn.parameters(), lr=3e-4, fused=True)
        # warmup
        for _ in range(10):
            opt.zero_grad()
            y = ffn(x)
            torch.manual_seed(2024)
            g = torch.randn_like(y)
            y.backward(g)
            opt.step()
        torch.cuda.synchronize()
        # measure
        ts = []
        for _ in range(50):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            opt.zero_grad()
            y = ffn(x)
            torch.manual_seed(2024)
            g = torch.randn_like(y)
            y.backward(g)
            opt.step()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return ts[len(ts) // 2], peak_mb

    out = {"production_d": 768, "configs": []}
    d = 768

    # NB v5 bwd needs N/N_PARTS/BLOCK_N ∈ {2,4,8} → N ∈ {1024, 2048, 4096}
    # for default N_PARTS=4, BLOCK_N=128.

    # --- Batch sweep ---
    for B in [1, 2, 4]:  # B*T = 1024, 2048, 4096
        T = 1024
        N = B * T
        if N not in (1024, 2048, 4096):
            continue
        torch.manual_seed(123)
        x = torch.randn(B, T, d, device=device, dtype=torch.bfloat16)

        for label, factory in [
            ("mlp_h_4d", lambda: make_mlp(d, 4)),
            ("rl_kv_v1+v1", lambda: make_rl_kv(d, 2.0, 32, 20, "triton", "wgmma_cuda")),
            ("rl_kv_v11+v5", lambda: make_rl_kv(d, 2.0, 32, 20, "v11_cuda", "wgmma_v5_cuda")),
        ]:
            try:
                ffn = factory()
                t_ms, peak_mb = step_ms_and_peak(ffn, x)
                row = {
                    "sweep": "batch", "B": B, "T": T, "N": N,
                    "h_ratio": 2.0, "r": 32, "G": 20,
                    "cell": label, "step_ms": t_ms, "peak_mb": peak_mb,
                }
                out["configs"].append(row)
                print(f"  [batch] B={B} T={T}  {label:15s}  step={t_ms:.4f} ms  peak={peak_mb:.1f} MB",
                       flush=True)
                del ffn
            except Exception as e:
                row = {"sweep": "batch", "B": B, "T": T, "cell": label, "error": str(e)[:200]}
                out["configs"].append(row)
                print(f"  [batch] B={B} T={T}  {label:15s}  ERROR: {e}", flush=True)
            torch.cuda.empty_cache()

    # --- Shape sweep at B*T=2048 ---
    B, T = 2, 1024
    torch.manual_seed(123)
    x = torch.randn(B, T, d, device=device, dtype=torch.bfloat16)
    for h_ratio in [1.0, 2.0]:
        for label, factory in [
            (f"rl_kv_v1+v1_h{int(h_ratio)}",
                lambda h=h_ratio: make_rl_kv(d, h, 32, 20, "triton", "wgmma_cuda")),
            (f"rl_kv_v11+v5_h{int(h_ratio)}",
                lambda h=h_ratio: make_rl_kv(d, h, 32, 20, "v11_cuda", "wgmma_v5_cuda")),
        ]:
            try:
                ffn = factory()
                t_ms, peak_mb = step_ms_and_peak(ffn, x)
                row = {
                    "sweep": "h_ratio", "B": B, "T": T,
                    "h_ratio": h_ratio, "r": 32, "G": 20,
                    "cell": label, "step_ms": t_ms, "peak_mb": peak_mb,
                }
                out["configs"].append(row)
                print(f"  [h_ratio] h={h_ratio}  {label:25s}  step={t_ms:.4f} ms  peak={peak_mb:.1f} MB",
                       flush=True)
                del ffn
            except Exception as e:
                row = {"sweep": "h_ratio", "h_ratio": h_ratio,
                        "cell": label, "error": str(e)[:200]}
                out["configs"].append(row)
                print(f"  [h_ratio] h={h_ratio}  {label}  ERROR: {e}", flush=True)
            torch.cuda.empty_cache()

    # --- r sweep at B*T=2048, h_ratio=2 ---
    for r in [32, 64]:
        for label, factory in [
            (f"rl_kv_v1+v1_r{r}",
                lambda rr=r: make_rl_kv(d, 2.0, rr, 20, "triton", "wgmma_cuda")),
            (f"rl_kv_v11+v5_r{r}",
                lambda rr=r: make_rl_kv(d, 2.0, rr, 20, "v11_cuda", "wgmma_v5_cuda")),
        ]:
            try:
                ffn = factory()
                t_ms, peak_mb = step_ms_and_peak(ffn, x)
                row = {
                    "sweep": "r", "B": B, "T": T,
                    "h_ratio": 2.0, "r": r, "G": 20,
                    "cell": label, "step_ms": t_ms, "peak_mb": peak_mb,
                }
                out["configs"].append(row)
                print(f"  [r] r={r}  {label:25s}  step={t_ms:.4f} ms  peak={peak_mb:.1f} MB",
                       flush=True)
                del ffn
            except Exception as e:
                row = {"sweep": "r", "r": r, "cell": label, "error": str(e)[:200]}
                out["configs"].append(row)
                print(f"  [r] r={r}  {label}  ERROR: {e}", flush=True)
            torch.cuda.empty_cache()

    print("\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main():
    print(run_sweep.remote())
