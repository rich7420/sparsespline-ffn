"""H100 layer-by-layer profile: RL-KV vs MLP per-op breakdown.

Compares:
  1) MLP h_4d vanilla:       z = W_in @ x; a = ReLU²(z); y = W_out @ a
  2) RL-KV h2 with-base:     z = K @ x; f = FlashSplineFeature(z, C); y = W_out @ f
  3) RL-KV h2 multiplicative: z = K @ x; δ = FlashSplineDelta(z, C);
                              g = (1+P·δ) ⊙ ReLU²(z); y = W_out @ g

Per-op breakdown is via torch.profiler.  Also reports:
  - kernel-by-kernel ms (median over 50 reps)
  - GPU-only time vs Python-side overhead
  - peak SM occupancy
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
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-layer-profile-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_profile() -> str:
    import sys, time, json
    sys.path.insert(0, "/repo/src")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sparsespline_ffn.rl_spline_kv_reference import (
        RLSplineKVConfig, RLSplineKVReference,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")

    # Production shape (matches nanochat all12 cell)
    B, T, d = 2, 1024, 384
    h_mlp = 4 * d   # MLP h_4d
    h_rl = 2 * d    # RL-KV h2
    r = 32
    G = 20

    x = torch.randn(B, T, d, device=device, dtype=torch.bfloat16)

    # ----- Build the three FFN flavors -----
    class MLPh4d(nn.Module):
        def __init__(self):
            super().__init__()
            self.W_in = nn.Linear(d, h_mlp, bias=False)
            self.W_out = nn.Linear(h_mlp, d, bias=False)
            nn.init.uniform_(self.W_in.weight, -0.1, 0.1)
            nn.init.uniform_(self.W_out.weight, -0.1, 0.1)
        def forward(self, x):
            z = self.W_in(x)
            a = torch.where(z > 0, z * z, torch.zeros_like(z))
            return self.W_out(a)

    cfg_v7 = RLSplineKVConfig(
        d=d, h_ratio=2.0, r=r, G=G, spline_order=2,
        use_kernel=True, bwd_kernel="wgmma_cuda", fwd_kernel="auto",
        gating_mode="additive",
    )
    rl_v7 = RLSplineKVReference(cfg_v7).to(device).to(torch.bfloat16)

    cfg_v8 = RLSplineKVConfig(
        d=d, h_ratio=2.0, r=r, G=G, spline_order=2,
        use_kernel=True, bwd_kernel="wgmma_cuda", fwd_kernel="triton",
        gating_mode="multiplicative", c_init_std=0.01,
    )
    rl_v8 = RLSplineKVReference(cfg_v8).to(device).to(torch.bfloat16)

    mlp = MLPh4d().to(device).to(torch.bfloat16)

    # ----- Bench helpers -----
    def bench_fwd_only(fn, warmup=10, iters=50):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        return ts[len(ts) // 2]

    def bench_fwd_bwd(fn, warmup=10, iters=50):
        # fn must return loss
        for _ in range(warmup):
            loss = fn(); loss.backward()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = fn(); loss.backward()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        return ts[len(ts) // 2]

    out = {"shape": {"B": B, "T": T, "d": d, "h_mlp": h_mlp, "h_rl": h_rl, "r": r, "G": G}}

    # 1) Forward-only ms
    print("=== FORWARD-ONLY ms ===", flush=True)
    out["fwd_ms"] = {
        "mlp_h4d":         bench_fwd_only(lambda: mlp(x.detach())),
        "rl_v7_additive":  bench_fwd_only(lambda: rl_v7(x.detach())),
        "rl_v8_multiplicative": bench_fwd_only(lambda: rl_v8(x.detach())),
    }

    # 2) Forward + backward ms (matches training step inner loop)
    print("=== FORWARD+BACKWARD ms ===", flush=True)
    def fb(model):
        x_in = x.detach().clone().requires_grad_(True)
        model.zero_grad()
        return lambda: (model(x_in).pow(2).mean())
    out["fwd_bwd_ms"] = {
        "mlp_h4d":         bench_fwd_bwd(fb(mlp)),
        "rl_v7_additive":  bench_fwd_bwd(fb(rl_v7)),
        "rl_v8_multiplicative": bench_fwd_bwd(fb(rl_v8)),
    }

    # 3) Per-op profiler (forward + backward together, just one rep with profile)
    print("=== PROFILER per-op breakdown ===", flush=True)
    def profile_model(model, name):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(20):
                x_in = x.detach().clone().requires_grad_(True)
                model.zero_grad()
                loss = model(x_in).pow(2).mean()
                loss.backward()
            torch.cuda.synchronize()
        # Extract top ops by CUDA time, GPU-only
        events = prof.key_averages()
        rows = []
        for ev in events:
            if ev.device_time_total <= 0:
                continue
            rows.append({
                "op": ev.key,
                "cuda_time_ms_per_call": (ev.device_time_total / 1000.0) / max(1, ev.count),
                "cuda_time_pct": 0.0,  # filled below
                "calls": ev.count,
            })
        total_cuda = sum(r["cuda_time_ms_per_call"] * r["calls"] for r in rows)
        for r in rows:
            r["cuda_time_pct"] = (r["cuda_time_ms_per_call"] * r["calls"]) / total_cuda * 100
        rows.sort(key=lambda r: r["cuda_time_ms_per_call"] * r["calls"], reverse=True)
        return rows[:15]  # top 15 ops

    out["profile_mlp_h4d"]  = profile_model(mlp,  "mlp")
    out["profile_rl_v7"]    = profile_model(rl_v7, "rl_v7")
    out["profile_rl_v8"]    = profile_model(rl_v8, "rl_v8")

    # 4) Component-level micro-bench (RL-KV components in isolation)
    print("=== COMPONENT MICRO-BENCH ===", flush=True)
    h = h_rl
    L = G + 2
    z_t = torch.randn(B * T, h, device=device, dtype=torch.bfloat16)
    C_t = torch.randn(h, L, r, device=device, dtype=torch.bfloat16) * 0.1
    K_w = rl_v7.K.weight
    Wout_v7 = rl_v7.W_out.weight     # [d, h+r]
    Wout_v8 = rl_v8.W_out.weight     # [d, h]
    P_v8    = rl_v8.W_d_proj.weight if rl_v8.W_d_proj is not None else None  # [h, r]

    def bench(fn, warmup=10, iters=50):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort(); return ts[len(ts)//2]

    # Components (forward only)
    x_flat = x.reshape(-1, d)
    out["component_fwd_ms"] = {}

    # K matmul (h_rl) vs W_in (h_mlp)
    out["component_fwd_ms"]["K_matmul_h_rl"]   = bench(lambda: rl_v7.K(x_flat))
    out["component_fwd_ms"]["W_in_matmul_h_mlp"] = bench(lambda: mlp.W_in(x_flat.reshape(B, T, d)))
    out["component_fwd_ms"]["ReLU_sq"]         = bench(lambda: torch.where(z_t > 0, z_t * z_t, torch.zeros_like(z_t)))
    # Spline forward — Triton fast path
    from sparsespline_ffn.kernels.triton_flash_spline_feature import flash_spline_delta_forward_v4 as _delta_v4
    out["component_fwd_ms"]["spline_delta_triton"] = bench(
        lambda: _delta_v4(z_t.float(), C_t.float(), grid_lo=-3.0, grid_hi=3.0, G=G)
    )
    # Cat [a, λδ]
    delta_dummy = torch.randn(B*T, r, device=device, dtype=torch.bfloat16)
    out["component_fwd_ms"]["cat_a_lambda_delta"] = bench(
        lambda: torch.cat([z_t, delta_dummy], dim=-1)
    )
    # W_out matmul (v7 form: [d, h+r]) vs (v8 form: [d, h])
    f_v7 = torch.cat([z_t, delta_dummy], dim=-1)
    f_v8 = z_t.clone()
    out["component_fwd_ms"]["W_out_v7_h_plus_r"] = bench(lambda: F.linear(f_v7, Wout_v7))
    out["component_fwd_ms"]["W_out_v8_h_only"]   = bench(lambda: F.linear(f_v8, Wout_v8))
    # MLP W_out (h_mlp = 4d)
    a_mlp = torch.randn(B*T, h_mlp, device=device, dtype=torch.bfloat16)
    out["component_fwd_ms"]["W_out_mlp_h_mlp"]   = bench(lambda: mlp.W_out(a_mlp))
    # P projection (v8 only)
    if P_v8 is not None:
        out["component_fwd_ms"]["P_proj_r_to_h"] = bench(
            lambda: F.linear(delta_dummy, P_v8)
        )

    print(json.dumps(out, indent=2), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main():
    print(run_profile.remote())
