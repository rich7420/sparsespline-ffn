"""H100 full-FFN forward+backward parity test.

Why this exists
---------------
The δ-half microbench shows v11 fwd has 0.5 % rel_err vs v1 — but the
actual transformer signal is `y = W_out · [a; λδ]`, where W_out is bf16.
The bf16 cast at W_out's output may wash out fp16-W's small precision gains.
We want to measure the END-TO-END FFN output difference, not just δ.

Tests:
  - Full FFN forward output (y = W_out · [a; λδ]) — v1 vs v11 vs reference
  - Full FFN backward gradients (dz, dC, dW_out) — v1+v1 vs v11+v5 vs ref
  - Gradient L2 norms per parameter (W_out, C, etc.) — same seed, run 1 step
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
app = modal.App("sparsespline-full-ffn-parity-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_probe() -> dict:
    import sys, json, time
    sys.path.insert(0, "/repo/src")
    import torch

    from sparsespline_ffn.rl_spline_kv_reference import RLSplineKVConfig, RLSplineKVReference as RLSplineKVFFN

    torch.manual_seed(0)
    device = torch.device("cuda")

    # Production shape (RL-KV h_ratio=2, r=32, G=20 → L=22)
    d = 768
    def make_cfg(fwd_k: str, bwd_k: str):
        return RLSplineKVConfig(
            d=d, h_ratio=2.0, r=32, G=20,
            spline_order=2, lambda_scale=1.0,
            grid_lo=-3.0, grid_hi=3.0,
            activation="relu_sq",
            fwd_kernel=fwd_k, bwd_kernel=bwd_k,
            use_kernel=True,
        )
    # NB v5 bwd needs N/N_PARTS/BLOCK_N ∈ {2,4,8} → for N_PARTS=4 BLOCK_N=128
    # we need N ∈ {1024, 2048, 4096}.  Pick 2048 to match nanochat production.
    N, T = 2, 1024
    out: dict = {}

    def stat(name: str, ours: torch.Tensor, ref: torch.Tensor) -> dict:
        ours_f = ours.float(); ref_f = ref.float()
        diff = ours_f - ref_f
        diff_abs = diff.abs()
        ref_max = ref_f.abs().max().item()
        return {
            "label": name,
            "max_abs_err": float(diff_abs.max().item()),
            "max_rel_err": float(diff_abs.max().item() / (ref_max + 1e-9)),
            "mean_abs_err": float(diff_abs.mean().item()),
            "mean_signed_err": float(diff.mean().item()),
        }

    # Build same model 3 times with different kernel configs ----------------
    def build_model(fwd_kernel: str, bwd_kernel: str):
        torch.manual_seed(42)
        cfg = make_cfg(fwd_kernel, bwd_kernel)
        ffn = RLSplineKVFFN(cfg).to(device).to(torch.bfloat16)
        return ffn

    ffn_v1     = build_model("triton",   "wgmma_cuda")
    ffn_v11v1  = build_model("v11_cuda", "wgmma_cuda")
    ffn_v11v5  = build_model("v11_cuda", "wgmma_v5_cuda")

    # Pre-conditions: same parameters across all 3 (seed 42 + same dtypes)
    p_v1   = list(ffn_v1.parameters())
    p_v11v1 = list(ffn_v11v1.parameters())
    p_v11v5 = list(ffn_v11v5.parameters())
    assert len(p_v1) == len(p_v11v1) == len(p_v11v5)
    for a, b, c in zip(p_v1, p_v11v1, p_v11v5):
        assert torch.equal(a, b) and torch.equal(b, c), "params differ at init"
    print("[init] all 3 models share identical params", flush=True)

    # ---- FORWARD parity ----
    print("\n=== forward output y = ffn(x) ===", flush=True)
    torch.manual_seed(123)
    # RLSplineKVReference takes 2-D input [N*T, d] (not [N, T, d]); use BT*d shape.
    x = torch.randn(N * T, d, device=device, dtype=torch.bfloat16)

    y_v1   = ffn_v1(x)
    y_v11v1 = ffn_v11v1(x)
    y_v11v5 = ffn_v11v5(x)

    out["fwd_y_v11v1_vs_v1"] = stat("y v11+v1 vs v1+v1", y_v11v1, y_v1)
    out["fwd_y_v11v5_vs_v1"] = stat("y v11+v5 vs v1+v1", y_v11v5, y_v1)
    for k in ["fwd_y_v11v1_vs_v1", "fwd_y_v11v5_vs_v1"]:
        v = out[k]
        print(f"  {k:25s}: signed={v['mean_signed_err']:+.3e} "
               f"max_abs={v['max_abs_err']:.3e} "
               f"max_rel={v['max_rel_err']:.3e} "
               f"mean_abs={v['mean_abs_err']:.3e}", flush=True)

    # ---- BACKWARD parity (dz, dW_out, dC) ----
    print("\n=== backward gradients ===", flush=True)

    def grads_for(ffn, x):
        ffn.zero_grad(set_to_none=True)
        x_t = x.detach().clone().requires_grad_(True)
        y = ffn(x_t)
        # Synthetic upstream gradient
        torch.manual_seed(2024)
        g = torch.randn_like(y)
        y.backward(g)
        # Collect named grads
        grads = {}
        grads["dx"] = x_t.grad.detach().clone()
        for name, p in ffn.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.detach().clone()
        return grads

    g_v1   = grads_for(ffn_v1, x)
    g_v11v1 = grads_for(ffn_v11v1, x)
    g_v11v5 = grads_for(ffn_v11v5, x)

    print(f"\nparam names: {list(g_v1.keys())}", flush=True)
    for name in g_v1.keys():
        if name not in g_v11v1 or name not in g_v11v5:
            continue
        s_v1   = stat(f"d{name} v11+v1 vs v1+v1", g_v11v1[name], g_v1[name])
        s_v5   = stat(f"d{name} v11+v5 vs v1+v1", g_v11v5[name], g_v1[name])
        out[f"grad_{name}_v11v1"] = s_v1
        out[f"grad_{name}_v11v5"] = s_v5
        print(f"  d{name:18s} v11v1: signed={s_v1['mean_signed_err']:+.3e} "
               f"max_abs={s_v1['max_abs_err']:.3e} mean_abs={s_v1['mean_abs_err']:.3e}",
               flush=True)
        print(f"  d{name:18s} v11v5: signed={s_v5['mean_signed_err']:+.3e} "
               f"max_abs={s_v5['max_abs_err']:.3e} mean_abs={s_v5['mean_abs_err']:.3e}",
               flush=True)

    # ---- per-param L2 norms (gradient signal magnitude) ----
    print("\n=== gradient L2 norms (per param) ===", flush=True)
    norm_table = {}
    for name in g_v1.keys():
        if name not in g_v11v1 or name not in g_v11v5:
            continue
        n_v1   = float(g_v1[name].float().norm().item())
        n_v11v1 = float(g_v11v1[name].float().norm().item())
        n_v11v5 = float(g_v11v5[name].float().norm().item())
        norm_table[name] = {
            "v1+v1": n_v1, "v11+v1": n_v11v1, "v11+v5": n_v11v5,
            "v11v1_div_v1": n_v11v1 / (n_v1 + 1e-12),
            "v11v5_div_v1": n_v11v5 / (n_v1 + 1e-12),
        }
        print(f"  {name:22s}: v1={n_v1:.4e}  v11+v1={n_v11v1:.4e} "
               f"({n_v11v1/(n_v1+1e-12):.4f})  "
               f"v11+v5={n_v11v5:.4e} ({n_v11v5/(n_v1+1e-12):.4f})",
               flush=True)
    out["norm_table"] = norm_table

    # ---- speed (median wall ms for one fwd+bwd+optim step) ----
    print("\n=== speed: one fwd+bwd+opt step (median of 50, after 10 warmup) ===",
           flush=True)

    def step_ms(ffn, x):
        opt = torch.optim.AdamW(ffn.parameters(), lr=3e-4, fused=True)
        for _ in range(10):
            opt.zero_grad()
            y = ffn(x)
            torch.manual_seed(2024)
            g = torch.randn_like(y)
            y.backward(g)
            opt.step()
        torch.cuda.synchronize()
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
        ts.sort(); return ts[len(ts) // 2]

    # rebuild fresh models for clean speed measurements
    ffn_v1   = build_model("triton",   "wgmma_cuda")
    t_v1   = step_ms(ffn_v1, x)
    ffn_v11v1 = build_model("v11_cuda", "wgmma_cuda")
    t_v11v1 = step_ms(ffn_v11v1, x)
    ffn_v11v5 = build_model("v11_cuda", "wgmma_v5_cuda")
    t_v11v5 = step_ms(ffn_v11v5, x)

    # Also build an MLP h_4d FFN as the reference baseline.
    from sparsespline_ffn import MLPFFN
    torch.manual_seed(42)
    mlp = MLPFFN(d=d, mlp_ratio=4).to(device).to(torch.bfloat16)
    t_mlp = step_ms(mlp, x)

    out["speed"] = {
        "v1+v1_ms": t_v1, "v11+v1_ms": t_v11v1, "v11+v5_ms": t_v11v5,
        "mlp_h_4d_ms": t_mlp,
        "v11v1_speedup": t_v1 / t_v11v1,
        "v11v5_speedup": t_v1 / t_v11v5,
        "v11v5_vs_mlp": t_v11v5 / t_mlp,
        "v1v1_vs_mlp":  t_v1   / t_mlp,
    }
    print(f"  v1+v1:     {t_v1:.4f} ms", flush=True)
    print(f"  v11+v1:    {t_v11v1:.4f} ms  ({t_v1/t_v11v1:.3f}x v1+v1)", flush=True)
    print(f"  v11+v5:    {t_v11v5:.4f} ms  ({t_v1/t_v11v5:.3f}x v1+v1)", flush=True)
    print(f"  mlp_h_4d:  {t_mlp:.4f} ms  (RL-KV v11+v5 / MLP = {t_v11v5/t_mlp:.3f}x)",
           flush=True)

    print("\nFINAL JSON:", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main():
    print(run_probe.remote())
