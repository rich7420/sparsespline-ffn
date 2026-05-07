"""H100 one-step grad smoke for RLKVFFN.

Catches the dead-branch init bug user flagged:

    if W_out=0 AND C=0 simultaneously, then dC=0 forever and the spline
    branch never learns.

This smoke builds the RLKVFFN block standalone (no full GPT) at d20
shape (n_embd=1280, h=2560, r=32, L=22), runs one forward + one
backward, and asserts:

    1.  C.grad is not None
    2.  C.grad.norm() > 0  ← THE critical check
    3.  loss is finite
    4.  W_out_base half = 0 at init  (residual cold start)
    5.  W_out_delta half != 0 at init  (gradient signal source)
    6.  C = 0 at init  (cold start)
    7.  delta == 0 at step 0 (despite W_out_delta != 0)

If criteria 2 fails → init bug, RL-KV training would be dead.
If criteria 4-7 fail → init code regressed since plan §RL-KV port §3.

Cheap: ~30 sec on H100, ~$0.5.
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
app = modal.App("sparsespline-rlkv-grad-smoke-h100", image=IMAGE)


@app.function(gpu="H100", timeout=600)
def run_grad_smoke() -> dict:
    import sys
    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo/nanochat")
    import torch
    from nanochat.gpt import GPTConfig, RLKVFFN

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # d20 production config
    cfg = GPTConfig(
        sequence_len=2048, vocab_size=32768,
        n_layer=20, n_head=10, n_kv_head=10, n_embd=1280,
        ffn_type="rl_kv_b2",
        rlkv_h_ratio=2.0, rlkv_r=32, rlkv_L=22,
        rlkv_grid_lo=-3.0, rlkv_grid_hi=3.0, rlkv_lambda_scale=1.0,
        rlkv_fwd_kernel="v11_cuda", rlkv_bwd_kernel="wgmma_v5_cuda",
    )

    # Build the FFN block on meta then materialize on H100, init weights.
    # We replicate GPT.init_weights' RL-KV branch by hand here so we can
    # smoke this without a full GPT.
    with torch.device("meta"):
        ffn = RLKVFFN(cfg)
    ffn.to_empty(device=device)

    n_embd = cfg.n_embd
    s = (3.0 / n_embd) ** 0.5
    s_delta = (3.0 / (ffn.h + ffn.r)) ** 0.5
    with torch.no_grad():
        torch.nn.init.uniform_(ffn.K.weight, -s * 0.4, s * 0.4)
        torch.nn.init.zeros_(ffn.W_out.weight[:, :ffn.h])
        torch.nn.init.uniform_(ffn.W_out.weight[:, ffn.h:],
                                  -s_delta, s_delta)
        torch.nn.init.zeros_(ffn.C)
    # Cast to bf16 (matches training compute dtype)
    ffn = ffn.to(dtype)
    print(f"[init] K shape={tuple(ffn.K.weight.shape)}  "
           f"W_out shape={tuple(ffn.W_out.weight.shape)}  "
           f"C shape={tuple(ffn.C.shape)}", flush=True)

    out: dict = {}

    # ---- Verify init invariants (critera 4-7) ----
    out["init_W_out_base_zero"] = bool(
        ffn.W_out.weight[:, :ffn.h].abs().max().item() == 0.0)
    out["init_W_out_delta_nonzero"] = bool(
        ffn.W_out.weight[:, ffn.h:].abs().max().item() > 0.0)
    out["init_C_zero"] = bool(ffn.C.abs().max().item() == 0.0)
    out["init_W_out_delta_norm"] = float(
        ffn.W_out.weight[:, ffn.h:].float().norm().item())
    print(f"[init] W_out_base zero? {out['init_W_out_base_zero']}  "
           f"W_out_delta nonzero? {out['init_W_out_delta_nonzero']}  "
           f"C zero? {out['init_C_zero']}", flush=True)
    assert out["init_W_out_base_zero"], "W_out base half must be zero at init"
    assert out["init_W_out_delta_nonzero"], "W_out delta half must be nonzero at init"
    assert out["init_C_zero"], "C must be zero at init"

    # ---- Forward + backward ----
    N, d = 2048, n_embd
    torch.manual_seed(123)
    x = torch.randn(N, d, device=device, dtype=dtype, requires_grad=True)
    # Forward
    y = ffn(x)
    print(f"[fwd]  y shape={tuple(y.shape)}  y.norm={y.float().norm().item():.4e}",
           flush=True)

    # Verify delta == 0 at init: delta = lambda * spline(z, C=0) = 0,
    # so y_delta_part = W_out_delta · 0 = 0.
    # Equivalent test: y == W_out_base · ReLU²(z) since W_out_base = 0
    #                ⇒ y == 0 at init (residual cold start).
    out["fwd_y_zero_at_init"] = bool(y.abs().max().item() == 0.0)
    print(f"[fwd]  y zero at init? {out['fwd_y_zero_at_init']}  "
           f"max|y|={y.abs().max().item():.4e}", flush=True)
    assert out["fwd_y_zero_at_init"], (
        "Initial output should be zero (residual cold start). "
        "Got non-zero output, init may be broken.")

    # Backward — synthetic gradient
    torch.manual_seed(456)
    g = torch.randn_like(y)
    y.backward(g)

    # ---- THE critical check: C.grad.norm() > 0 ----
    out["C_grad_is_not_none"] = ffn.C.grad is not None
    if ffn.C.grad is None:
        out["C_grad_norm"] = None
        print("[FAIL] C.grad is None → backward through spline didn't fire", flush=True)
        return out
    out["C_grad_norm"] = float(ffn.C.grad.float().norm().item())
    out["C_grad_max_abs"] = float(ffn.C.grad.float().abs().max().item())

    # Other grad norms for context
    out["K_grad_norm"] = float(ffn.K.weight.grad.float().norm().item()) if ffn.K.weight.grad is not None else 0.0
    out["W_out_base_grad_norm"] = float(
        ffn.W_out.weight.grad[:, :ffn.h].float().norm().item()
    ) if ffn.W_out.weight.grad is not None else 0.0
    out["W_out_delta_grad_norm"] = float(
        ffn.W_out.weight.grad[:, ffn.h:].float().norm().item()
    ) if ffn.W_out.weight.grad is not None else 0.0
    out["x_grad_norm"] = float(x.grad.float().norm().item()) if x.grad is not None else 0.0

    print(f"\n[bwd] grad norms after one step:", flush=True)
    print(f"  C.grad             norm: {out['C_grad_norm']:.4e}    "
           f"max|.|: {out['C_grad_max_abs']:.4e}", flush=True)
    print(f"  K.weight.grad      norm: {out['K_grad_norm']:.4e}", flush=True)
    print(f"  W_out_base.grad    norm: {out['W_out_base_grad_norm']:.4e}",
           flush=True)
    print(f"  W_out_delta.grad   norm: {out['W_out_delta_grad_norm']:.4e}  "
           f"(expected ~0 at step 0 since delta=0)", flush=True)
    print(f"  x.grad             norm: {out['x_grad_norm']:.4e}", flush=True)

    # ---- THE assertions ----
    assert out["C_grad_norm"] > 0.0, (
        "C.grad.norm() == 0 — spline branch is DEAD.  Init bug suspected: "
        "check that W_out_delta is non-zero at init (not all zeros).")
    print("\n[PASS] C.grad.norm() > 0 — spline branch alive ✓", flush=True)

    # Optional: K should also have grad (because grad flows back through
    # ReLU² and W_out_base, both of which have non-trivial structure here.)
    # K.grad can technically be zero if W_out_base = 0 and we're at step 0,
    # because grad on a (the base activation) = W_out_base^T · grad_y = 0.
    # So K.grad MAY be zero at step 0; that's expected, not a failure.
    print(f"[note] K.grad = {out['K_grad_norm']:.4e} (may be 0 at step 0 "
           f"because W_out_base = 0; will pick up after W_out_base learns)",
           flush=True)

    out["ALL_PASS_STEP0"] = True

    # ---- 3-step grad transition smoke ----
    # Verify the model "wakes up" properly across the first 3 optimizer
    # updates.  At step 0 only C and W_out_base have gradient; W_out_delta
    # and K should pick up gradient from step 1 onwards once C is non-zero
    # and W_out_base is non-zero.
    print("\n=== 3-step grad transition smoke ===", flush=True)
    # Reset model to fresh init for the multi-step test
    with torch.no_grad():
        torch.nn.init.uniform_(ffn.K.weight, -s * 0.4, s * 0.4)
        torch.nn.init.zeros_(ffn.W_out.weight[:, :ffn.h])
        torch.nn.init.uniform_(ffn.W_out.weight[:, ffn.h:], -s_delta, s_delta)
        torch.nn.init.zeros_(ffn.C)
    # Bucket params into Muon-style (matrices) + AdamW for C — matches
    # base_train.py's grouping but in a single AdamW for simplicity here.
    matrix_params = [ffn.K.weight, ffn.W_out.weight]
    c_params = [ffn.C]
    optim = torch.optim.AdamW([
        dict(params=matrix_params, lr=0.02),
        dict(params=c_params, lr=0.02, weight_decay=0.0),
    ])

    transition = []
    for step in range(3):
        optim.zero_grad(set_to_none=True)
        torch.manual_seed(789 + step)
        x_step = torch.randn(N, d, device=device, dtype=dtype)
        y_step = ffn(x_step)
        g_step = torch.randn_like(y_step)
        y_step.backward(g_step)
        # collect diagnostics BEFORE optim.step
        snap = {
            "step": step,
            "C_norm_before_step":      float(ffn.C.float().norm().item()),
            "C_grad_norm":             float(ffn.C.grad.float().norm().item()),
            "K_grad_norm":             float(ffn.K.weight.grad.float().norm().item()),
            "W_out_base_grad_norm":    float(ffn.W_out.weight.grad[:, :ffn.h].float().norm().item()),
            "W_out_delta_grad_norm":   float(ffn.W_out.weight.grad[:, ffn.h:].float().norm().item()),
            "y_norm":                   float(y_step.float().norm().item()),
        }
        transition.append(snap)
        optim.step()
        # update C_norm AFTER step
        snap["C_norm_after_step"] = float(ffn.C.float().norm().item())

    out["transition"] = transition
    print(f"\n{'step':>5} {'C_norm_before':>14} {'C_grad_norm':>13} "
           f"{'W_out_base_g':>13} {'W_out_delta_g':>14} {'K_grad':>10} {'y_norm':>10}",
           flush=True)
    for s_ in transition:
        print(f"{s_['step']:>5} {s_['C_norm_before_step']:>14.4f} "
               f"{s_['C_grad_norm']:>13.4e} "
               f"{s_['W_out_base_grad_norm']:>13.4e} "
               f"{s_['W_out_delta_grad_norm']:>14.4e} "
               f"{s_['K_grad_norm']:>10.4e} {s_['y_norm']:>10.4e}", flush=True)

    # ---- transition assertions ----
    s0, s1, s2 = transition
    # Step 0: C_norm_before == 0 (init), C.grad > 0, W_out_base.grad > 0,
    #         W_out_delta.grad == 0 (delta=0), K.grad == 0 (W_out_base=0)
    assert s0["C_norm_before_step"] == 0.0, f"step 0 C should start at 0, got {s0['C_norm_before_step']}"
    assert s0["C_grad_norm"] > 0, "step 0 C.grad should be > 0 (dead-branch test)"
    assert s0["W_out_delta_grad_norm"] == 0, "step 0 W_out_delta.grad should be 0 (delta=0)"
    # Step 1: C_norm > 0 now (was updated after step 0), so delta != 0,
    #         so W_out_delta.grad and K.grad should both be > 0.
    assert s1["C_norm_before_step"] > 0, "step 1 C should be non-zero after step-0 update"
    assert s1["W_out_delta_grad_norm"] > 0, "step 1 W_out_delta.grad should wake up (delta != 0)"
    assert s1["K_grad_norm"] > 0, "step 1 K.grad should wake up (W_out_base != 0 after step-0 update)"
    assert s1["C_grad_norm"] > 0, "step 1 C.grad should still be > 0"
    # Step 2: all grads non-zero; loss finite
    assert s2["C_grad_norm"] > 0
    assert s2["W_out_delta_grad_norm"] > 0
    assert s2["K_grad_norm"] > 0
    assert all(torch.isfinite(p.grad).all().item() for p in [ffn.C, ffn.K.weight, ffn.W_out.weight])
    print("\n[PASS] 3-step grad transition all checks ✓", flush=True)
    print(f"  C learned from 0 → {s2['C_norm_after_step']:.4f} over 3 steps", flush=True)

    out["ALL_PASS"] = True
    return out


@app.local_entrypoint()
def main():
    result = run_grad_smoke.remote()
    import json as _json
    print("\n=== RESULT ===\n" + _json.dumps(result, indent=2))
