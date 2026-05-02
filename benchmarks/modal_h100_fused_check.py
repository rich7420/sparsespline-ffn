from __future__ import annotations

import modal


IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.9.1",
        "triton",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install("ninja")
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)

app = modal.App("sparsespline-h100-fused-check", image=IMAGE)


@app.function(gpu="H100", timeout=1800)
def check() -> str:
    import io
    import sys

    sys.path.insert(0, "/repo/src")
    import torch

    out = io.StringIO()

    def log(s: str = "") -> None:
        print(s, flush=True)
        out.write(s + "\n")

    from sparsespline_ffn.cuda_ext import spline_kv_bwd_wgmma_cuda
    from sparsespline_ffn.rl_spline_kv_reference import (
        RLSplineKVConfig,
        RLSplineKVReference,
    )

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    for N, H, L, R, G in [(128, 64, 22, 32, 20), (2048, 768, 22, 32, 20)]:
        log(f"shape N={N} H={H} L={L} R={R}")
        z = torch.randn(N, H, device=device, dtype=dtype)
        C = torch.randn(H, L, R, device=device, dtype=dtype) * 0.01
        g_delta = torch.randn(N, R, device=device, dtype=dtype)
        g_a = torch.randn(N, H, device=device, dtype=dtype)

        dC_old_f32, dz_spline_f32 = spline_kv_bwd_wgmma_cuda(
            z, C, g_delta, -3.0, 3.0, G,
        )
        dz_base = g_a * ((2.0 * z) * (z > 0).to(z.dtype))
        dz_old = (dz_base + dz_spline_f32).to(dtype)
        dC_old = dC_old_f32.to(dtype)

        dC_new, dz_new = spline_kv_bwd_wgmma_cuda(
            z, C, g_delta, -3.0, 3.0, G,
            g_a=g_a, activation="relu_sq", fused_post=True,
        )
        torch.cuda.synchronize()

        for name, old, new in [("dC", dC_old, dC_new), ("dz", dz_old, dz_new)]:
            finite = bool(torch.isfinite(new).all().item())
            max_abs = float((old.float() - new.float()).abs().max().item())
            mean_abs = float((old.float() - new.float()).abs().mean().item())
            log(f"  {name}: finite={finite} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}")
        log("")

    log("autograd end-to-end C.grad check")
    for use_kernel, bwd_kernel in [
        (False, "reference"),
        (True, "hopper_cuda"),
        (True, "wgmma_cuda"),
    ]:
        torch.manual_seed(123)
        cfg = RLSplineKVConfig(
            d=128,
            h_ratio=1.0,
            r=32,
            G=20,
            spline_order=2,
            use_kernel=use_kernel,
            bwd_kernel=bwd_kernel if use_kernel else "triton",
            init_C_zero=True,
        )
        m = RLSplineKVReference(cfg).to(device=device, dtype=dtype).train()
        x = torch.randn(2, 64, cfg.d, device=device, dtype=dtype)
        target = torch.randn_like(x)
        y = m(x)
        loss = (y - target).float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()

        w_delta_norm = float(m.W_out.weight[:, m.h:].detach().float().norm().item())
        c_grad_norm = float(m.C.grad.detach().float().norm().item()) if m.C.grad is not None else -1.0
        c_grad_max = float(m.C.grad.detach().float().abs().max().item()) if m.C.grad is not None else -1.0
        k_grad_norm = float(m.K.weight.grad.detach().float().norm().item()) if m.K.weight.grad is not None else -1.0
        log(
            f"  path={bwd_kernel:<11} use_kernel={use_kernel} "
            f"loss={float(loss.item()):.6f} W_delta_norm={w_delta_norm:.6e} "
            f"C_grad_norm={c_grad_norm:.6e} C_grad_max={c_grad_max:.6e} "
            f"K_grad_norm={k_grad_norm:.6e}"
        )
    return out.getvalue()


@app.local_entrypoint()
def main() -> None:
    print(check.remote())
