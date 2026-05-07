"""H100 PyTorch profiler for RL-Spline-KV training step.

Reports per-op CUDA time so we can see which exact op is the bottleneck
after v3 + bf16 lands.  Compare against MLP for reference.
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy")
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-profile-h100", image=IMAGE)


@app.function(gpu="H100", timeout=900)
def run_profile(d: int = 768, b: int = 2, t: int = 1024) -> str:
    import sys, time, json, io
    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo/benchmarks")
    import torch, torch.nn as nn
    from torch.profiler import profile, ProfilerActivity, record_function

    from sparsespline_ffn import MLPFFN
    from ffn_full_compare import _RLKVWrap

    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: B={b} T={t} d={d} dtype=bf16\n")

    def build_wrap(name):
        if name == "mlp_h_4d":
            return MLPFFN(d=d, mlp_ratio=4)
        if name == "rl_kv_r32":
            return _RLKVWrap(d=d, r=32, use_kernel=True)
        if name == "rl_kv_r64":
            return _RLKVWrap(d=d, r=64, use_kernel=True)
        raise ValueError(name)

    out = io.StringIO()
    for name in ["mlp_h_4d", "rl_kv_r32"]:
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        model = build_wrap(name).to(device=device, dtype=dtype).train()
        target = torch.randn(b, t, d, device=device, dtype=dtype)
        x_const = torch.randn(b, t, d, device=device, dtype=dtype)

        # warmup (also triggers autotune)
        for _ in range(15):
            x = x_const.detach().requires_grad_(True)
            y = model(x)
            loss = (y - target).pow(2).sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # profile 5 steps
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                      record_shapes=False) as prof:
            for _ in range(5):
                model.zero_grad(set_to_none=True)
                with record_function("STEP"):
                    with record_function("fwd"):
                        x = x_const.detach().requires_grad_(True)
                        y = model(x)
                    loss = (y - target).pow(2).sum()
                    with record_function("bwd"):
                        loss.backward()

        out.write(f"\n{'='*100}\n")
        out.write(f"=== Profile: {name} (d={d}, B={b}, T={t}, bf16, H100) ===\n")
        out.write(f"{'='*100}\n")
        out.write(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20))
        out.write("\n")
        del model
        torch.cuda.empty_cache()

    text = out.getvalue()
    print(text)
    return text


@app.local_entrypoint()
def main(d: int = 768, b: int = 2, t: int = 1024) -> None:
    print(run_profile.remote(d=d, b=b, t=t))
