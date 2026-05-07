"""Test: does the ncu profile target script run standalone (without ncu)?

Diagnoses whether the SIGSEGV under ncu is from the script itself or
from ncu's instrumentation.
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
app = modal.App("sparsespline-ncu-test-h100", image=IMAGE)


@app.function(gpu="H100", timeout=600)
def run_test() -> str:
    import subprocess
    PROFILE_SCRIPT = """
import sys
sys.path.insert(0, '/repo/src')
import torch
print('torch ok')
from sparsespline_ffn.cuda_ext import spline_kv_bwd_wgmma_cuda, get_ext_v4
print('import ok')
ext = get_ext_v4()
print('ext jit ok')
torch.manual_seed(0)
device = torch.device('cuda')
N, H, L, R = 2048, 768, 22, 32
G = L - 2
z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1
g_delta = torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5
print('tensors ok')
out = spline_kv_bwd_wgmma_cuda(z, C, g_delta, -3.0, 3.0, G)
torch.cuda.synchronize()
print(f'kernel ok; dC.shape={out[0].shape} dz.shape={out[1].shape}')
"""
    with open("/tmp/k.py", "w") as f:
        f.write(PROFILE_SCRIPT)
    r = subprocess.run(["python", "/tmp/k.py"], capture_output=True, text=True, timeout=120)
    return f"STDOUT:\n{r.stdout}\n\nSTDERR:\n{r.stderr}\n\nrc={r.returncode}"


@app.local_entrypoint()
def main():
    print(run_test.remote())
