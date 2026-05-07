"""H100 NCU profile of fwd kernels — v1 (CUDA scalar) vs v10 (dense-W wgmma).

Follows Ubospica gist EXACTLY: debian_slim base, fresh nsight-compute via apt,
glob to find ncu binary.
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "gnupg", "git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "ninja")
    .run_commands(
        "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub "
        "| gpg --batch --no-tty --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] "
        "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /' "
        "> /etc/apt/sources.list.d/cuda.list",
        "apt-get update && (apt-get install -y nsight-compute-2026.1.0 || "
        "apt-get install -y nsight-compute-2025.2.0 || "
        "apt-get install -y nsight-compute-2025.1.0 || "
        "apt-get install -y nsight-compute || "
        "apt-get install -y --no-install-recommends $(apt-cache search nsight-compute | awk '{print $1}' | head -1))",
        "apt-get install -y cuda-toolkit-12-6 || apt-get install -y cuda-nvcc-12-6 || "
        "apt-get install -y nvidia-cuda-toolkit",
    )
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "nanochat/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
app = modal.App("sparsespline-ncu-fwd-profile-h100", image=IMAGE)


PROFILE_SCRIPT_FWD = r'''
import sys
sys.path.insert(0, "/repo/src")
import torch
from sparsespline_ffn.cuda_ext import spline_kv_fwd_cuda, spline_kv_fwd_v10_cuda

torch.manual_seed(0)
device = torch.device("cuda")
N, H, L, R = 2048, 768, 22, 32
G = L - 2

z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1

# Warmup
for _ in range(5):
    spline_kv_fwd_cuda(z, C, grid_lo=-3.0, grid_hi=3.0, G=G,
                        activation="relu_sq", lambda_scale=1.0)
    spline_kv_fwd_v10_cuda(z, C, grid_lo=-3.0, grid_hi=3.0, G=G,
                              activation="relu_sq", lambda_scale=1.0)
torch.cuda.synchronize()

# Profile target: alternate v1 and v10 a few times
for _ in range(3):
    spline_kv_fwd_cuda(z, C, grid_lo=-3.0, grid_hi=3.0, G=G,
                        activation="relu_sq", lambda_scale=1.0)
    spline_kv_fwd_v10_cuda(z, C, grid_lo=-3.0, grid_hi=3.0, G=G,
                              activation="relu_sq", lambda_scale=1.0)
torch.cuda.synchronize()
print("done")
'''


@app.function(gpu="H100", timeout=600)
def run_ncu_fwd() -> str:
    import subprocess, glob
    with open("/tmp/k_fwd.py", "w") as f:
        f.write(PROFILE_SCRIPT_FWD)

    ncu_candidates = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
    print(f"glob candidates: {ncu_candidates}", flush=True)
    if not ncu_candidates:
        for fallback in ["/usr/local/cuda/bin/ncu", "ncu"]:
            r = subprocess.run([fallback, "--version"], capture_output=True, text=True)
            if r.returncode == 0:
                ncu = fallback
                break
        else:
            return "NCU not found"
    else:
        ncu = ncu_candidates[-1]
    print(f"Using ncu: {ncu}", flush=True)

    v = subprocess.run([ncu, "--version"], capture_output=True, text=True)
    print(f"--- ncu version ---\n{v.stdout}\n{v.stderr}", flush=True)

    print("=== sanity: run script WITHOUT ncu ===", flush=True)
    sanity = subprocess.run(["python", "/tmp/k_fwd.py"],
                              capture_output=True, text=True, timeout=120)
    print(f"sanity rc: {sanity.returncode}; stdout: {sanity.stdout[:300]}", flush=True)
    if sanity.returncode != 0:
        return f"Script segfaults WITHOUT ncu.\nstdout:\n{sanity.stdout}\nstderr:\n{sanity.stderr}"

    print("=== ncu run ===", flush=True)
    cmd = [
        ncu,
        "--set", "basic",
        "--target-processes", "all",
        "--kernel-name", "regex:spline_kv_fwd_(delta_kernel|v10_kernel)",
        "--launch-count", "5",
        "python", "/tmp/k_fwd.py",
    ]
    print(f"CMD: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
    print("--- ncu stdout ---", flush=True)
    print(r.stdout[-4000:] if len(r.stdout) > 4000 else r.stdout, flush=True)
    print("--- ncu stderr ---", flush=True)
    print(r.stderr[-1500:] if len(r.stderr) > 1500 else r.stderr, flush=True)
    print(f"rc: {r.returncode}", flush=True)
    return r.stdout


@app.local_entrypoint()
def main():
    print(run_ncu_fwd.remote())
