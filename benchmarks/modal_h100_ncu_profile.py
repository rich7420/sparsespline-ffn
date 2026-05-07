"""H100 NCU profile — follow Ubospica gist EXACTLY (use freshly-installed ncu).

Profiles spline_kv_bwd_wgmma_kernel<128, 8, 24, 32> (v1 production bwd).
"""
from __future__ import annotations

import modal


# Per Ubospica gist: debian_slim base + apt-install nsight-compute
# We additionally need nvcc for JIT, so we layer cuda-toolkit on top.
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "gnupg", "git", "build-essential")
    .pip_install("torch==2.9.1", "triton",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "ninja")
    .run_commands(
        # Add NVIDIA CUDA repo (debian12) — gist version
        "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub "
        "| gpg --batch --no-tty --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] "
        "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /' "
        "> /etc/apt/sources.list.d/cuda.list",
        # apt-get update + install nsight-compute (try multiple recent versions)
        "apt-get update && (apt-get install -y nsight-compute-2026.1.0 || "
        "apt-get install -y nsight-compute-2025.2.0 || "
        "apt-get install -y nsight-compute-2025.1.0 || "
        "apt-get install -y nsight-compute || "
        "apt-get install -y --no-install-recommends $(apt-cache search nsight-compute | awk '{print $1}' | head -1))",
        # Also need cuda-toolkit for nvcc (for JIT compile of our extension)
        "apt-get install -y cuda-toolkit-12-6 || apt-get install -y cuda-nvcc-12-6 || "
        "apt-get install -y nvidia-cuda-toolkit",
        # Verify ncu installed
        "ls -la /opt/nvidia/nsight-compute/ || echo 'nsight-compute dir not found'",
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
app = modal.App("sparsespline-ncu-profile-h100", image=IMAGE)


# Profile script — minimal, just runs the bwd kernel a few times.
PROFILE_SCRIPT = r'''
import sys
sys.path.insert(0, "/repo/src")
import torch
from sparsespline_ffn.cuda_ext import spline_kv_bwd_wgmma_cuda

torch.manual_seed(0)
device = torch.device("cuda")
N, H, L, R = 2048, 768, 22, 32
G = L - 2

z = torch.randn(N, H, device=device, dtype=torch.bfloat16) * 1.5
C = torch.randn(H, L, R, device=device, dtype=torch.bfloat16) * 0.1
g_delta = torch.randn(N, R, device=device, dtype=torch.bfloat16) * 0.5

# Warmup (force JIT compile + warm caches)
for _ in range(5):
    spline_kv_bwd_wgmma_cuda(z, C, g_delta, -3.0, 3.0, G)
torch.cuda.synchronize()

# THE measured call(s)
for _ in range(3):
    spline_kv_bwd_wgmma_cuda(z, C, g_delta, -3.0, 3.0, G)
torch.cuda.synchronize()
print("done")
'''


@app.function(gpu="H100", timeout=600)
def run_ncu() -> str:
    import subprocess, glob, os
    with open("/tmp/k.py", "w") as f:
        f.write(PROFILE_SCRIPT)

    # Per gist: find freshly-installed ncu via glob
    ncu_candidates = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
    print(f"glob candidates: {ncu_candidates}", flush=True)
    if not ncu_candidates:
        # Fallback: try standard locations
        for fallback in ["/usr/local/cuda/bin/ncu", "ncu"]:
            r = subprocess.run([fallback, "--version"], capture_output=True, text=True)
            if r.returncode == 0:
                ncu = fallback
                print(f"Fallback ncu: {fallback}", flush=True)
                break
        else:
            return "NCU not found anywhere"
    else:
        ncu = ncu_candidates[-1]
    print(f"Using ncu: {ncu}", flush=True)

    # Verify ncu version
    v = subprocess.run([ncu, "--version"], capture_output=True, text=True)
    print(f"--- ncu version ---\n{v.stdout}\n{v.stderr}", flush=True)

    # First sanity check: run script standalone (no ncu) to confirm it works
    print("=== sanity: run script WITHOUT ncu ===", flush=True)
    sanity = subprocess.run(["python", "/tmp/k.py"],
                              capture_output=True, text=True, timeout=120)
    print(f"sanity stdout: {sanity.stdout[:500]}", flush=True)
    print(f"sanity stderr: {sanity.stderr[:500]}", flush=True)
    print(f"sanity rc: {sanity.returncode}", flush=True)
    if sanity.returncode != 0:
        return f"Script segfaults WITHOUT ncu — extension issue, not ncu issue.\nstdout:\n{sanity.stdout}\nstderr:\n{sanity.stderr}"

    # Run ncu — exact gist-style invocation
    print("=== ncu run ===", flush=True)
    cmd = [
        ncu,
        "--set", "basic",
        "--target-processes", "all",
        "--kernel-name", "regex:spline_kv_bwd_wgmma_kernel",
        "--launch-count", "1",
        "python", "/tmp/k.py",
    ]
    print(f"CMD: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
    print("--- ncu stdout (last 3000 chars) ---", flush=True)
    print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout, flush=True)
    print("--- ncu stderr (last 1500 chars) ---", flush=True)
    print(r.stderr[-1500:] if len(r.stderr) > 1500 else r.stderr, flush=True)
    print(f"rc: {r.returncode}", flush=True)
    return r.stdout


@app.local_entrypoint()
def main():
    print(run_ncu.remote())
