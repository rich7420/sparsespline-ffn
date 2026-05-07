"""H100 NCU profile of FULL nanochat training step — RL-KV vs MLP head-to-head.

Captures top kernels per training step (fwd + bwd + optim).
Compares:
  - rl_kv_B2_r32_L22_wgmmaCUDA_h2_all12  (v10 fwd + v1 bwd)
  - mlp_baseline                          (cuBLAS only)

Uses --set basic on top kernels by time, no graph capture (ncu doesn't play
nice with graph mode).
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
        "apt-get install -y nsight-compute || true)",
        "apt-get install -y cuda-toolkit-12-6 || apt-get install -y cuda-nvcc-12-6 || "
        "apt-get install -y nvidia-cuda-toolkit",
    )
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
app = modal.App("sparsespline-ncu-full-step-h100", image=IMAGE)


# Profile script: build model, run 3 fwd+bwd+optim steps.
# Force v10 fwd by setting fwd_kernel="auto" — the autograd already has v10
# integrated (or we set use_kernel=True with R=32, L=22 → v10 dispatch kicks in).
PROFILE_SCRIPT_TEMPLATE = r"""
import os, sys
sys.path.insert(0, "/repo/src")
sys.path.insert(0, "/repo/nanochat")
os.environ["NANOCHAT_BASE_DIR"] = "/data/nanochat"
import torch
from nanochat_integration.nanochat_v41_redesign import build_model

torch.manual_seed(0)
device = torch.device("cuda")
B, T = 2, 1024
n_layer, n_embd, n_head = 12, 384, 6
vocab_size = 50304
cell_name = "{cell_name}"

model, cell, selected = build_model(
    cell_name=cell_name, n_layer=n_layer, n_embd=n_embd, n_head=n_head,
    seq_len=T, vocab_size=vocab_size,
    use_kernel=True, device=device, dtype=torch.bfloat16,
)
idx = torch.randint(0, vocab_size, (B, T), device=device)
targets = idx.clone()
optim = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            capturable=False, fused=True)

# Warmup
for _ in range(5):
    optim.zero_grad()
    loss = model(idx, targets=targets)
    loss.backward()
    optim.step()
torch.cuda.synchronize()

# ncu captures these
for _ in range(3):
    optim.zero_grad()
    loss = model(idx, targets=targets)
    loss.backward()
    optim.step()
torch.cuda.synchronize()
print("done")
"""


@app.function(gpu="H100", timeout=900,
                volumes={"/data": DATA_VOLUME})
def run_ncu_full_step(cell_name: str) -> dict:
    import subprocess, glob

    script = PROFILE_SCRIPT_TEMPLATE.format(cell_name=cell_name)
    with open("/tmp/k.py", "w") as f:
        f.write(script)

    ncu_candidates = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
    print(f"glob candidates: {ncu_candidates}", flush=True)
    if not ncu_candidates:
        for fallback in ["/usr/local/cuda/bin/ncu", "ncu"]:
            r = subprocess.run([fallback, "--version"], capture_output=True, text=True)
            if r.returncode == 0:
                ncu = fallback
                break
        else:
            return {"error": "NCU not found"}
    else:
        ncu = ncu_candidates[-1]
    print(f"Using ncu: {ncu}", flush=True)

    print("=== sanity (without ncu) ===", flush=True)
    sanity = subprocess.run(["python", "/tmp/k.py"],
                              capture_output=True, text=True, timeout=300)
    print(f"sanity rc: {sanity.returncode}", flush=True)
    print(f"stdout: {sanity.stdout[-1000:]}", flush=True)
    if sanity.returncode != 0:
        print(f"stderr: {sanity.stderr[-1500:]}", flush=True)
        return {"error": "sanity script failed", "stderr": sanity.stderr}

    # ncu — capture all kernels with --set basic, sorted by time at end.
    print(f"\n=== ncu profile of cell: {cell_name} ===", flush=True)
    cmd = [
        ncu,
        "--set", "basic",
        "--target-processes", "all",
        # Top-N kernels by time
        "--launch-count", "200",  # enough to capture most unique kernels
        "python", "/tmp/k.py",
    ]
    print(f"CMD: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print("--- ncu stdout (last 6000 chars) ---", flush=True)
    print(r.stdout[-6000:] if len(r.stdout) > 6000 else r.stdout, flush=True)
    if r.stderr:
        print("--- ncu stderr (last 1500 chars) ---", flush=True)
        print(r.stderr[-1500:], flush=True)
    print(f"rc: {r.returncode}", flush=True)
    return {"cell": cell_name, "stdout": r.stdout, "stderr": r.stderr, "rc": r.returncode}


@app.local_entrypoint()
def main():
    # Run RL-KV first (it's the more interesting case)
    print("\n========== RL-KV h2 (v10 fwd + v1 bwd) ==========\n")
    out_rl = run_ncu_full_step.remote("rl_kv_B2_r32_L22_wgmmaCUDA_h2_all12")

    print("\n========== MLP h_4d ==========\n")
    out_mlp = run_ncu_full_step.remote("mlp_baseline")

    print("\n========== summary RL-KV ==========\n")
    print(out_rl.get("stdout", "")[-3000:])
    print("\n========== summary MLP ==========\n")
    print(out_mlp.get("stdout", "")[-3000:])
