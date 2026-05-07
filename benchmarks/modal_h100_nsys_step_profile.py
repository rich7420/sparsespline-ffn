"""H100 nsys timeline profile of one full FFN training step.

Replaces the failing ncu probes — nsys is more reliable, gives a
top-N kernel time table that's good enough for the paper's systems chapter.

Captures kernel time breakdown for:
  1.  v1 fwd + v1 bwd  (production)
  2.  v11 fwd + v5 bwd (new fast path)
  3.  MLP h_4d         (cuBLAS only, reference)

For each: 5 warmup steps, then nsys profiles 3 measurement steps.
Output: top kernels sorted by total time, with grid/block/SMEM/registers.
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
        "apt-get update && apt-get install -y nsight-systems-2025.5.1 || "
        "apt-get install -y nsight-systems || true",
        "apt-get install -y cuda-toolkit-12-6 || apt-get install -y cuda-nvcc-12-6 || "
        "apt-get install -y nvidia-cuda-toolkit",
    )
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**", "benchmark_runs/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
DATA_VOLUME = modal.Volume.from_name("sparsefuse-phase3-data",
                                       create_if_missing=False)
app = modal.App("sparsespline-nsys-step-profile-h100", image=IMAGE)


PROFILE_SCRIPT = r"""
import os, sys, time
sys.path.insert(0, "/repo/src")
sys.path.insert(0, "/repo/nanochat")
os.environ["NANOCHAT_BASE_DIR"] = "/data/nanochat"
import torch
from nanochat_integration.nanochat_v41_redesign import build_model

torch.manual_seed(0)
device = torch.device("cuda")
B, T = 2, 1024
n_layer, n_embd, n_head = 12, 768, 6
vocab_size = 50304
cell_name = "{cell_name}"

model, cell, selected = build_model(
    cell_name=cell_name, n_layer=n_layer, n_embd=n_embd, n_head=n_head,
    seq_len=T, vocab_size=vocab_size,
    use_kernel=True, device=device, dtype=torch.bfloat16,
)
idx = torch.randint(0, vocab_size, (B, T), device=device)
targets = idx.clone()
optim = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

# Warmup
for _ in range(5):
    optim.zero_grad()
    loss = model(idx, targets=targets)
    loss.backward()
    optim.step()
torch.cuda.synchronize()

# Measurement region (nsys captures these 3 steps)
torch.cuda.cudart().cudaProfilerStart()
for _ in range(3):
    optim.zero_grad()
    loss = model(idx, targets=targets)
    loss.backward()
    optim.step()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("done")
"""


@app.function(gpu="H100", timeout=900,
              volumes={"/data": DATA_VOLUME})
def run_nsys(cell_name: str) -> dict:
    import subprocess, glob, json

    script_path = "/tmp/k.py"
    with open(script_path, "w") as f:
        f.write(PROFILE_SCRIPT.format(cell_name=cell_name))

    # find nsys binary
    candidates = sorted(glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys"))
    if not candidates:
        for fb in ["/usr/local/cuda/bin/nsys", "nsys"]:
            r = subprocess.run([fb, "--version"], capture_output=True, text=True)
            if r.returncode == 0:
                candidates = [fb]
                break
    if not candidates:
        return {"error": "nsys not found"}
    nsys = candidates[-1]
    print(f"Using nsys: {nsys}", flush=True)

    out_dir = "/tmp/nsys_out"
    os_makedirs(out_dir)
    out_prefix = f"{out_dir}/{cell_name}"

    # nsys profile: capture cuda + nvtx, output sqlite for stats.
    cmd = [
        nsys, "profile",
        "-o", out_prefix,
        "--capture-range=cudaProfilerApi",
        "-t", "cuda",
        "--export=sqlite",
        "--force-overwrite=true",
        "python", script_path,
    ]
    print(f"\n=== profile cell={cell_name} ===", flush=True)
    print(f"CMD: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(f"rc: {r.returncode}", flush=True)
    if r.stdout:
        print(f"stdout: {r.stdout[-2000:]}", flush=True)
    if r.stderr:
        print(f"stderr: {r.stderr[-2000:]}", flush=True)

    # Use nsys stats to extract top kernels
    sqlite = out_prefix + ".sqlite"
    if not (subprocess.run(["test", "-f", sqlite]).returncode == 0):
        print(f"sqlite missing: {sqlite}", flush=True)
        return {"cell": cell_name, "rc": r.returncode}
    stats_cmd = [nsys, "stats",
                 "--report", "cuda_gpu_kern_sum",
                 "--format", "csv",
                 sqlite]
    s = subprocess.run(stats_cmd, capture_output=True, text=True, timeout=300)
    print(f"\n=== top kernels (cell={cell_name}) ===", flush=True)
    print(s.stdout[-6000:] if len(s.stdout) > 6000 else s.stdout, flush=True)

    return {"cell": cell_name, "rc": r.returncode, "kernel_table": s.stdout}


def os_makedirs(p):
    import os
    os.makedirs(p, exist_ok=True)


@app.local_entrypoint()
def main():
    cells = [
        "rl_kv_B2_r32_L22_wgmmaCUDA_h2_all12",     # v1 + v1
        "rl_kv_B2_r32_L22_v11fwd_v5bwd_h2_all12",  # v11 + v5 (new)
        "mlp_baseline",
    ]
    for cell in cells:
        print(f"\n========== {cell} ==========\n", flush=True)
        out = run_nsys.remote(cell)
        print(f"finished: {cell}, rc={out.get('rc')}", flush=True)
