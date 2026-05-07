"""Modal CPU job — download ClimbMix-400B shards onto our Modal volume.

Wraps nanochat's stock `python -m nanochat.dataset -n N` which:
  - downloads train shards 0..N-1
  - always also downloads validation shard MAX_SHARD (6542)
  - skips files that already exist (idempotent)
  - uses 4 parallel workers + retry-with-backoff

For our d20 8.7 B-token target with ~46 M tokens/shard we need ≥190
shards for a clean 1-epoch pass.  Using N=240 matches Karpathy's
nanochat#1 reference exactly — gives ~11.04 B tokens (1.27× our target,
clean margin).

Existing shards on the volume: 5 (00000-00003 + 06542).  After this job
the volume should have 240 train shards (00000-00239) + 06542 val shard
= 241 total parquet files, ~22 GB.

Cost: ~5-10 min wall on Modal CPU (no GPU), ~$0.05.

Usage:
    modal run benchmarks/modal_download_climbmix.py
    modal run benchmarks/modal_download_climbmix.py --num-shards 200
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # CPU-only torch — `nanochat.common.get_base_dir` imports torch
    # transitively even though the download script doesn't use the GPU.
    .pip_install(
        "torch",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install("requests", "pyarrow", "tiktoken", "tokenizers")
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn/nanochat",
        remote_path="/repo/nanochat",
        ignore=[".venv/**", "__pycache__/**", "*.pyc",
                ".nanochat-runtime/**"],
        copy=True,
    )
)
DATA_VOLUME = modal.Volume.from_name("sparsefuse-phase3-data",
                                       create_if_missing=False)
app = modal.App("sparsespline-climbmix-download", image=IMAGE)


@app.function(
    timeout=3 * 3600,            # 3h cap, expected ~10 min
    cpu=4,                        # 4 cores for parallel download workers
    volumes={"/data": DATA_VOLUME},
    # No gpu needed; this is purely network + disk
)
def download_shards(num_shards: int = 240, num_workers: int = 4) -> dict:
    import os, subprocess, glob

    env = {
        **os.environ,
        "PYTHONPATH": "/repo/nanochat",
        "NANOCHAT_BASE_DIR": "/data/nanochat",
    }
    cm_dir = "/data/nanochat/base_data_climbmix"
    before = sorted(glob.glob(f"{cm_dir}/*.parquet")) if os.path.isdir(cm_dir) else []
    print(f"[before] {len(before)} parquet files in {cm_dir}", flush=True)
    for f in before[:5]:
        print(f"  {os.path.basename(f)}", flush=True)
    if len(before) > 5:
        print(f"  ... +{len(before) - 5} more", flush=True)

    cmd = [
        "python", "-m", "nanochat.dataset",
        "-n", str(num_shards),
        "-w", str(num_workers),
    ]
    print(f"\nCMD: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/repo/nanochat", env=env,
                            capture_output=False, check=False)

    after = sorted(glob.glob(f"{cm_dir}/*.parquet"))
    delta = set(after) - set(before)
    total_bytes = sum(os.path.getsize(f) for f in after)
    print(f"\n[after]  {len(after)} parquet files, total {total_bytes/1e9:.2f} GB",
           flush=True)
    print(f"[delta] {len(delta)} new shards downloaded", flush=True)
    if delta:
        deltas = sorted([os.path.basename(f) for f in delta])
        for f in deltas[:10]:
            print(f"  + {f}", flush=True)
        if len(deltas) > 10:
            print(f"  + ... ({len(deltas) - 10} more)", flush=True)

    return {
        "rc": proc.returncode,
        "before_count": len(before),
        "after_count": len(after),
        "delta_count": len(delta),
        "total_bytes": int(total_bytes),
        "total_gb": round(total_bytes / 1e9, 2),
    }


@app.local_entrypoint()
def main(num_shards: int = 240, num_workers: int = 4) -> None:
    print(f"[launcher] downloading {num_shards} ClimbMix-400B shards "
           f"(+ val shard 06542)", flush=True)
    out = download_shards.remote(num_shards=num_shards, num_workers=num_workers)
    print(f"\n[launcher] result: {out}", flush=True)
