"""Audit Modal volume `sparsefuse-phase3-data` — count shards, total bytes,
total tokens (approx), and locate where ClimbMix shards live.

Read-only.  No download yet.
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("tree")
    .pip_install("pyarrow")
)
DATA_VOLUME = modal.Volume.from_name("sparsefuse-phase3-data",
                                       create_if_missing=False)
app = modal.App("sparsespline-data-audit", image=IMAGE)


@app.function(timeout=300, volumes={"/data": DATA_VOLUME})
def audit() -> dict:
    import os, glob

    out: dict = {}
    # Top-level layout
    print("=== /data top-level ===", flush=True)
    for entry in sorted(os.listdir("/data")):
        full = os.path.join("/data", entry)
        if os.path.isdir(full):
            count = 0
            size = 0
            try:
                for root, _, files in os.walk(full):
                    for f in files:
                        count += 1
                        size += os.path.getsize(os.path.join(root, f))
            except Exception as e:
                print(f"  walk failed: {e}", flush=True)
            print(f"  {entry}/  files={count}  size={size/1e9:.2f} GB", flush=True)
        else:
            print(f"  {entry}   {os.path.getsize(full)/1e9:.4f} GB", flush=True)

    # nanochat data shards
    print("\n=== /data/nanochat/base_data_climbmix/ ===", flush=True)
    cm_dir = "/data/nanochat/base_data_climbmix"
    if os.path.isdir(cm_dir):
        files = sorted(os.listdir(cm_dir))
        print(f"  total files: {len(files)}", flush=True)
        for f in files[:10]:
            sz = os.path.getsize(os.path.join(cm_dir, f))
            print(f"    {f}   {sz/1e6:.1f} MB", flush=True)
        if len(files) > 10:
            print(f"    ... ({len(files)-10} more)", flush=True)
        out["climbmix_shard_count"] = len(files)
        out["climbmix_total_bytes"] = sum(
            os.path.getsize(os.path.join(cm_dir, f)) for f in files
        )
    else:
        print("  (missing)", flush=True)

    # Token-byte file
    tb = "/data/nanochat/tokenizer/token_bytes.pt"
    out["token_bytes_exists"] = os.path.exists(tb)
    if out["token_bytes_exists"]:
        out["token_bytes_size"] = os.path.getsize(tb)

    # Try to read the first parquet to estimate tokens-per-shard
    parquets = sorted(glob.glob(f"{cm_dir}/*.parquet")) if os.path.isdir(cm_dir) else []
    if parquets:
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(parquets[0], columns=None)
            n_rows = t.num_rows
            # ClimbMix usually has a 'text' column; sum char lengths to estimate.
            if "text" in t.schema.names:
                text_col = t.column("text")
                # Sample first 100 rows for char-count estimate
                import pyarrow as pa
                sample = text_col.slice(0, min(100, n_rows)).to_pylist()
                avg_chars = sum(len(s or "") for s in sample) / max(1, len(sample))
                est_chars_per_shard = avg_chars * n_rows
                # Assume ~5 chars/token (common for English subword tokenizers)
                est_tokens_per_shard = est_chars_per_shard / 5
                out["sample_shard_rows"] = int(n_rows)
                out["sample_shard_avg_chars"] = float(avg_chars)
                out["est_chars_per_shard"] = float(est_chars_per_shard)
                out["est_tokens_per_shard"] = float(est_tokens_per_shard)
                print(f"\n=== first parquet stats ===", flush=True)
                print(f"  rows: {n_rows}", flush=True)
                print(f"  avg chars/row: {avg_chars:.0f}", flush=True)
                print(f"  est tokens/shard: {est_tokens_per_shard/1e6:.1f} M", flush=True)
        except Exception as e:
            print(f"  parquet inspect failed: {e}", flush=True)

    print(f"\nFINAL: {out}", flush=True)
    return out


@app.local_entrypoint()
def main():
    print(audit.remote())
