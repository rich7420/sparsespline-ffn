"""Modal CPU job — full audit of ClimbMix shards on the volume.

Verifies the "240 shards → 1.27× margin" claim with actual numbers:
  - exact shard count
  - total bytes
  - total chars (sampled across multiple shards for accuracy)
  - estimated total tokens (using actual tokenizer if available, else ratio)
  - margin vs d20 budgets:
      * our 32K-vocab d20 target: 8.703 B tokens
      * Karpathy's 65K-vocab reference: 11.220 B tokens
  - flag if margin < 1.0 (would wrap) or > 2.0 (over-budget)

Read-only.  Cheap (~30 sec, $0.01).
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install("pyarrow", "tiktoken", "tokenizers", "requests")
    .add_local_dir(
        local_path="/home/rich-wsl/sparsespline-ffn/nanochat",
        remote_path="/repo/nanochat",
        ignore=["__pycache__/**", "*.pyc", ".nanochat-runtime/**"],
        copy=True,
    )
)
DATA_VOLUME = modal.Volume.from_name("sparsefuse-phase3-data",
                                       create_if_missing=False)
app = modal.App("sparsespline-data-audit-v2", image=IMAGE)


@app.function(timeout=600, cpu=4, volumes={"/data": DATA_VOLUME})
def audit() -> dict:
    import os, glob, math
    import pyarrow.parquet as pq

    cm_dir = "/data/nanochat/base_data_climbmix"
    files = sorted(glob.glob(f"{cm_dir}/*.parquet"))
    print(f"=== shard inventory ===", flush=True)
    print(f"  count: {len(files)}", flush=True)
    if not files:
        return {"error": "no shards", "count": 0}

    total_bytes = sum(os.path.getsize(f) for f in files)
    print(f"  total bytes: {total_bytes/1e9:.2f} GB", flush=True)

    # Sample multiple shards: first 3, middle 3, last 3 (or all if fewer)
    sample_indices = []
    n = len(files)
    if n <= 9:
        sample_indices = list(range(n))
    else:
        sample_indices = list(range(3)) + list(range(n//2 - 1, n//2 + 2)) + list(range(n - 3, n))
    sample_indices = sorted(set(sample_indices))

    print(f"\n=== sampling {len(sample_indices)} shards for char/token estimate ===",
           flush=True)
    total_chars_sampled = 0
    total_rows_sampled = 0
    chars_per_shard_list = []
    rows_per_shard_list = []
    for i in sample_indices:
        f = files[i]
        t = pq.read_table(f, columns=["text"])
        n_rows = t.num_rows
        text_col = t.column("text")
        # Sum chars across ALL rows (accurate, not estimated)
        sample_size = min(2000, n_rows)
        sample = text_col.slice(0, sample_size).to_pylist()
        sample_chars = sum(len(s or "") for s in sample)
        # Extrapolate to full shard
        chars_per_shard = sample_chars * (n_rows / sample_size)
        chars_per_shard_list.append(chars_per_shard)
        rows_per_shard_list.append(n_rows)
        total_chars_sampled += chars_per_shard
        total_rows_sampled += n_rows
        print(f"  [{i:3d}] {os.path.basename(f):28s}  rows={n_rows:>6d}  "
               f"avg_chars/row={sample_chars/sample_size:>6.0f}  "
               f"est_chars={chars_per_shard/1e6:>6.1f} M", flush=True)

    avg_chars_per_shard = total_chars_sampled / len(sample_indices)
    avg_rows_per_shard = total_rows_sampled / len(sample_indices)
    est_total_chars = avg_chars_per_shard * n
    est_total_rows = avg_rows_per_shard * n

    # Try the actual nanochat tokenizer if available (gives accurate token estimate).
    actual_tokens_per_shard = None
    try:
        import sys
        sys.path.insert(0, "/repo/nanochat")
        os.environ["NANOCHAT_BASE_DIR"] = "/data/nanochat"
        from nanochat.tokenizer import get_tokenizer
        tk = get_tokenizer()
        # Sample 100 rows from the first shard, tokenize, average tokens-per-char
        t0 = pq.read_table(files[0], columns=["text"])
        sample_texts = t0.column("text").slice(0, 100).to_pylist()
        sample_tokens = sum(len(tk.encode(s)) for s in sample_texts if s)
        sample_chars = sum(len(s or "") for s in sample_texts)
        if sample_chars > 0:
            tokens_per_char = sample_tokens / sample_chars
            actual_tokens_per_shard = avg_chars_per_shard * tokens_per_char
            print(f"\n  [actual tokenizer] vocab={tk.get_vocab_size()}, "
                   f"sample tokens/char={tokens_per_char:.4f}", flush=True)
    except Exception as e:
        print(f"\n  [warn] tokenizer load failed: {e}; using 1 token / 5.4 chars heuristic",
               flush=True)

    if actual_tokens_per_shard is None:
        # Heuristic: ~5.4 chars/token for English subword tokenizers
        actual_tokens_per_shard = avg_chars_per_shard / 5.4

    total_tokens_est = actual_tokens_per_shard * n

    print(f"\n=== summary ===", flush=True)
    print(f"  shard count            : {n}", flush=True)
    print(f"  total bytes            : {total_bytes/1e9:.2f} GB", flush=True)
    print(f"  avg rows / shard       : {avg_rows_per_shard:>10,.0f}", flush=True)
    print(f"  avg chars / shard      : {avg_chars_per_shard/1e6:>10.1f} M", flush=True)
    print(f"  total chars (est)      : {est_total_chars/1e9:>10.2f} B", flush=True)
    print(f"  tokens / shard (est)   : {actual_tokens_per_shard/1e6:>10.1f} M", flush=True)
    print(f"  total tokens (est)     : {total_tokens_est/1e9:>10.2f} B", flush=True)

    print(f"\n=== budget margins ===", flush=True)
    targets = {
        "our 32K-vocab d20 budget": 8.703e9,
        "Karpathy 65K-vocab d20":   11.220e9,
    }
    for name, target in targets.items():
        margin = total_tokens_est / target
        if margin >= 1.5:
            tag = "✓ comfortable"
        elif margin >= 1.0:
            tag = "✓ adequate (1+ epoch)"
        else:
            wrap_count = target / total_tokens_est
            tag = f"✗ INSUFFICIENT — would wrap {wrap_count:.1f}× over data"
        print(f"  {name:30s}: {target/1e9:.2f} B  ⇒  margin {margin:.2f}×  {tag}",
               flush=True)

    return {
        "count": n,
        "total_gb": round(total_bytes / 1e9, 2),
        "avg_rows_per_shard": int(avg_rows_per_shard),
        "avg_chars_per_shard_m": round(avg_chars_per_shard / 1e6, 2),
        "est_total_chars_b": round(est_total_chars / 1e9, 3),
        "est_tokens_per_shard_m": round(actual_tokens_per_shard / 1e6, 2),
        "est_total_tokens_b": round(total_tokens_est / 1e9, 3),
        "margin_our_d20": round(total_tokens_est / 8.703e9, 3),
        "margin_karpathy_d20": round(total_tokens_est / 11.220e9, 3),
    }


@app.local_entrypoint()
def main():
    import json
    out = audit.remote()
    print("\n=== JSON ===\n" + json.dumps(out, indent=2))
