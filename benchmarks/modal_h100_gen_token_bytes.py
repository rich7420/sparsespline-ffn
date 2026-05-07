"""One-shot Modal job: generate /data/nanochat/tokenizer/token_bytes.pt.

base_train.py needs this file (computed UTF-8 byte length per token id)
for evaluate_bpb.  Our existing tokenizer was trained without writing it
out, so we recompute from the existing tokenizer.pkl on the volume.

Logic mirrors `nanochat/scripts/tok_train.py` lines 79-91 — does not
retrain the tokenizer.
"""
from __future__ import annotations

import modal


IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1",
                  index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "tokenizers", "tiktoken", "regex")
    .add_local_dir(
        local_path="/home/anon/sparsespline-ffn",
        remote_path="/repo",
        ignore=[".venv/**", ".git/**",
                "**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .run_commands("cd /repo && pip install -e .")
)
DATA_VOLUME = modal.Volume.from_name("sparsefuse-phase3-data",
                                       create_if_missing=False)
app = modal.App("sparsespline-gen-token-bytes", image=IMAGE)


@app.function(gpu="H100", timeout=600,
                volumes={"/data": DATA_VOLUME})
def generate_token_bytes() -> dict:
    import os, sys
    sys.path.insert(0, "/repo/nanochat")
    os.environ["NANOCHAT_BASE_DIR"] = "/data/nanochat"
    import torch
    from nanochat.tokenizer import get_tokenizer
    from nanochat.common import get_base_dir

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"vocab_size = {vocab_size}", flush=True)

    # Replicate tok_train.py exactly (lines 78-87 in scripts/tok_train.py):
    # `get_special_tokens()` returns SPECIAL TOKEN STRINGS (not IDs).  Compare
    # decoded strings against this set, not integer ids.
    special_set = set(tokenizer.get_special_tokens())
    print(f"special_token_strings (n={len(special_set)}): "
           f"{list(special_set)[:5]}{'...' if len(special_set) > 5 else ''}",
           flush=True)
    token_strings = [tokenizer.decode([i]) for i in range(vocab_size)]
    token_bytes = []
    for i, ts in enumerate(token_strings):
        if ts in special_set:
            token_bytes.append(0)
        else:
            token_bytes.append(len(ts.encode("utf-8")))
    token_bytes_t = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    out_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(out_path, "wb") as f:
        torch.save(token_bytes_t, f)

    nz = (token_bytes_t > 0)
    nz_vals = token_bytes_t[nz].to(torch.float32)
    summary = {
        "out_path": out_path,
        "vocab_size": int(vocab_size),
        "n_special_or_zero": int((~nz).sum().item()),
        "min_bytes": int(nz_vals.min().item()) if nz_vals.numel() else 0,
        "max_bytes": int(nz_vals.max().item()) if nz_vals.numel() else 0,
        "mean_bytes": float(nz_vals.mean().item()) if nz_vals.numel() else 0.0,
    }
    print(f"\nSAVED: {summary}", flush=True)
    return summary


@app.local_entrypoint()
def main():
    print(generate_token_bytes.remote())
