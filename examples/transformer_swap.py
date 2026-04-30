"""Replace MLP FFN slots with FullMix-Tucker in a transformer.

This is the most common usage pattern: take an existing nanoGPT/nanochat-style
model and swap a subset of FFN slots for SparseSpline-FFN modules without
touching the model code.

The example uses ``integrations.tiny_transformer.TinyTransformerLM`` as a
stand-in for nanochat; the same adapter call works on real nanochat models
because they share the ``model.transformer.h[i].mlp`` attribute layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running ``python examples/transformer_swap.py`` directly from the
# repo root: ``integrations`` lives at the same level as ``examples``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402  (path-shimmed import order)

from integrations.nanochat.adapter import (  # noqa: E402
    replace_mlp_with_sparsespline,
    summarize_replacement,
)
from integrations.tiny_transformer import TinyConfig, TinyTransformerLM  # noqa: E402
from sparsespline_ffn import MLPFFN  # noqa: E402


def _mlp_factory(d: int):
    def make(*, layer_idx: int) -> torch.nn.Module:
        return MLPFFN(d=d, mlp_ratio=2)
    return make


def main() -> None:
    torch.manual_seed(0)
    cfg = TinyConfig(vocab_size=128, d=64, n_head=4, n_layer=6, block_size=32)
    model = TinyTransformerLM(cfg, _mlp_factory(cfg.d))
    print(f"Built TinyTransformerLM with {model.num_params():,} params "
          f"(all-MLP).")

    # Swap the late half of FFN slots with FullMix-Tucker.
    replace_mlp_with_sparsespline(
        model,
        schedule="late",            # see sparsespline_ffn.should_replace_layer
        R_o=cfg.d // 2,
        R_i=cfg.d // 2,
        R_b=4,
        G=8,
    )
    summary = summarize_replacement(model)
    print(f"After swap: {summary}")

    # Forward + loss confirms the swapped model still trains end-to-end.
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(idx, targets)
    print(f"forward: logits {tuple(logits.shape)}, loss {loss.item():.3f}")

    loss.backward()
    bad = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not bad, f"non-finite grad: {bad}"
    print("backward: all parameter grads are finite.")


if __name__ == "__main__":
    main()
