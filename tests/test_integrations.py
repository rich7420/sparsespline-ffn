"""Integration tests for the nanochat adapter and the tiny transformer.

The library's promise is "drop FullMix-Tucker into a transformer's FFN slot
without forking model code."  These tests pin that promise:

  - ``TinyTransformerLM`` runs forward and computes a finite cross-entropy
    loss with both an MLP factory and a FullMix-Tucker factory.
  - ``replace_mlp_with_sparsespline`` swaps the right layers for several
    schedules and leaves output shape intact.
  - ``summarize_replacement`` accurately reports which layer indices were
    swapped.
  - The adapter raises clear errors when invariants break (no layer list,
    no embedding dim, missing mlp attribute).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from integrations.nanochat.adapter import (
    replace_mlp_with_sparsespline,
    summarize_replacement,
)
from integrations.tiny_transformer import (
    Block,
    CausalSelfAttention,
    RMSNorm,
    TinyConfig,
    TinyTransformerLM,
)
from sparsespline_ffn import MLPFFN, FullMixTuckerFFN, build_ffn

# ---- TinyTransformerLM: forward + loss with both FFN factories -----------


def _mlp_factory(d: int):
    def make(*, layer_idx: int) -> nn.Module:
        return MLPFFN(d=d, mlp_ratio=2)
    return make


def _fullmix_factory(d: int, num_layers: int, schedule: str = "all"):
    def make(*, layer_idx: int) -> nn.Module:
        return build_ffn(
            ffn_type="fullmix_tucker",
            d=d,
            layer_idx=layer_idx,
            num_layers=num_layers,
            schedule=schedule,
            R_o=d // 2,
            R_i=d // 2,
            R_b=4,
            G=8,
        )
    return make


def test_tiny_transformer_with_mlp_runs():
    cfg = TinyConfig(vocab_size=64, d=32, n_head=4, n_layer=2, block_size=16)
    torch.manual_seed(0)
    model = TinyTransformerLM(cfg, _mlp_factory(cfg.d))

    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(idx, targets)

    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_tiny_transformer_with_fullmix_runs_and_backwards():
    cfg = TinyConfig(vocab_size=64, d=32, n_head=4, n_layer=3, block_size=16)
    torch.manual_seed(1)
    model = TinyTransformerLM(
        cfg, _fullmix_factory(cfg.d, cfg.n_layer, schedule="late")
    )

    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(idx, targets)
    assert torch.isfinite(loss)

    loss.backward()
    finite_grads = [
        torch.isfinite(p.grad).all().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    assert all(finite_grads), "non-finite gradient through tiny transformer"


def test_tiny_transformer_num_params_excludes_tied_head():
    cfg = TinyConfig(vocab_size=64, d=32, n_head=4, n_layer=2, block_size=16)
    model = TinyTransformerLM(cfg, _mlp_factory(cfg.d))

    raw = sum(p.numel() for p in model.parameters())
    reported = model.num_params()
    # lm_head.weight is tied to wte.weight; num_params subtracts it once.
    assert reported == raw - model.lm_head.weight.numel()


def test_tiny_block_pre_rmsnorm_residual_shape():
    """Sanity for the Block primitive used inside TinyTransformerLM."""
    cfg = TinyConfig(vocab_size=64, d=32, n_head=4, n_layer=1, block_size=8)
    block = Block(cfg, MLPFFN(d=cfg.d, mlp_ratio=2))
    x = torch.randn(2, cfg.block_size, cfg.d)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rmsnorm_preserves_shape_and_normalizes():
    rms = RMSNorm(d=8)
    x = torch.randn(3, 4, 8) * 5.0
    y = rms(x)
    assert y.shape == x.shape
    # The default weight=1 RMSNorm makes the per-row RMS ~1.
    row_rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(row_rms, torch.ones_like(row_rms), atol=1e-3)


def test_causal_self_attention_runs():
    cfg = TinyConfig(vocab_size=8, d=16, n_head=4, n_layer=1, block_size=8)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, cfg.block_size, cfg.d)
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# ---- nanochat adapter: replacement + summary ------------------------------


def _make_tiny_model(d: int = 32, n_layer: int = 4) -> TinyTransformerLM:
    cfg = TinyConfig(
        vocab_size=64, d=d, n_head=4, n_layer=n_layer, block_size=16
    )
    return TinyTransformerLM(cfg, _mlp_factory(d))


def test_adapter_swaps_late_layers():
    model = _make_tiny_model(n_layer=4)
    replace_mlp_with_sparsespline(
        model, schedule="late", R_o=8, R_i=8, R_b=4, G=6
    )
    layers = list(model.transformer.h)
    # 'late' selects layers >= n // 2 = 2 (indices 2, 3).
    assert isinstance(layers[0].mlp, MLPFFN)
    assert isinstance(layers[1].mlp, MLPFFN)
    assert isinstance(layers[2].mlp, FullMixTuckerFFN)
    assert isinstance(layers[3].mlp, FullMixTuckerFFN)


def test_adapter_swaps_all_layers_with_schedule_all():
    model = _make_tiny_model(n_layer=3)
    replace_mlp_with_sparsespline(
        model, schedule="all", R_o=8, R_i=8, R_b=4, G=6
    )
    for blk in model.transformer.h:
        assert isinstance(blk.mlp, FullMixTuckerFFN)


def test_adapter_no_replacement_when_schedule_none():
    """schedule='none' should leave every block.mlp as MLPFFN (the fallback)."""
    model = _make_tiny_model(n_layer=3)
    replace_mlp_with_sparsespline(
        model, schedule="none", R_o=8, R_i=8, R_b=4, G=6
    )
    for blk in model.transformer.h:
        assert isinstance(blk.mlp, MLPFFN)


def test_adapter_preserves_forward_after_replacement():
    """After swapping, the model should still produce finite logits/loss."""
    model = _make_tiny_model(n_layer=3)
    replace_mlp_with_sparsespline(
        model, schedule="late", R_o=8, R_i=8, R_b=4, G=6
    )
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, 64)
    assert torch.isfinite(loss)


def test_summarize_replacement_after_late():
    model = _make_tiny_model(n_layer=4)
    replace_mlp_with_sparsespline(
        model, schedule="late", R_o=8, R_i=8, R_b=4, G=6
    )
    summary = summarize_replacement(model)
    assert summary["n_layers"] == 4
    assert summary["swapped"] == [2, 3]
    assert summary["kept_mlp"] == [0, 1]


def test_summarize_replacement_before_swap_lists_all_mlp():
    model = _make_tiny_model(n_layer=3)
    summary = summarize_replacement(model)
    assert summary["n_layers"] == 3
    assert summary["swapped"] == []
    assert summary["kept_mlp"] == [0, 1, 2]


def test_adapter_raises_when_layer_list_missing():
    """Passing a model without ``transformer.h`` and no explicit ``layers``
    must raise an informative AttributeError."""
    bare = nn.Linear(4, 4)
    with pytest.raises(AttributeError, match="transformer layer list"):
        replace_mlp_with_sparsespline(bare, schedule="all", d=4,
                                      R_o=2, R_i=2, R_b=2, G=4)


def test_adapter_raises_when_d_missing():
    """Pass an explicit ``layers=`` list but no ``d=`` and no model.config:
    the adapter should fail with an informative error."""
    blocks = nn.ModuleList([
        nn.Module()  # placeholder; would also fail on missing 'mlp'
        for _ in range(2)
    ])
    container = nn.Module()
    container.layers = blocks  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match="embedding dim"):
        replace_mlp_with_sparsespline(container, schedule="all",
                                      R_o=2, R_i=2, R_b=2, G=4)


def test_adapter_raises_when_block_lacks_mlp_attr():
    """A custom block-like module that has no ``mlp`` attribute should raise
    when ``mlp_attr='mlp'`` (the default)."""
    blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    container = nn.Module()
    container.layers = blocks  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match="no attribute"):
        replace_mlp_with_sparsespline(container, schedule="all", d=4,
                                      R_o=2, R_i=2, R_b=2, G=4)


def test_adapter_empty_layer_list_rejected():
    container = nn.Module()
    container.layers = nn.ModuleList()  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="No transformer layers"):
        replace_mlp_with_sparsespline(container, schedule="all", d=4,
                                      R_o=2, R_i=2, R_b=2, G=4)


def test_adapter_custom_mlp_attr():
    """Some forks call the FFN slot ``ffn`` instead of ``mlp``.  The adapter
    should respect ``mlp_attr=`` to support that."""
    class CustomBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ffn = nn.Linear(8, 8)

    container = nn.Module()
    container.layers = nn.ModuleList([CustomBlock(), CustomBlock()])  # type: ignore[attr-defined]
    replace_mlp_with_sparsespline(
        container, schedule="all", d=8, mlp_attr="ffn",
        R_o=4, R_i=4, R_b=2, G=4,
    )
    assert isinstance(container.layers[0].ffn, FullMixTuckerFFN)
    assert isinstance(container.layers[1].ffn, FullMixTuckerFFN)


def test_adapter_can_train_with_few_steps():
    """End-to-end smoke: adapter-swapped model can be SGD-trained on dummy
    targets and loss decreases over a small number of steps."""
    torch.manual_seed(99)
    model = _make_tiny_model(d=24, n_layer=3)
    replace_mlp_with_sparsespline(
        model, schedule="late", R_o=12, R_i=12, R_b=4, G=8
    )
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    initial = final = float("inf")
    for step in range(15):
        opt.zero_grad()
        _logits, loss = model(idx, targets)
        if step == 0:
            initial = loss.item()
        if step == 14:
            final = loss.item()
        loss.backward()
        opt.step()

    assert torch.isfinite(torch.tensor(final))
    # 15 SGD steps on 2x16 tokens should reduce loss meaningfully on a
    # small enough model.  Loose 5% threshold is enough to catch totally
    # broken training (e.g., grads stuck at zero).
    assert final < initial * 0.95, (
        f"adapter-swapped model failed to reduce loss: "
        f"{initial:.4f} -> {final:.4f}"
    )
