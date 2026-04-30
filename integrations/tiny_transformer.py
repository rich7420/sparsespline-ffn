"""Tiny causal-LM transformer for SparseSpline-FFN prototype integration.

Block layout (pre-RMSNorm, nanochat-style):
    h = x  + Attn(RMSNorm(x))
    h = h  + FFN(RMSNorm(h))

The FFN slot accepts any ``nn.Module`` whose forward takes (..., d) -> (..., d),
so we can drop in either ``MLPFFN`` or ``FullMixTuckerFFN`` (or anything from
``build_ffn``) without changing transformer code.

This module is *not* a nanochat replacement — it is a small, self-contained
testbed for the replacement plumbing.  Production integration should follow
``integrations/nanochat/adapter.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    vocab_size: int = 256
    d: int = 128
    n_head: int = 4
    n_layer: int = 2
    block_size: int = 64
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        assert cfg.d % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d // cfg.n_head
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)  # (B, T, 3d)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(y)


class Block(nn.Module):
    """Pre-RMSNorm transformer block with a pluggable FFN slot."""

    def __init__(self, cfg: TinyConfig, ffn: nn.Module) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg.d)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d)
        self.mlp = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    """Minimal causal LM for prototype integration.

    The block list lives at ``model.transformer.h`` to mirror nanochat /
    nanoGPT, so the same adapter that swaps ``block.mlp`` works here.
    """

    def __init__(self, cfg: TinyConfig, ffn_factory) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.d),
            wpe=nn.Embedding(cfg.block_size, cfg.d),
            h=nn.ModuleList([
                Block(cfg, ffn_factory(layer_idx=i)) for i in range(cfg.n_layer)
            ]),
            ln_f=RMSNorm(cfg.d),
        ))
        self.lm_head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        # Tie weights, nanochat-style.
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self) -> int:
        # Excludes the tied lm_head weight to avoid double counting.
        n = sum(p.numel() for p in self.parameters())
        n -= self.lm_head.weight.numel()
        return n


__all__ = ["TinyConfig", "TinyTransformerLM", "Block", "RMSNorm",
           "CausalSelfAttention"]
