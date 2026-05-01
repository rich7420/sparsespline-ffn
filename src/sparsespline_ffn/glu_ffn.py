"""SwiGLU and SplineGLU narrow FFN variants for Phase A controls.

These are intentionally minimal; their job is to localize where the
SimpleSpline / RL-Spline-KV quality lift comes from:

  SwiGLU  : tests whether GLU gating alone (without spline) is the lever.
  SplineGLU: tests whether B2 spline as the *gate function* (rather than
            the activation function) helps.

Both use h = mlp_ratio * d hidden units (default 1.0).  Three linears:
  W_g : d -> h   (gate projection)
  W_v : d -> h   (value projection)
  W_d : h -> d   (down projection)

  SwiGLU:    y = W_d (silu(W_g x) * (W_v x))
  SplineGLU: y = W_d (B2_spline(W_g x) * (W_v x))
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GLUConfig:
    d: int
    mlp_ratio: float = 1.0
    bias: bool = False


class SwiGLU(nn.Module):
    """Narrow SwiGLU, h = mlp_ratio * d (default d)."""

    def __init__(self, cfg: GLUConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = max(1, int(cfg.d * cfg.mlp_ratio))
        self.h = h
        self.W_g = nn.Linear(cfg.d, h, bias=cfg.bias)
        self.W_v = nn.Linear(cfg.d, h, bias=cfg.bias)
        self.W_d = nn.Linear(h, cfg.d, bias=cfg.bias)
        self._init_parameters()

    def _init_parameters(self) -> None:
        s = (3.0 / self.cfg.d) ** 0.5
        nn.init.uniform_(self.W_g.weight, -s, s)
        nn.init.uniform_(self.W_v.weight, -s, s)
        # Down projection zero-init (nanochat-style "output starts at 0")
        nn.init.zeros_(self.W_d.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.W_g(x))
        value = self.W_v(x)
        return self.W_d(gate * value)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class SplineGLUConfig:
    d: int
    mlp_ratio: float = 1.0
    G: int = 20
    spline_order: int = 2
    grid_lo: float = -3.0
    grid_hi: float = 3.0
    use_kernel: bool = True
    bias: bool = False
    init_from_silu: bool = True   # warm-start spline ≈ SiLU


class SplineGLU(nn.Module):
    """Narrow GLU with B2-spline gate.

    y = W_d ( spline_B2(W_g x) ⊙ (W_v x) )
    """

    def __init__(self, cfg: SplineGLUConfig) -> None:
        super().__init__()
        if cfg.spline_order != 2:
            raise NotImplementedError(f"only B2 supported, got {cfg.spline_order}")
        self.cfg = cfg
        h = max(1, int(cfg.d * cfg.mlp_ratio))
        self.h = h
        L = cfg.G + cfg.spline_order

        self.W_g = nn.Linear(cfg.d, h, bias=cfg.bias)
        self.W_v = nn.Linear(cfg.d, h, bias=cfg.bias)
        self.Q = nn.Parameter(torch.empty(h, L))     # per-channel spline gate
        self.W_d = nn.Linear(h, cfg.d, bias=cfg.bias)

        self.register_buffer("grid_lo_buf", torch.tensor(float(cfg.grid_lo)))
        self.register_buffer("grid_hi_buf", torch.tensor(float(cfg.grid_hi)))
        self._init_parameters()

    def _init_parameters(self) -> None:
        d = self.cfg.d
        s = (3.0 / d) ** 0.5
        nn.init.uniform_(self.W_g.weight, -s, s)
        nn.init.uniform_(self.W_v.weight, -s, s)
        nn.init.zeros_(self.W_d.weight)

        with torch.no_grad():
            G, k = self.cfg.G, self.cfg.spline_order
            grid = torch.linspace(
                self.cfg.grid_lo, self.cfg.grid_hi, G + k,
                dtype=self.Q.dtype, device=self.Q.device,
            )
            if self.cfg.init_from_silu:
                # SiLU(grid) at each grid point, broadcast to all channels.
                vals = grid * torch.sigmoid(grid)
                self.Q.copy_(vals.unsqueeze(0).expand(self.h, -1))
            else:
                self.Q.normal_(mean=0.0, std=0.1)

    def _spline_gate_reference(self, z: torch.Tensor) -> torch.Tensor:
        G, k = self.cfg.G, self.cfg.spline_order
        scale = G / (self.cfg.grid_hi - self.cfg.grid_lo)
        u = (z - self.cfg.grid_lo) * scale
        u = u.clamp(0.0, float(G - 1))
        bin_idx = u.floor().to(torch.long)
        tau = (u - bin_idx.to(u.dtype)).clamp(0.0, 1.0)
        omt = 1.0 - tau
        B0 = 0.5 * omt * omt
        B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
        B2 = 0.5 * tau * tau
        h_idx = torch.arange(self.h, device=z.device).expand_as(bin_idx)
        Q0 = self.Q[h_idx, bin_idx]
        Q1 = self.Q[h_idx, bin_idx + 1]
        Q2 = self.Q[h_idx, bin_idx + 2]
        return Q0 * B0 + Q1 * B1 + Q2 * B2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.W_g(x)
        v = self.W_v(x)
        if self.cfg.use_kernel and z.is_cuda:
            from sparsespline_ffn.kernels.b2_autograd import B2SplineActivation
            gate = B2SplineActivation.apply(
                z, self.Q,
                float(self.cfg.grid_lo), float(self.cfg.grid_hi),
                int(self.cfg.G),
            )
        else:
            gate = self._spline_gate_reference(z)
        return self.W_d(gate * v)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["SwiGLU", "GLUConfig", "SplineGLU", "SplineGLUConfig"]
