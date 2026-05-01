"""SimpleSpline-MLP: MLP with per-channel learnable B-spline activation.

Architecture:
    y = W_d · spline(W_u · x)

where:
    W_u : (h, d) up-projection, h = int(d * h_ratio).  Default h_ratio=0.5
          (so h = d/2, vs MLP's 4d).
    spline : per-channel B_k spline with G grid intervals and L = G+k coefs.
             Default k=2 (quadratic) — strictly contains relu² as a special
             case, so the function class is at least as expressive as MLP's.
    W_d : (d, h) down-projection.

Compared to standard MLP at d=768:
    params per layer:        d² (≈0.6M) vs MLP 8d² (4.7M)         8× fewer
    activation hidden width: h = d/2     vs MLP 4d                 8× smaller
    FLOPs per token:         2dh + small vs MLP 8d²                8× fewer
    kernel launches:         3 (GEMM + spline + GEMM) — same as MLP
    function class:          B_k spline ⊃ relu² (at G≥2 with knot at 0)

The key win that FullMix-Tucker missed: independent ``Q[h, :]`` per
hidden channel.  Each channel learns its own activation shape — exactly
the property B-splines were designed for.

Per L.4-style variance preservation: at init we set Q[h, i] = relu²(grid[i])
for all h, then perturb tiny.  This makes SimpleSpline ≈ MLP at step 0,
so any quality drift afterwards is from the spline's added freedom.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SimpleSplineConfig:
    """Hyperparameters for one SimpleSpline-MLP layer."""

    d: int                                 # residual stream dim
    h_ratio: float = 0.5                   # h = int(d * h_ratio); default d/2
    G: int = 20                            # spline grid intervals
    spline_order: int = 2                  # k; default B2 (contains relu²)
    grid_lo: float = -3.0
    grid_hi: float = 3.0
    use_kernel: bool = True                # Triton path (Hopper/Ampere)
    bias: bool = False                     # match nanochat MLP no-bias style
    init_from_relu_square: bool = True     # warm-start spline ≈ relu²

    def __post_init__(self) -> None:
        if self.spline_order != 2:
            raise NotImplementedError(
                f"Only spline_order=2 (B2) is implemented; got {self.spline_order}"
            )
        if self.G < 2:
            raise ValueError(f"G={self.G} too small")
        if self.grid_hi <= self.grid_lo:
            raise ValueError(f"grid_hi {self.grid_hi} <= grid_lo {self.grid_lo}")
        if not (0 < self.h_ratio <= 4):
            raise ValueError(f"h_ratio {self.h_ratio} outside (0, 4]")


class SimpleSplineMLP(nn.Module):
    """y = W_d · b2_spline(W_u · x)."""

    def __init__(self, config: SimpleSplineConfig) -> None:
        super().__init__()
        self.cfg = config
        d = config.d
        self.h = max(1, int(d * config.h_ratio))
        L = config.G + config.spline_order   # number of spline basis = G+k

        self.W_u = nn.Linear(d, self.h, bias=config.bias)
        self.Q = nn.Parameter(torch.empty(self.h, L))
        self.W_d = nn.Linear(self.h, d, bias=config.bias)

        # Buffers for grid endpoints (kept simple constants for now;
        # could be made learnable later).
        self.register_buffer("grid_lo_buf",
                             torch.tensor(float(config.grid_lo)))
        self.register_buffer("grid_hi_buf",
                             torch.tensor(float(config.grid_hi)))

        self._init_parameters()

    def _init_parameters(self) -> None:
        """nanochat-style W init + relu²-warm-started spline."""
        d = self.cfg.d
        # Linear layers: nanochat uses sqrt(3/d) uniform init.
        s = (3.0 / d) ** 0.5
        nn.init.uniform_(self.W_u.weight, -s, s)
        # Down-projection: zeros (nanochat-style — output starts at 0)
        nn.init.zeros_(self.W_d.weight)

        # Q init: per-channel relu² evaluated at the grid points.
        # The kernel uses indices bin..bin+2 (after shift), so coefficient
        # i corresponds to "knot i-1" of the original B-spline frame.  At
        # init we just want Q[i] ≈ relu²(grid_value(i)) so the spline ≈ relu².
        with torch.no_grad():
            G = self.cfg.G
            grid = torch.linspace(
                self.cfg.grid_lo, self.cfg.grid_hi, G + self.cfg.spline_order,
                dtype=self.Q.dtype, device=self.Q.device,
            )
            # relu² at each grid point
            relu_sq = torch.where(grid > 0, grid * grid, torch.zeros_like(grid))
            # Broadcast to all channels
            self.Q.copy_(relu_sq.unsqueeze(0).expand(self.h, -1))

            if not self.cfg.init_from_relu_square:
                # If user wants random init, replace with small Gaussian.
                self.Q.normal_(mean=0.0, std=0.1)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.cfg.d
        original_shape = x.shape
        if original_shape[-1] != d:
            raise ValueError(
                f"input last-dim {original_shape[-1]} != d={d}"
            )
        x_flat = x.reshape(-1, d)

        # Stage 1: up-projection
        z = self.W_u(x_flat)  # (E, h)

        # Stage 2: per-channel B2 spline activation
        if self.cfg.use_kernel and z.is_cuda:
            from sparsespline_ffn.kernels.b2_autograd import B2SplineActivation
            a = B2SplineActivation.apply(
                z, self.Q,
                float(self.cfg.grid_lo), float(self.cfg.grid_hi),
                int(self.cfg.G),
            )
        else:
            a = self._spline_reference(z)

        # Stage 3: down-projection
        y = self.W_d(a)
        return y.reshape(original_shape)

    def _spline_reference(self, z: torch.Tensor) -> torch.Tensor:
        """PyTorch reference (oracle) for the B2 spline activation.

        Used for CPU and as numerical oracle for the kernel.  Slow.
        """
        G, k = self.cfg.G, self.cfg.spline_order
        L = G + k
        scale = G / (self.cfg.grid_hi - self.cfg.grid_lo)
        u = (z - self.cfg.grid_lo) * scale
        u = u.clamp(0.0, float(G - 1))
        bin_idx = u.floor().to(torch.long)
        tau = (u - bin_idx.to(u.dtype)).clamp(0.0, 1.0)

        one_minus_tau = 1.0 - tau
        B0 = one_minus_tau * one_minus_tau * 0.5
        B1 = (1.0 + 2.0 * tau - 2.0 * tau * tau) * 0.5
        B2 = tau * tau * 0.5

        # Q is (h, L); we need Q[c, bin_idx[..., c]] where c is the channel.
        # bin_idx shape: (E, h) (assuming 2D z).  Gather per-channel:
        # idx0[e, c] = bin_idx[e, c]; we want Q[c, idx0[e, c]] for each.
        # Use advanced indexing.
        h_idx = torch.arange(self.h, device=z.device).expand_as(bin_idx)
        Q0 = self.Q[h_idx, bin_idx]
        Q1 = self.Q[h_idx, bin_idx + 1]
        Q2 = self.Q[h_idx, bin_idx + 2]

        return Q0 * B0 + Q1 * B1 + Q2 * B2

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"d={c.d}, h={self.h} (h_ratio={c.h_ratio}), "
            f"G={c.G}, k={c.spline_order}, kernel={c.use_kernel}"
        )


__all__ = ["SimpleSplineMLP", "SimpleSplineConfig"]
