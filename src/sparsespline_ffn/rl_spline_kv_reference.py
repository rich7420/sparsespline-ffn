"""Residual Low-Rank Spline-KV — PyTorch reference implementation.

Phase B1 of the v7 plan (docs/THEORY_v7_RL_SPLINE_KV.md).  This is the
oracle / correctness baseline against which the eventual FlashSplineFeature
Triton kernel will be tested.  Pure PyTorch; slow; no fused kernel.

Forward (per token x in R^d):
    z = K x                                     # [h]
    a = relu_sq(z)                              # [h]
    delta_c = sum_j sum_{b in active(z_j)} B_b(z_j) * C[j, b, c]   # [r]
    y = W_out [a; lambda * delta]               # [d]

with C in R^{h x L x r}, B-spline order = 2 (3 active bins per key).

The reference uses the same B2 basis formulas as
``sparsespline_ffn.simple_spline_mlp._spline_reference`` so that any
discrepancy with the (later) Triton kernel comes from the kernel, not
from a different basis convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RLSplineKVConfig:
    """Hyperparameters for one RL-Spline-KV layer (v7 spec defaults)."""

    d: int                                 # residual stream dim
    h_ratio: float = 1.0                   # base hidden = h_ratio * d (default h=d)
    r: int = 64                            # residual code dim
    G: int = 20                            # spline grid intervals (L = G + spline_order)
    spline_order: int = 2                  # 1=B1 (linear), 2=B2 (quadratic)
    grid_lo: float = -3.0
    grid_hi: float = 3.0
    lambda_scale: float = 1.0              # cold-start: C=0 makes delta=0 already
    activation: str = "relu_sq"            # "relu_sq" | "gelu" | "identity"
    bias: bool = False
    init_C_zero: bool = True               # v7 §R.5: C=0 soft cold start
    use_kernel: bool = False               # B2 only: route through FlashSplineFeature
                                            # kernel — saves ~600 MB/layer activations
    bwd_kernel: str = "triton"             # "triton" | "hopper_cuda" | "wgmma_cuda"

    def __post_init__(self) -> None:
        if self.G < 2:
            raise ValueError(f"G={self.G} too small")
        if self.grid_hi <= self.grid_lo:
            raise ValueError(f"grid_hi {self.grid_hi} <= grid_lo {self.grid_lo}")
        if not (0 < self.h_ratio <= 4):
            raise ValueError(f"h_ratio {self.h_ratio} outside (0, 4]")
        if self.r < 1:
            raise ValueError(f"r={self.r} must be >= 1")
        if self.spline_order not in (1, 2):
            raise ValueError(f"spline_order must be 1 or 2; got {self.spline_order}")
        if self.activation not in ("relu_sq", "gelu", "identity"):
            raise ValueError(f"unknown activation: {self.activation}")


def _activation(z: torch.Tensor, name: str) -> torch.Tensor:
    if name == "relu_sq":
        return torch.where(z > 0, z * z, torch.zeros_like(z))
    if name == "gelu":
        return F.gelu(z)
    if name == "identity":
        return z
    raise ValueError(name)


def _b2_basis_and_bins(
    z: torch.Tensor, grid_lo: float, grid_hi: float, G: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (bin_idx, B0, B1, B2, in_range) for a B2 spline.

    Active basis indices are {bin_idx, bin_idx+1, bin_idx+2}.  Returns:
      bin_idx : long, shape z.shape, in [0, G-1]
      B0,B1,B2: float, shape z.shape, partition-of-unity weights
      in_range: bool,  shape z.shape, True where z is in [grid_lo, grid_hi]
    """
    scale = G / (grid_hi - grid_lo)
    u = (z - grid_lo) * scale
    in_range = (u >= 0.0) & (u <= float(G))
    u_clip = u.clamp(0.0, float(G - 1))
    bin_idx = u_clip.floor().to(torch.long)
    tau = (u_clip - bin_idx.to(u.dtype)).clamp(0.0, 1.0)

    one_minus_tau = 1.0 - tau
    B0 = 0.5 * one_minus_tau * one_minus_tau
    B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
    B2 = 0.5 * tau * tau
    return bin_idx, B0, B1, B2, in_range


def _bin_diag(
    z: torch.Tensor, grid_lo: float, grid_hi: float, G: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (bin_idx, tau, in_range) for diagnostics.  Returns clipped bin."""
    scale = G / (grid_hi - grid_lo)
    u = (z - grid_lo) * scale
    in_range = (u >= 0.0) & (u <= float(G))
    u_clip = u.clamp(0.0, float(G - 1))
    bin_idx = u_clip.floor().to(torch.long)
    tau = (u_clip - bin_idx.to(u.dtype)).clamp(0.0, 1.0)
    return bin_idx, tau, in_range


def _b1_basis_and_bins(
    z: torch.Tensor, grid_lo: float, grid_hi: float, G: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (bin_idx, B0, B1, in_range) for a B1 (linear) spline.

    Active basis indices are {bin_idx, bin_idx+1}.  L = G + 1.
      B0(τ) = 1 - τ
      B1(τ) = τ
    """
    scale = G / (grid_hi - grid_lo)
    u = (z - grid_lo) * scale
    in_range = (u >= 0.0) & (u <= float(G))
    u_clip = u.clamp(0.0, float(G - 1))
    bin_idx = u_clip.floor().to(torch.long)
    tau = (u_clip - bin_idx.to(u.dtype)).clamp(0.0, 1.0)
    B0 = 1.0 - tau
    B1 = tau
    return bin_idx, B0, B1, in_range


def flash_spline_feature_reference(
    z: torch.Tensor,             # [N, h]
    C: torch.Tensor,             # [h, L, r]
    grid_lo: float,
    grid_hi: float,
    G: int,
    activation: str = "relu_sq",
    lambda_scale: float = 1.0,
    spline_order: int = 2,
) -> torch.Tensor:                # [N, h+r]
    """Pure-PyTorch reference: f = [phi(z); lambda * delta(z, C)].

    Uses dense `torch.einsum` for the delta accumulation — slow but
    differentiable through autograd, suitable as oracle.

    spline_order: 1 (B1 linear, L=G+1) or 2 (B2 quadratic, L=G+2).
    """
    if z.dim() != 2:
        raise ValueError(f"z must be 2D [N, h], got {z.shape}")
    N, h = z.shape
    h_C, L, r = C.shape
    if h_C != h:
        raise ValueError(f"C[0]={h_C} != h={h}")
    if L != G + spline_order:
        raise ValueError(
            f"L={L} should equal G+spline_order={G + spline_order}"
        )

    # phi(z)
    a = _activation(z, activation)               # [N, h]

    h_idx = torch.arange(h, device=z.device).unsqueeze(0).expand(N, h)  # [N, h]

    if spline_order == 1:
        bin_idx, B0, B1, in_range = _b1_basis_and_bins(z, grid_lo, grid_hi, G)
        mask = in_range.to(B0.dtype)
        B0 = B0 * mask
        B1 = B1 * mask
        C0 = C[h_idx, bin_idx, :]
        C1 = C[h_idx, bin_idx + 1, :]
        delta = (
            (B0.unsqueeze(-1) * C0) + (B1.unsqueeze(-1) * C1)
        ).sum(dim=1)
    else:
        bin_idx, B0, B1, B2, in_range = _b2_basis_and_bins(z, grid_lo, grid_hi, G)
        mask = in_range.to(B0.dtype)
        B0 = B0 * mask
        B1 = B1 * mask
        B2 = B2 * mask
        C0 = C[h_idx, bin_idx, :]
        C1 = C[h_idx, bin_idx + 1, :]
        C2 = C[h_idx, bin_idx + 2, :]
        delta = (
            (B0.unsqueeze(-1) * C0)
            + (B1.unsqueeze(-1) * C1)
            + (B2.unsqueeze(-1) * C2)
        ).sum(dim=1)

    return torch.cat([a, lambda_scale * delta], dim=-1)


class RLSplineKVReference(nn.Module):
    """Pure-PyTorch RL-Spline-KV FFN.

    Forward graph:
        z = K x
        f = flash_spline_feature_reference(z, C)  # [N, h+r]
        y = W_out f

    Used as: (a) numerical oracle for the FlashSplineFeature kernel,
             (b) reference architecture for ablations before kernel work.
    Speed is not a goal here.
    """

    def __init__(self, cfg: RLSplineKVConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        h = max(1, int(d * cfg.h_ratio))
        self.h = h
        L = cfg.G + cfg.spline_order

        self.K = nn.Linear(d, h, bias=cfg.bias)
        self.C = nn.Parameter(torch.empty(h, L, cfg.r))
        self.W_out = nn.Linear(h + cfg.r, d, bias=cfg.bias)

        self.register_buffer("grid_lo_buf", torch.tensor(float(cfg.grid_lo)))
        self.register_buffer("grid_hi_buf", torch.tensor(float(cfg.grid_hi)))

        # Diagnostic stash — populated each forward when enabled.
        self._diag: dict[str, float] = {}
        self._diag_enabled: bool = False

        self._init_parameters()

    def _init_parameters(self) -> None:
        """v7 §R.5 init: K Kaiming, W_out Xavier-like, C zero."""
        d = self.cfg.d
        # nanochat-style sqrt(3/d) uniform for both linears
        s_in = (3.0 / d) ** 0.5
        s_h = (3.0 / (self.h + self.cfg.r)) ** 0.5
        nn.init.uniform_(self.K.weight, -s_in, s_in)
        # W_out: Xavier-style with non-zero spline columns (so gradient
        # flows into C from step 0 — see v7 §R.5).
        nn.init.uniform_(self.W_out.weight, -s_h, s_h)
        # C: zero init for soft cold-start (delta=0 at step 0 but dC nonzero)
        if self.cfg.init_C_zero:
            with torch.no_grad():
                self.C.zero_()
        else:
            nn.init.normal_(self.C, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.cfg.d
        original_shape = x.shape
        if original_shape[-1] != d:
            raise ValueError(f"input last-dim {original_shape[-1]} != d={d}")
        x_flat = x.reshape(-1, d)

        z = self.K(x_flat)                              # [N, h]
        if (self.cfg.use_kernel and self.cfg.spline_order == 2
                and z.is_cuda):
            from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
                FlashSplineFeature as _FF,
            )
            f = _FF.apply(
                z, self.C,
                float(self.cfg.grid_lo), float(self.cfg.grid_hi),
                int(self.cfg.G), self.cfg.activation,
                float(self.cfg.lambda_scale), True,
                self.cfg.bwd_kernel,
            )                                           # [N, h+r]
        else:
            f = flash_spline_feature_reference(
                z, self.C,
                grid_lo=float(self.cfg.grid_lo),
                grid_hi=float(self.cfg.grid_hi),
                G=int(self.cfg.G),
                activation=self.cfg.activation,
                lambda_scale=float(self.cfg.lambda_scale),
                spline_order=int(self.cfg.spline_order),
            )                                           # [N, h+r]
        y = self.W_out(f)                               # [N, d]
        # Stash diagnostics (no grad path) — used by training loop hooks.
        if not self.training or self._diag_enabled:
            with torch.no_grad():
                base = f[:, : self.h].float()
                delta = f[:, self.h:].float()
                G = int(self.cfg.G)
                # ρ_δ = RMS(W_δ · λδ) / RMS(W_a · a)  — v7 §R.1.4 metric.
                # Split W_out along input axis: cols [:h] = W_a, [h:] = W_δ.
                W_a = self.W_out.weight[:, :self.h].float()
                W_d = self.W_out.weight[:, self.h:].float()
                ya = base @ W_a.T            # [N, d]
                yd = delta @ W_d.T           # [N, d] (delta already includes λ)
                rms_a = float(ya.pow(2).mean().sqrt().item())
                rms_d = float(yd.pow(2).mean().sqrt().item())
                # W_out grad split: cols [:h] = W_a, [h:] = W_δ
                W_out_grad = self.W_out.weight.grad
                w_a_grad_norm = (W_out_grad[:, :self.h].detach().norm().item()
                                 if W_out_grad is not None else 0.0)
                w_d_grad_norm = (W_out_grad[:, self.h:].detach().norm().item()
                                 if W_out_grad is not None else 0.0)
                self._diag = {
                    "C_norm":    float(self.C.detach().norm().item()),
                    "C_grad_norm": float(self.C.grad.detach().norm().item())
                                 if self.C.grad is not None else 0.0,
                    "base_rms":  float(base.pow(2).mean().sqrt().item()),
                    "delta_rms": float(delta.pow(2).mean().sqrt().item()),
                    "rho_delta": rms_d / max(1e-9, rms_a),
                    "y_base_rms":  rms_a,
                    "y_delta_rms": rms_d,
                    "W_a_grad_norm": float(w_a_grad_norm),
                    "W_d_grad_norm": float(w_d_grad_norm),
                }
                # Bin diagnostics (sample-based, cheap)
                bin_idx, _, in_range_mask = _bin_diag(
                    z, float(self.cfg.grid_lo), float(self.cfg.grid_hi), G,
                )
                hist = torch.bincount(bin_idx.flatten(), minlength=G).float()
                p = hist / hist.sum().clamp_min(1.0)
                entropy = -(p * (p + 1e-12).log()).sum().item()
                edge_count = (
                    (bin_idx == 0).sum().item()
                    + (bin_idx == G - 1).sum().item()
                )
                self._diag["bin_entropy"] = float(entropy)
                self._diag["bin_entropy_norm"] = float(entropy / max(1e-9, math.log(G)))
                self._diag["edge_bin_frac"] = float(edge_count) / max(1, bin_idx.numel())
                self._diag["active_frac"] = float(in_range_mask.float().mean().item())
        return y.reshape(original_shape)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"d={c.d}, h={self.h} (h_ratio={c.h_ratio}), r={c.r}, "
            f"G={c.G}, lambda={c.lambda_scale}, activation={c.activation}"
        )


__all__ = [
    "RLSplineKVConfig", "RLSplineKVReference",
    "flash_spline_feature_reference",
]
