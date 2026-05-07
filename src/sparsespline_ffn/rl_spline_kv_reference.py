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
    fwd_kernel: str = "auto"               # "auto" (CUDA when eligible) | "triton" | "wgmma_cuda"
    no_base: bool = False                  # Plan A: drop ReLU² base, pure-spline FFN

    # v8 amendment (THEORY_v8_MULTIPLICATIVE_GATING.md) — opt-in, default
    # leaves v7 additive behaviour unchanged.
    gating_mode: str = "additive"          # "additive" (v7) | "multiplicative" (v8)
    c_init_std: float = 0.0                # 0 → keep init_C_zero behaviour;
                                            # >0 → init C ~ N(0, c_init_std), overrides init_C_zero

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
        if self.gating_mode not in ("additive", "multiplicative"):
            raise ValueError(f"gating_mode must be 'additive' or 'multiplicative'; got {self.gating_mode}")
        if self.c_init_std < 0:
            raise ValueError(f"c_init_std must be >= 0; got {self.c_init_std}")
        if self.gating_mode == "multiplicative" and self.no_base:
            raise ValueError("multiplicative gating requires base path (no_base=True is incompatible)")


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
        # v7 cases:
        #   no_base=True  → W_out is Linear(r, d) — spline-only branch
        #   default       → W_out is Linear(h+r, d) — additive cat
        # v8 amendment (multiplicative gating, THEORY_v8):
        #   gating_mode="multiplicative" → W_out is Linear(h, d).  The spline
        #   modulates the base path elementwise; output projection is on the
        #   gated activation only.  A new W_d_proj Linear(r, h) lifts the
        #   r-dim spline residual to the h-dim base hidden for gating.
        if cfg.gating_mode == "multiplicative":
            wout_in = h
            self.W_d_proj = nn.Linear(cfg.r, h, bias=False)
        else:
            wout_in = cfg.r if cfg.no_base else (h + cfg.r)
            self.W_d_proj = None
        self.W_out = nn.Linear(wout_in, d, bias=cfg.bias)

        self.register_buffer("grid_lo_buf", torch.tensor(float(cfg.grid_lo)))
        self.register_buffer("grid_hi_buf", torch.tensor(float(cfg.grid_hi)))

        # Diagnostic stash — populated each forward when enabled.
        self._diag: dict[str, float] = {}
        self._diag_enabled: bool = False

        self._init_parameters()

    def _init_parameters(self) -> None:
        """v7 §R.5 init: K Kaiming, W_out Xavier-like, C zero (or non-zero)."""
        d = self.cfg.d
        # nanochat-style sqrt(3/d) uniform
        s_in = (3.0 / d) ** 0.5
        if self.cfg.gating_mode == "multiplicative":
            wout_in = self.h
        else:
            wout_in = self.cfg.r if self.cfg.no_base else (self.h + self.cfg.r)
        s_h = (3.0 / wout_in) ** 0.5
        nn.init.uniform_(self.K.weight, -s_in, s_in)
        nn.init.uniform_(self.W_out.weight, -s_h, s_h)
        # v8: lift projection P. Init small so initial gate ≈ 1 (preserves
        # vanilla-MLP cold start when C_init_std=0 and ensures gate excursion
        # stays bounded for nonzero C init).
        if self.W_d_proj is not None:
            nn.init.uniform_(self.W_d_proj.weight, -0.01, 0.01)
        # C: zero init for soft cold-start (delta=0 at step 0 but dC nonzero
        # through W_out_d gradient — works for BOTH with-base and no_base):
        #   no_base zero-C cold-start:
        #     y = W_out_d @ (λ·0) = 0  at step 1
        #     dL/dy != 0 (from MSE/CE loss)
        #     dL/dδ = W_out_d^T @ dL/dy != 0  (W_out_d random init)
        #     dL/dC[j,b,r] = B_b(z_j) * dL/dδ[r] != 0  → C learns from step 1
        #     dL/dW_out_d = δ^T @ dL/dy = 0 at step 1, but unblocks at step 2
        #   So zero-C is correct for no_base too — earlier code force-non-zero
        #   was based on a wrong analysis (Plan A Fix 1).
        # c_init_std (v8.B) takes precedence: if > 0, use that std, else
        # fall back to the v7 init_C_zero / std=0.02 path.
        if self.cfg.c_init_std > 0:
            nn.init.normal_(self.C, mean=0.0, std=float(self.cfg.c_init_std))
        elif self.cfg.init_C_zero:
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
        if self.cfg.no_base:
            # Plan A — pure-spline FFN.  W_out is Linear(r, d).
            from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
                FlashSplineDelta as _FD,
            )
            if self.cfg.spline_order != 2:
                raise NotImplementedError("no_base currently requires spline_order=2")
            delta = _FD.apply(
                z, self.C,
                float(self.cfg.grid_lo), float(self.cfg.grid_hi),
                int(self.cfg.G), float(self.cfg.lambda_scale),
                self.cfg.bwd_kernel,
            )                                           # [N, r]
            f = delta                                   # alias for diag below
            y = self.W_out(delta)                       # [N, d]
        elif self.cfg.gating_mode == "multiplicative":
            # v8 amendment — multiplicative spline gating of the base path.
            # See THEORY_v8_MULTIPLICATIVE_GATING.md.
            #   y = W_out @ ((1 + λ·(P · δ)) ⊙ ReLU²(z))
            # Note: FlashSplineDelta / reference both already multiply by λ
            # internally — `delta` below already contains λ.
            if self.cfg.spline_order != 2:
                raise NotImplementedError(
                    "multiplicative gating currently requires spline_order=2"
                )
            if self.cfg.use_kernel and z.is_cuda:
                from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
                    FlashSplineDelta as _FD,
                )
                delta = _FD.apply(
                    z, self.C,
                    float(self.cfg.grid_lo), float(self.cfg.grid_hi),
                    int(self.cfg.G), float(self.cfg.lambda_scale),
                    self.cfg.bwd_kernel,
                )                                       # [N, r]
            else:
                f_full = flash_spline_feature_reference(
                    z, self.C,
                    grid_lo=float(self.cfg.grid_lo),
                    grid_hi=float(self.cfg.grid_hi),
                    G=int(self.cfg.G),
                    activation=self.cfg.activation,
                    lambda_scale=float(self.cfg.lambda_scale),
                    spline_order=int(self.cfg.spline_order),
                )
                delta = f_full[:, self.h:]              # [N, r], includes λ
            delta_h = self.W_d_proj(delta)              # [N, h]
            a = _activation(z, self.cfg.activation)     # [N, h] base path
            gate = 1.0 + delta_h                        # [N, h]
            gated = gate * a                            # [N, h]
            f = gated                                   # alias for diag
            y = self.W_out(gated)                       # [N, d]
        elif (self.cfg.use_kernel and self.cfg.spline_order == 2
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
                getattr(self.cfg, "fwd_kernel", "auto"),  # default = native CUDA
            )                                           # [N, h+r]
            y = self.W_out(f)                           # [N, d]
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
            y = self.W_out(f)                           # [N, d]
        # Stash diagnostics (no grad path) — used by training loop hooks.
        if not self.training or self._diag_enabled:
            with torch.no_grad():
                G = int(self.cfg.G)
                if self.cfg.no_base:
                    # Plan A — no base path; f is delta [N, r] only.
                    base = torch.zeros_like(f[:, :0])
                    delta = f.float()
                    W_d = self.W_out.weight.float()         # [d, r]
                    rms_a = 0.0
                    yd = delta @ W_d.T
                    rms_d = float(yd.pow(2).mean().sqrt().item())
                    W_out_grad = self.W_out.weight.grad
                    w_a_grad_norm = 0.0
                    w_d_grad_norm = (W_out_grad.detach().norm().item()
                                     if W_out_grad is not None else 0.0)
                elif self.cfg.gating_mode == "multiplicative":
                    # v8 — f is the gated activation [N, h] = (1+λδ_h)·a.
                    # We re-derive `a` from z and `δ_h` from f/a to avoid
                    # plumbing extra intermediates from forward.
                    a = _activation(z, self.cfg.activation).float()  # [N, h]
                    gated = f.float()
                    eps = 1e-9
                    # gate = gated / a, but a may be 0 in dead-ReLU zones.
                    # Use a-masked gate to avoid div-by-zero noise.
                    a_active = (a.abs() > eps)
                    gate = torch.where(a_active, gated / a.clamp(min=eps),
                                       torch.ones_like(a))
                    delta_h = gate - 1.0
                    base = a
                    delta = delta_h
                    W_out = self.W_out.weight.float()  # [d, h]
                    # y_base = W_out @ a; y_delta = W_out @ ((gate-1)·a) = W_out @ (delta_h·a)
                    ya = a @ W_out.T
                    yd = (delta_h * a) @ W_out.T
                    rms_a = float(ya.pow(2).mean().sqrt().item())
                    rms_d = float(yd.pow(2).mean().sqrt().item())
                    # Gradient norms — for multiplicative there is no
                    # input-axis split.  W_a_grad_norm reports W_out (the
                    # output projection); W_d_grad_norm reports P (the lift).
                    W_out_grad = self.W_out.weight.grad
                    w_a_grad_norm = (W_out_grad.detach().norm().item()
                                     if W_out_grad is not None else 0.0)
                    P_grad = self.W_d_proj.weight.grad if self.W_d_proj is not None else None
                    w_d_grad_norm = (P_grad.detach().norm().item()
                                     if P_grad is not None else 0.0)
                else:
                    base = f[:, : self.h].float()
                    delta = f[:, self.h:].float()
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
                valid_bins = bin_idx[in_range_mask & torch.isfinite(z)]
                valid_bins = valid_bins[(valid_bins >= 0) & (valid_bins < G)]
                hist = torch.bincount(valid_bins.flatten(), minlength=G).float()
                p = hist / hist.sum().clamp_min(1.0)
                entropy = -(p * (p + 1e-12).log()).sum().item()
                edge_count = (
                    (valid_bins == 0).sum().item()
                    + (valid_bins == G - 1).sum().item()
                )
                self._diag["bin_entropy"] = float(entropy)
                self._diag["bin_entropy_norm"] = float(entropy / max(1e-9, math.log(G)))
                self._diag["edge_bin_frac"] = float(edge_count) / max(1, valid_bins.numel())
                self._diag["active_frac"] = float(in_range_mask.float().mean().item())
                # v8 — gate distribution diagnostics
                if self.cfg.gating_mode == "multiplicative":
                    a = _activation(z, self.cfg.activation).float()
                    gated = f.float()
                    a_active = (a.abs() > 1e-9)
                    gate = torch.where(a_active, gated / a.clamp(min=1e-9),
                                       torch.ones_like(a))
                    self._diag["gate_mean"] = float(gate.mean().item())
                    self._diag["gate_std"] = float(gate.std().item())
                    self._diag["gate_min"] = float(gate.min().item())
                    self._diag["gate_max"] = float(gate.max().item())
                    self._diag["gating_mode"] = "multiplicative"
                else:
                    self._diag["gating_mode"] = "additive"
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
