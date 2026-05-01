"""Diagnostics for RL-Spline-KV and SimpleSpline / GLU FFN families.

Implements the metrics specified in:
  - v7 §R.1.4: rho_delta = RMS(W_delta @ delta) / RMS(W_a @ a)
               (the "is the spline branch load-bearing?" diagnostic)
  - v7 §R.5/R.6.10: bin occupancy histogram and entropy
                    (detects dead bins / grid coverage problems)
  - v7 §R.6.8: optimizer-group norms (C, W_delta, etc.)

These are intended to be cheap (callable every N training steps) and
return scalar / 1-D tensors suitable for logging to JSON.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# rho_delta: spline-branch load-bearing diagnostic (v7 §R.1.4)


def rms(t: torch.Tensor) -> float:
    """Root-mean-square of a tensor, returned as Python float."""
    return float(t.detach().to(torch.float32).pow(2).mean().sqrt().item())


def rho_delta_ratio(
    base_output: torch.Tensor,    # W_a @ a, shape [..., d]
    spline_output: torch.Tensor,  # W_delta @ (lambda * delta), shape [..., d]
) -> float:
    """Ratio RMS(spline_output) / RMS(base_output).

    v7 §R.1.4 pass criterion: rho_delta >= 0.20 in >= 8/12 layers at end
    of training.  If rho_delta collapses below 0.05 in most layers, the
    spline branch is decorative and the architecture is reducing to
    narrow MLP.
    """
    r_base = rms(base_output)
    r_spline = rms(spline_output)
    if r_base < 1e-12:
        return float("inf") if r_spline > 0 else 0.0
    return r_spline / r_base


def rho_delta_from_module(
    module,
    x: torch.Tensor,
) -> dict[str, float]:
    """Compute rho_delta by hooking the two W_out half-products inside an
    RLSplineKVReference module.

    Returns:
      {"rho_delta": float, "rms_base": float, "rms_spline": float}
    """
    # Forward pass with hooks on the two halves.
    # We use a manual forward that exposes both halves rather than
    # trying to introspect the module — cleaner and more robust.
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference,
    )
    cfg = module.cfg
    z = module.K(x.reshape(-1, cfg.d))
    f = flash_spline_feature_reference(
        z, module.C,
        grid_lo=float(cfg.grid_lo), grid_hi=float(cfg.grid_hi),
        G=int(cfg.G), activation=cfg.activation,
        lambda_scale=float(cfg.lambda_scale),
    )
    h = module.h
    # W_out[:, :h] @ a    and    W_out[:, h:] @ (lambda * delta)
    W = module.W_out.weight  # [d, h+r]
    base_out = F.linear(f[:, :h], W[:, :h])
    spline_out = F.linear(f[:, h:], W[:, h:])
    return {
        "rho_delta": rho_delta_ratio(base_out, spline_out),
        "rms_base": rms(base_out),
        "rms_spline": rms(spline_out),
    }


# ---------------------------------------------------------------------------
# Bin occupancy & entropy (v7 §R.6.10 dead-bin detection)


def bin_occupancy(
    z: torch.Tensor,        # [..., h]
    grid_lo: float,
    grid_hi: float,
    G: int,
) -> torch.Tensor:           # [G+2] int64 counts
    """Count how many tokens land in each B2-active bin.

    For a token with z in bin b, it contributes weight to bins {b, b+1, b+2}.
    We count the *primary* bin floor(u) only (the leftmost active bin),
    which is the standard per-token bin assignment.

    Returns histogram of length L = G + 2 (matches C's middle dim).
    """
    L = G + 2
    scale = G / (grid_hi - grid_lo)
    u = (z.detach().to(torch.float32) - grid_lo) * scale
    in_range = (u >= 0.0) & (u <= float(G))
    u_clip = u.clamp(0.0, float(G - 1))
    bin_idx = u_clip.floor().to(torch.long)
    # Only count in-range tokens; out-of-range are clamped (artificial bin)
    bin_idx = bin_idx[in_range].flatten()
    return torch.bincount(bin_idx, minlength=L)


def bin_entropy(occupancy: torch.Tensor) -> float:
    """Shannon entropy of the bin occupancy distribution, in nats.

    Uniform distribution over L bins gives entropy = log(L).
    A pathological "all in one bin" gives 0.
    Compare to log(L) to see how peaked the distribution is.
    """
    total = float(occupancy.sum().item())
    if total <= 0:
        return 0.0
    p = occupancy.to(torch.float64) / total
    p = p[p > 0]
    return float(-(p * p.log()).sum().item())


def dead_bin_fraction(
    occupancy: torch.Tensor,
    threshold_frac: float = 1e-3,
) -> float:
    """Fraction of bins that received < threshold_frac of total tokens.

    v7 §R.6.10: if dead_bin_fraction > 0.20, switch to EMA-quantile grid.
    """
    total = float(occupancy.sum().item())
    if total <= 0:
        return 1.0
    threshold = threshold_frac * total
    dead = (occupancy.to(torch.float64) < threshold).sum().item()
    return float(dead / occupancy.numel())


# ---------------------------------------------------------------------------
# Code-tensor norms (for monitoring C learning under C=0 cold start)


def code_norms(C: torch.Tensor) -> dict[str, float]:
    """Magnitude diagnostics for the local code tensor C [h, L, r].

    Useful for monitoring v7 §R.5 cold-start: |C| should grow from 0
    early in training, then stabilize.
    """
    Cf = C.detach().to(torch.float32)
    return {
        "frobenius": float(Cf.norm().item()),
        "mean_abs": float(Cf.abs().mean().item()),
        "max_abs": float(Cf.abs().max().item()),
    }


def grad_norm(param: torch.Tensor | None) -> float:
    """L2 norm of a parameter's gradient, or 0 if no grad attached."""
    if param is None or param.grad is None:
        return 0.0
    return float(param.grad.detach().to(torch.float32).norm().item())


# ---------------------------------------------------------------------------
# All-in-one snapshot for an RLSplineKVReference layer


@dataclass
class RLSplineKVSnapshot:
    rho_delta: float
    rms_base: float
    rms_spline: float
    bin_entropy_nats: float
    bin_entropy_max_nats: float       # = log(L)
    dead_bin_fraction: float
    C_frobenius: float
    C_mean_abs: float
    C_max_abs: float
    C_grad_norm: float
    W_delta_grad_norm: float          # W_out spline columns gradient norm
    K_grad_norm: float


def snapshot_rl_spline_kv(
    module,                # RLSplineKVReference (or v7 production module)
    x: torch.Tensor,       # current batch input [..., d]
) -> RLSplineKVSnapshot:
    """Take a full diagnostic snapshot.

    Should be called *after* a backward pass so gradients are populated.
    Forward computation here is a fresh pass for rho_delta and bin
    occupancy (does not affect training gradients because we use detach
    inside).
    """
    import math
    cfg = module.cfg

    # rho_delta on a forward pass (no grad)
    with torch.no_grad():
        rd = rho_delta_from_module(module, x)
        # bin occupancy on the same z
        z = module.K(x.reshape(-1, cfg.d))
        occ = bin_occupancy(z, float(cfg.grid_lo), float(cfg.grid_hi),
                             int(cfg.G))
        ent = bin_entropy(occ)
        max_ent = math.log(occ.numel()) if occ.numel() > 0 else 0.0
        dead = dead_bin_fraction(occ)

    # Code norms
    cn = code_norms(module.C)

    # Gradient norms (assume backward has been called recently)
    h = module.h
    # W_delta corresponds to W_out columns [h:].  We don't have a
    # separate Parameter for it, so compute its grad norm from the slice.
    W_grad = module.W_out.weight.grad
    if W_grad is not None:
        W_delta_g = float(W_grad[:, h:].detach().to(torch.float32).norm().item())
    else:
        W_delta_g = 0.0

    return RLSplineKVSnapshot(
        rho_delta=rd["rho_delta"],
        rms_base=rd["rms_base"],
        rms_spline=rd["rms_spline"],
        bin_entropy_nats=ent,
        bin_entropy_max_nats=max_ent,
        dead_bin_fraction=dead,
        C_frobenius=cn["frobenius"],
        C_mean_abs=cn["mean_abs"],
        C_max_abs=cn["max_abs"],
        C_grad_norm=grad_norm(module.C),
        W_delta_grad_norm=W_delta_g,
        K_grad_norm=grad_norm(module.K.weight),
    )


__all__ = [
    "rms", "rho_delta_ratio", "rho_delta_from_module",
    "bin_occupancy", "bin_entropy", "dead_bin_fraction",
    "code_norms", "grad_norm",
    "RLSplineKVSnapshot", "snapshot_rl_spline_kv",
]
