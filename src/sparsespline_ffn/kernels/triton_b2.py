"""Triton kernels for B2 (quadratic) per-channel spline activation.

This is the kernel suite for ``SimpleSplineMLP`` — y = W_d · spline(W_u · x)
with a per-channel quadratic B-spline as the activation.

Math (uniform knots; index-shifted so all indices ≥ 0):
    L = G + 2 spline coefficients per channel
    For input z, u = scale·(z - grid_lo) clipped to [0, G],
    bin = floor(u), τ = u - bin in [0, 1]
    Three active basis (after shift):
        B0(u) = (1 - τ)²/2          → coef Q[c, bin]
        B1(u) = (1 + 2τ - 2τ²)/2    → coef Q[c, bin+1]
        B2(u) = τ²/2                → coef Q[c, bin+2]
    Sum = 1 ∀τ (partition of unity).
    y(z, c) = Q[c, bin]·B0 + Q[c, bin+1]·B1 + Q[c, bin+2]·B2

Backward:
    dQ[c, bin]   += dy(z, c) · B0
    dQ[c, bin+1] += dy(z, c) · B1
    dQ[c, bin+2] += dy(z, c) · B2
    dz(z, c)      = dy(z, c) · scale · (Q[c, bin]·(τ-1) + Q[c, bin+1]·(1-2τ) + Q[c, bin+2]·τ)

Per-channel: Q is shape (H, L) — every hidden channel has its own coefficients,
unlike the FullMix-Tucker form which factorizes Q across channels.  This is
the key win for SimpleSpline: each channel learns its own activation shape.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_E": 64},  num_warps=4),
        triton.Config({"BLOCK_E": 128}, num_warps=4),
        triton.Config({"BLOCK_E": 256}, num_warps=4),
        triton.Config({"BLOCK_E": 128}, num_warps=8),
        triton.Config({"BLOCK_E": 256}, num_warps=8),
    ],
    key=["E", "H", "G"],
)
@triton.jit
def _b2_forward_kernel(
    z_ptr,          # (E, H)
    Q_ptr,          # (H, L)  L = G + 2
    y_ptr,          # (E, H)
    E: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    L: tl.constexpr,
    grid_lo: tl.constexpr,
    scale: tl.constexpr,
    z_stride_e: tl.constexpr,
    Q_stride_h: tl.constexpr,
    y_stride_e: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """One program == one (E_tile, single channel).

    Grid: (cdiv(E, BLOCK_E), H).  We tile E (token positions) and process
    one channel per program — Q[h, :] then sits hot in registers.
    """
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)

    e_offsets = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offsets < E

    # Load z[e, h] for this channel
    z_ptrs = z_ptr + e_offsets * z_stride_e + pid_h
    z = tl.load(z_ptrs, mask=e_mask, other=0.0).to(tl.float32)

    # u, bin, tau (clamp to [0, G-1] so bin+2 is always valid index in [0..L-1])
    u = (z - grid_lo) * scale
    u = tl.minimum(tl.maximum(u, 0.0), tl.cast(G - 1, tl.float32))
    bin_idx = tl.floor(u).to(tl.int32)
    tau = u - tl.cast(bin_idx, tl.float32)

    # B-spline basis values (always non-negative, sum to 1).
    one_minus_tau = 1.0 - tau
    B0 = one_minus_tau * one_minus_tau * 0.5
    B1 = (1.0 + 2.0 * tau - 2.0 * tau * tau) * 0.5
    B2 = tau * tau * 0.5

    # Gather Q[h, bin], Q[h, bin+1], Q[h, bin+2]
    Q_base = Q_ptr + pid_h * Q_stride_h
    Q0 = tl.load(Q_base + bin_idx,     mask=e_mask, other=0.0).to(tl.float32)
    Q1 = tl.load(Q_base + bin_idx + 1, mask=e_mask, other=0.0).to(tl.float32)
    Q2 = tl.load(Q_base + bin_idx + 2, mask=e_mask, other=0.0).to(tl.float32)

    y = Q0 * B0 + Q1 * B1 + Q2 * B2

    # Cast back to z's dtype on store (kernel auto-casts via store)
    y_ptrs = y_ptr + e_offsets * y_stride_e + pid_h
    tl.store(y_ptrs, y, mask=e_mask)


def b2_forward(z: torch.Tensor, Q: torch.Tensor,
               grid_lo: float, grid_hi: float, G: int) -> torch.Tensor:
    """B2 spline forward.

    Parameters
    ----------
    z       : (..., H) float tensor (any leading dims OK; will be flattened)
    Q       : (H, L) float tensor where L = G + 2
    grid_lo : float lower bound of input range
    grid_hi : float upper bound of input range
    G       : int number of grid intervals; L must equal G + 2

    Returns
    -------
    y : same shape as z
    """
    assert Q.dim() == 2
    H, L = Q.shape
    assert L == G + 2, f"L={L} != G+2={G+2}"
    out_shape = z.shape
    z_flat = z.reshape(-1, H).contiguous()
    E = z_flat.shape[0]
    y_flat = torch.empty_like(z_flat)
    scale = G / (grid_hi - grid_lo)

    grid = lambda meta: (triton.cdiv(E, meta["BLOCK_E"]), H)  # noqa: E731
    _b2_forward_kernel[grid](
        z_flat,
        Q,
        y_flat,
        E=E,
        H=H,
        G=G,
        L=L,
        grid_lo=float(grid_lo),
        scale=float(scale),
        z_stride_e=z_flat.stride(0),
        Q_stride_h=Q.stride(0),
        y_stride_e=y_flat.stride(0),
    )
    return y_flat.view(out_shape)


# ---------------------------------------------------------------------------
# Backward kernel: dQ + dz produced together (similar to b1_backward_dq_dt)
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_E": 64},  num_warps=4),
        triton.Config({"BLOCK_E": 128}, num_warps=4),
        triton.Config({"BLOCK_E": 256}, num_warps=4),
        triton.Config({"BLOCK_E": 128}, num_warps=8),
    ],
    key=["E", "H", "G"],
    reset_to_zero=["dQ_ptr"],
)
@triton.jit
def _b2_backward_dq_dz_kernel(
    z_ptr,           # (E, H)  -- saved from forward
    Q_ptr,           # (H, L)
    dy_ptr,          # (E, H)  -- upstream gradient (dy/dx of next layer)
    dQ_ptr,          # (H, L) fp32 accumulator
    dz_ptr,          # (E, H)  -- gradient w.r.t. z
    E: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    L: tl.constexpr,
    grid_lo: tl.constexpr,
    scale: tl.constexpr,
    z_stride_e: tl.constexpr,
    Q_stride_h: tl.constexpr,
    dy_stride_e: tl.constexpr,
    dQ_stride_h: tl.constexpr,
    dz_stride_e: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """One program owns a tile of (BLOCK_E tokens, single channel).

    Per-channel processing means dQ[h, :] gets atomic-added by O(num_programs)
    times (one per E_tile in this h), much less contention than the global
    dQ in B1 which is across all (n, m) source elements.
    """
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)

    e_offsets = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offsets < E

    # Load z and dy for this (tile, channel)
    z_ptrs = z_ptr + e_offsets * z_stride_e + pid_h
    z = tl.load(z_ptrs, mask=e_mask, other=0.0).to(tl.float32)
    dy_ptrs = dy_ptr + e_offsets * dy_stride_e + pid_h
    dy = tl.load(dy_ptrs, mask=e_mask, other=0.0).to(tl.float32)

    u_raw = (z - grid_lo) * scale
    G_max = tl.cast(G - 1, tl.float32)
    # Track clamp boundary so dz can match PyTorch's clamp-gradient
    # convention (zero gradient outside the valid range).
    in_range = (u_raw >= 0.0) & (u_raw <= G_max)
    u = tl.minimum(tl.maximum(u_raw, 0.0), G_max)
    bin_idx = tl.floor(u).to(tl.int32)
    tau = u - tl.cast(bin_idx, tl.float32)

    # Basis values (for dQ contribution)
    one_minus_tau = 1.0 - tau
    B0 = one_minus_tau * one_minus_tau * 0.5
    B1 = (1.0 + 2.0 * tau - 2.0 * tau * tau) * 0.5
    B2 = tau * tau * 0.5

    # Load Q[h, bin], Q[h, bin+1], Q[h, bin+2] for dz computation.
    Q_base = Q_ptr + pid_h * Q_stride_h
    Q0 = tl.load(Q_base + bin_idx,     mask=e_mask, other=0.0).to(tl.float32)
    Q1 = tl.load(Q_base + bin_idx + 1, mask=e_mask, other=0.0).to(tl.float32)
    Q2 = tl.load(Q_base + bin_idx + 2, mask=e_mask, other=0.0).to(tl.float32)

    # ---- dz: per-element ------------------------------------------------
    # dB0/du = τ-1, dB1/du = 1-2τ, dB2/du = τ; du/dz = scale.
    dB0_du = tau - 1.0
    dB1_du = 1.0 - 2.0 * tau
    dB2_du = tau
    dz = dy * scale * (Q0 * dB0_du + Q1 * dB1_du + Q2 * dB2_du)
    # Zero out dz at clamp boundaries (matches PyTorch's clamp gradient).
    dz = tl.where(in_range, dz, 0.0)
    dz_ptrs = dz_ptr + e_offsets * dz_stride_e + pid_h
    tl.store(dz_ptrs, dz, mask=e_mask)

    # ---- dQ: atomic-add into per-channel Q[h, bin..bin+2] ---------------
    contrib0 = dy * B0
    contrib1 = dy * B1
    contrib2 = dy * B2
    # dQ[h, k] is row of (H, L) at offset pid_h * dQ_stride_h + k.
    # We accumulate three positions per element with atomic_add to avoid
    # cross-block races.  Per-channel L ≤ G+2 (≤ 22 default), small.
    dQ_base = dQ_ptr + pid_h * dQ_stride_h
    # Build per-row reductions: for each row k in [0, L), sum contributions
    # whose target bin is exactly k (resp. k-1 / k-2 for the three offsets).
    # Triton-friendly loop over L.
    for k in tl.static_range(L):
        # Token contributes to dQ[h, k] iff one of its bin offsets equals k:
        m0 = (bin_idx == k)         & e_mask
        m1 = (bin_idx == (k - 1))   & e_mask  # bin+1 == k -> bin == k-1
        m2 = (bin_idx == (k - 2))   & e_mask  # bin+2 == k -> bin == k-2
        acc = tl.sum(
            tl.where(m0, contrib0, 0.0)
            + tl.where(m1, contrib1, 0.0)
            + tl.where(m2, contrib2, 0.0)
        )
        # Single atomic per (program, k)
        tl.atomic_add(dQ_base + k, acc)


def b2_backward_dq_dz(
    z: torch.Tensor,
    Q: torch.Tensor,
    dy: torch.Tensor,
    grid_lo: float,
    grid_hi: float,
    G: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dQ and dz for the B2 spline activation.

    Returns
    -------
    dQ : (H, L) fp32 — to be cast to Q's dtype downstream.
    dz : same shape as z, in z's dtype.
    """
    assert Q.dim() == 2
    H, L = Q.shape
    assert L == G + 2
    in_shape = z.shape
    z_flat = z.reshape(-1, H).contiguous()
    dy_flat = dy.reshape(-1, H).contiguous()
    E = z_flat.shape[0]

    dz_flat = torch.empty_like(z_flat)
    dQ = torch.zeros((H, L), dtype=torch.float32, device=Q.device)
    scale = G / (grid_hi - grid_lo)

    grid = lambda meta: (triton.cdiv(E, meta["BLOCK_E"]), H)  # noqa: E731
    _b2_backward_dq_dz_kernel[grid](
        z_flat,
        Q,
        dy_flat,
        dQ,
        dz_flat,
        E=E,
        H=H,
        G=G,
        L=L,
        grid_lo=float(grid_lo),
        scale=float(scale),
        z_stride_e=z_flat.stride(0),
        Q_stride_h=Q.stride(0),
        dy_stride_e=dy_flat.stride(0),
        dQ_stride_h=dQ.stride(0),
        dz_stride_e=dz_flat.stride(0),
    )
    return dQ, dz_flat.view(in_shape)


__all__ = ["b2_forward", "b2_backward_dq_dz"]
