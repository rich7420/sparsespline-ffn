"""FlashSplineFeature backward — analytical reference (Task 5).

This is **not a Triton kernel yet** (B2.4) — it is the explicit
analytic-formula PyTorch implementation of the backward pass derived in
v7 §R.3.  The eventual Triton kernel will be tested against this exact
function (which itself is tested against autograd).

Why analytical (instead of just letting autograd do it)?
  - The Triton bwd kernel must implement *these specific formulas*,
    so we want a side-by-side reference that does not rely on autograd's
    chain rule (which would not exist inside Triton).
  - Stress-tests for collapsed / skewed bin distributions (atomic
    contention) need a non-autograd dC formula to verify against.

Formulas (v7 §R.3.0–R.3.3):

  delta_c = sum_j sum_{b in active(z_j)} B_b(tau_j) * C[j, b, c]
          = sum_j [B0(tau_j)*C[j,bin_j,c] + B1*C[j,bin_j+1,c] + B2*C[j,bin_j+2,c]]

  with tau_j = u_j - bin_j,   u_j = (z_j - grid_lo) * scale,   scale = G/(grid_hi-grid_lo)

  B0'(tau) = -(1-tau)
  B1'(tau) = 1 - 2*tau
  B2'(tau) = tau

For a backward gradient g_delta in R^[N, r] (gradient of loss w.r.t. delta):

  dC[j, b, c]    = sum_{n: bin_n,j == b - k for some k in {0,1,2}}
                     B_k(tau_n,j) * g_delta[n, c]

  In other words, for each token n and key j with active bins
  {bin_n,j, bin_n,j+1, bin_n,j+2}, accumulate
      dC[j, bin_n,j + k, :] += B_k(tau_n,j) * g_delta[n, :]
  for k in {0, 1, 2}.

  dz[n, j] (spline contribution) = (1/Delta) * sum_c g_delta[n, c]
                * (B0'*C[j,bin,c] + B1'*C[j,bin+1,c] + B2'*C[j,bin+2,c])

  where Delta = (grid_hi - grid_lo) / G is the bin width (so 1/Delta = scale).

Important: out-of-range tokens contribute 0 (multiplicative mask).
"""
from __future__ import annotations

import torch


def _b2_basis_and_bins(
    z: torch.Tensor, grid_lo: float, grid_hi: float, G: int,
):
    """Compute (bin_idx, B0, B1, B2, dB0, dB1, dB2, in_range, scale)."""
    scale = G / (grid_hi - grid_lo)
    u = (z - grid_lo) * scale
    in_range = (u >= 0.0) & (u <= float(G))
    u_clip = u.clamp(0.0, float(G - 1))
    bin_idx = u_clip.floor().to(torch.long)
    tau = (u_clip - bin_idx.to(u.dtype)).clamp(0.0, 1.0)

    omt = 1.0 - tau
    B0 = 0.5 * omt * omt
    B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
    B2 = 0.5 * tau * tau

    dB0 = -omt                # = -(1 - tau)
    dB1 = 1.0 - 2.0 * tau
    dB2 = tau                 # = tau

    return bin_idx, B0, B1, B2, dB0, dB1, dB2, in_range, scale


def flash_spline_delta_backward_ref(
    z: torch.Tensor,             # [N, h]
    C: torch.Tensor,             # [h, L, r]
    g_delta: torch.Tensor,       # [N, r], grad-of-loss w.r.t. delta
    grid_lo: float,
    grid_hi: float,
    G: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (dC, dz_spline) explicitly via the analytic formulas.

    dC      : [h, L, r] in float32
    dz_spline : [N, h] in z's dtype, gradient of loss w.r.t. z from the
              spline path only (does not include phi(z) contribution).
    """
    if z.dim() != 2:
        raise ValueError(f"z must be [N, h], got {tuple(z.shape)}")
    N, h = z.shape
    h_C, L, r = C.shape
    if h_C != h:
        raise ValueError(f"z.h={h} != C.h={h_C}")
    if L != G + 2:
        raise ValueError(f"L={L} should equal G+2={G + 2}")
    if g_delta.shape != (N, r):
        raise ValueError(f"g_delta must be [N, r]={N, r}, got {tuple(g_delta.shape)}")

    bin_idx, B0, B1, B2, dB0, dB1, dB2, in_range, scale = _b2_basis_and_bins(
        z.to(torch.float32), grid_lo, grid_hi, G,
    )
    # Forward mask: matches the reference's (u in [0, G]) — this is what
    # multiplies B0/B1/B2 inside the forward, and therefore appears in dC.
    mask_fwd = in_range.to(torch.float32)
    # Backward mask: autograd's clamp(0, G-1) kills the derivative whenever
    # u is outside [0, G-1], even if the forward keeps the token "active"
    # at u in (G-1, G].  Apply this mask only to dB (which controls dz),
    # not to B (which controls dC).
    z_f = z.to(torch.float32)
    u_for_mask = (z_f - grid_lo) * scale
    clamp_active = ((u_for_mask >= 0.0) & (u_for_mask <= float(G - 1))).to(torch.float32)
    B0 = B0 * mask_fwd; B1 = B1 * mask_fwd; B2 = B2 * mask_fwd
    dB0 = dB0 * clamp_active
    dB1 = dB1 * clamp_active
    dB2 = dB2 * clamp_active

    g_delta_f = g_delta.to(torch.float32)
    C_f = C.to(torch.float32)

    # ----- dC via scatter-add
    # For each (n, j), bins bin_idx[n,j] + k receive B_k * g_delta[n, :]
    dC = torch.zeros(h, L, r, dtype=torch.float32, device=z.device)

    h_idx = torch.arange(h, device=z.device).unsqueeze(0).expand(N, h)  # [N, h]

    # Flatten (n, j) for index_put into dC[j_flat, bin_flat, :]
    for k, B_k in enumerate([B0, B1, B2]):
        bin_k = bin_idx + k                  # [N, h]
        # contribution[n, j, c] = B_k[n, j] * g_delta[n, c]
        contrib = B_k.unsqueeze(-1) * g_delta_f.unsqueeze(1)  # [N, h, r]
        # scatter-add into dC[j, bin_k, :]
        dC.index_put_(
            (h_idx.flatten(), bin_k.flatten()),
            contrib.reshape(N * h, r),
            accumulate=True,
        )

    # ----- dz_spline via chain rule through tau
    # dL/dz_j = (1/Delta) * sum_c g_delta[n,c] *
    #            (dB0 * C[j, bin, c] + dB1 * C[j, bin+1, c] + dB2 * C[j, bin+2, c])
    # 1/Delta == scale.
    C0 = C_f[h_idx, bin_idx, :]      # [N, h, r]
    C1 = C_f[h_idx, bin_idx + 1, :]
    C2 = C_f[h_idx, bin_idx + 2, :]

    # inner product over c with g_delta
    inner0 = (C0 * g_delta_f.unsqueeze(1)).sum(dim=-1)  # [N, h]
    inner1 = (C1 * g_delta_f.unsqueeze(1)).sum(dim=-1)
    inner2 = (C2 * g_delta_f.unsqueeze(1)).sum(dim=-1)

    dz_spline = scale * (dB0 * inner0 + dB1 * inner1 + dB2 * inner2)
    dz_spline = dz_spline.to(z.dtype)
    return dC, dz_spline


__all__ = ["flash_spline_delta_backward_ref", "_b2_basis_and_bins"]
