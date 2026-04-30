"""FullMix-Tucker FFN — Phase 1 reference implementation.

This is the PERMANENT REFERENCE (form B in K.0 of JHCG_REDESIGN_THEORY.md).
It is intentionally written as five clearly-separated PyTorch ops so that:

  1. it is easy to read, audit, and unit-test;
  2. autograd handles the backward pass automatically (no atomic adds, no
     custom Function);
  3. it serves as the numerical oracle for the Phase 2 fused Triton kernel
     (which must match this within bf16 1e-3 relative tolerance).

Mathematical form (per F.4.b):

    z_j      = (A x)_j                                # mixer, dense d -> m
    beta_jc  = sum_i Q[i,c] * B_i(z_j)                # B1 lerp on (L, R_b)
    xi_bc    = sum_j V[j,b] * beta_jc                  # GEMM
    eta_a    = sum_{b,c} C[a,b,c] * xi_bc              # core contraction
    y_k      = gamma * sum_a U[k,a] * eta_a            # readout

The key algebraic identity that makes this equivalent to a full Tucker
spline FFN is:

    W_kji = sum_{abc} U[k,a] * C[a,b,c] * V[j,b] * Q[i,c]
    y_k   = gamma * sum_{j,i} W_kji * B_i(z_j)

See JHCG_REDESIGN_THEORY.md K.0.1 for the equivalence proof.

Usage:
    from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN
    cfg = FullMixTuckerConfig(d=768, m=768, R_o=96, R_i=96, R_b=16, G=20)
    ffn = FullMixTuckerFFN(cfg)
    y = ffn(x)  # x: (..., d)  -> y: (..., d)

Memory note:
    The intermediate beta has shape (B*T, m, R_b) and is retained for
    backward by autograd.  At nanochat scale (B*T=8192, m=d=768, R_b=16),
    beta is ~200 MB / layer in bf16.  Wrap with torch.utils.checkpoint
    for Pattern Full (12 layers).  See K.0.3.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sparsespline_ffn.tucker_init import (
    init_mixer,
    init_tucker_factors,
    variance_preserving_spline_coef_init,
)


@dataclass
class FullMixTuckerConfig:
    """Hyperparameters for one FullMix-Tucker FFN layer.

    Attributes
    ----------
    d : int
        Residual-stream dim (output dim of the FFN).
    m : int
        Mixer width (input to spline).  Must satisfy m >= d for non-compressive
        mixer; m < d is forbidden (re-introduces JHCG's Defect 1).
    R_o : int
        Tucker output rank (column rank of U).  Per F.4.b, this caps the
        per-layer FFN-update subspace dim.
    R_i : int
        Tucker input rank.
    R_b : int
        Tucker basis-mode rank.
    G : int
        Number of grid intervals.  L = G + k (B1: L = G + 1).
    grid_lo : float
        Lower end of spline domain.
    grid_hi : float
        Upper end of spline domain.
    use_mixer : bool
        If False, A is the identity (T_direct topology).  Used for the
        `direct` ablation cell.
    bias_in_mixer : bool
        Whether the mixer A has a bias.  Default False (matches MLP W1 in
        nanochat).
    """

    d: int
    m: int
    R_o: int
    R_i: int
    R_b: int
    G: int = 20
    grid_lo: float = -3.0
    grid_hi: float = 3.0
    use_mixer: bool = True
    bias_in_mixer: bool = False

    def __post_init__(self) -> None:
        if self.m < self.d:
            raise ValueError(
                f"m={self.m} < d={self.d} would re-introduce JHCG's "
                "input-side bottleneck (Defect 1).  Use m >= d."
            )
        if self.G < 2:
            raise ValueError(f"G={self.G} too small; need at least 2 intervals")
        if self.grid_hi <= self.grid_lo:
            raise ValueError(f"grid_hi ({self.grid_hi}) <= grid_lo ({self.grid_lo})")


class FullMixTuckerFFN(nn.Module):
    """Reference (form B) FullMix-Tucker FFN.

    Parameters
    ----------
    config : FullMixTuckerConfig
        Hyperparameters.

    Shapes
    ------
    Input  x : (..., d)
    Output y : (..., d)

    Parameters
    ----------
    A : (m, d)            mixer (or identity if use_mixer=False)
    Q : (L, R_b)          spline-mode factor    (L = G + 1 for B1)
    V : (m, R_i)          input-mode factor
    C : (R_o, R_i, R_b)   Tucker core
    U : (d, R_o)          output-mode factor (the readout)
    gamma : (1,)          per-layer scalar gain
    """

    def __init__(self, config: FullMixTuckerConfig) -> None:
        super().__init__()
        self.cfg = config
        d, m = config.d, config.m
        R_o, R_i, R_b = config.R_o, config.R_i, config.R_b
        G = config.G
        L = G + 1  # B1 basis count

        # Mixer A: d -> m (or identity).
        if config.use_mixer:
            self.A = nn.Linear(d, m, bias=config.bias_in_mixer)
        else:
            if m != d:
                raise ValueError(
                    f"use_mixer=False (T_direct) requires m == d; got m={m}, d={d}"
                )
            self.A = None  # type: ignore[assignment]

        # Tucker factors and spline-mode lookup table.
        self.Q = nn.Parameter(torch.empty(L, R_b))
        self.V = nn.Parameter(torch.empty(m, R_i))
        self.C = nn.Parameter(torch.empty(R_o, R_i, R_b))
        self.U = nn.Parameter(torch.empty(d, R_o))
        self.gamma = nn.Parameter(torch.ones(1))

        # Grid endpoints stored as buffers (non-learnable for now; see L.4
        # for the future EMA-adaptive grid plan).
        self.register_buffer("grid_lo", torch.tensor(float(config.grid_lo)))
        self.register_buffer("grid_hi", torch.tensor(float(config.grid_hi)))

        self._L = L
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Apply variance-preserving + orthogonal-factor inits per L.4."""
        if self.A is not None:
            init_mixer(self.A)
        # Tucker factors first (they set the variance through V, U, C).
        init_tucker_factors(self.V, self.C, self.U, orthogonal_factors=True)
        # Then Q with the calibrated sigma that accounts for the full pipeline.
        variance_preserving_spline_coef_init(
            self.Q, d=self.cfg.d, R_o=self.cfg.R_o
        )
        with torch.no_grad():
            self.gamma.fill_(1.0)

    # ------------------------------------------------------------------
    # Reference forward — five clearly-named stages
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Five-stage reference forward.  x: (..., d) -> y: (..., d)."""
        original_shape = x.shape
        d = self.cfg.d
        if original_shape[-1] != d:
            raise ValueError(
                f"input last-dim {original_shape[-1]} != configured d={d}"
            )

        # Flatten leading dims so the einsums are over a single token axis.
        x_flat = x.reshape(-1, d)  # (N, d), N = product of leading dims

        # Stage 1 — mixer: z = A x.  Output (N, m).
        if self.A is not None:
            z = self.A(x_flat)
        else:
            z = x_flat  # identity mixer (T_direct ablation)

        # Stage 2 — B1 spline lookup: beta_{j,c} = sum_i Q[i,c] * B_i(z_j).
        # For B1 only two basis are active per input scalar:
        #   beta = (1 - t) * Q[bin] + t * Q[bin+1]
        bin_idx, t = self._bin_and_frac(z)  # both (N, m)
        Q0 = self.Q[bin_idx]                # (N, m, R_b)
        Q1 = self.Q[bin_idx + 1]            # (N, m, R_b)
        beta = torch.lerp(Q0, Q1, t.unsqueeze(-1))  # (N, m, R_b)

        # Stage 3 — input-mode contraction: xi = V^T beta.  Output (N, R_i, R_b).
        xi = torch.einsum("nmc, mb -> nbc", beta, self.V)

        # Stage 4 — core contraction: eta = C : xi.  Output (N, R_o).
        eta = torch.einsum("nbc, abc -> na", xi, self.C)

        # Stage 5 — readout: y = gamma * U eta.  Output (N, d).
        y_flat = (eta @ self.U.t()) * self.gamma

        return y_flat.reshape(original_shape)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bin_and_frac(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (bin_idx, frac) for B1 lookup.

        z   : (N, m)   pre-spline activations
        bin_idx in [0, G-1] (long)
        frac in [0, 1)  (same dtype as z)
        """
        G = self.cfg.G
        # Map z linearly into [0, G] then floor; clamp for out-of-range safety.
        scale = G / (self.grid_hi - self.grid_lo)
        u = (z - self.grid_lo) * scale  # (N, m), real-valued
        bin_idx = u.floor().to(torch.long).clamp_(min=0, max=G - 1)
        # frac = u - bin_idx (cast back to z's dtype to keep autograd clean)
        frac = (u - bin_idx.to(u.dtype)).clamp_(min=0.0, max=1.0)
        return bin_idx, frac

    # ------------------------------------------------------------------
    # Diagnostics — used by tests and by training loops to verify F.4.b
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct_dense_W(self) -> torch.Tensor:
        """Materialize the equivalent dense Tucker tensor W_{kji}.

        W has shape (d, m, L).  This is for unit tests only — at training
        scale (d=768, m=768, L=21) it is 26 MB / layer in fp32 and should
        not be used in any forward path.
        """
        return torch.einsum(
            "ka, abc, jb, ic -> kji", self.U, self.C, self.V, self.Q
        )

    @torch.no_grad()
    def output_subspace_dim(self) -> int:
        """Return rank(U) — an upper bound on per-layer FFN-update subspace dim.

        Per F.4.b, output is constrained to col-space(U), so dim <= R_o.
        Numerically rank-deficient U would mean even less effective rank.
        """
        return int(torch.linalg.matrix_rank(self.U).item())

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"d={c.d}, m={c.m}, R=({c.R_o},{c.R_i},{c.R_b}), "
            f"G={c.G}, L={c.G + 1}, mixer={c.use_mixer}"
        )


__all__ = ["FullMixTuckerConfig", "FullMixTuckerFFN"]
