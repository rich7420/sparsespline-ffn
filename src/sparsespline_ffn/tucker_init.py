"""Variance-preserving initialization for FullMix-Tucker FFN.

References:
- KAT (Yang & Wang, ICLR 2025, arxiv 2409.10594) on the variance-preserving
  init problem for KAN-style activations.
- ECCV 2020 (arxiv 2008.05441) on CP instability and the SVD warm-start
  recipe for tensor-decomposed factors.

See JHCG_REDESIGN_THEORY.md Part L.4.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def variance_preserving_spline_coef_init(
    Q: torch.Tensor,
    *,
    d: int,
    R_o: int,
    target_output_var: float = 1.0,
) -> None:
    """Initialize spline-mode factor Q so post-FFN output has target variance.

    Derivation (assuming orthogonal-column V, U init with entry std 1/sqrt(.),
    and C ~ N(0, 1/(R_i R_b)), input z ~ N(0, 1)):

        Var[beta_jc]  = sigma_c^2 * E[B0^2 + B1^2]
                      = sigma_c^2 * 2/3                     (B1 uniform-t)
        Var[xi_bc]    = Var[beta]                            (V orthog)
        Var[eta_a]    = Var[xi]                              (C contracts R_i*R_b
                                                                independent terms)
        Var[y_k]      = gamma^2 * (R_o / d) * Var[eta_a]    (U orthog cols)

    Solving Var[y] = target with gamma=1:

        sigma_c = sqrt( 3 * d * target / (2 * R_o) )

    The previous v6-doc formula sqrt(3/m) only accounts for the spline edge
    and forgot the Tucker readout's variance shrinkage; it under-shoots by
    sqrt(d * m / (2 * R_o * m)) = sqrt(d / (2 R_o)).  At nanochat scale
    (d=768, R_o=96) the correction is sqrt(4) = 2x, but the original wrong
    formula was *much* smaller because it had `1/m` instead of `d/R_o`.

    Q has shape (L, R_b) where L = G + k.  Initialized in-place.
    """
    sigma_c = math.sqrt(3.0 * d * target_output_var / (2.0 * R_o))
    with torch.no_grad():
        Q.normal_(mean=0.0, std=sigma_c)


def init_mixer(A: nn.Linear) -> None:
    """Kaiming-uniform init for the non-compressive linear mixer A: d -> m.

    A is just a standard dense layer; no spline-specific magic needed.
    """
    nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
    if A.bias is not None:
        nn.init.zeros_(A.bias)


def init_tucker_factors(
    V: torch.Tensor,
    C: torch.Tensor,
    U: torch.Tensor,
    *,
    orthogonal_factors: bool = True,
) -> None:
    """Initialize Tucker factors V, C, U for FullMix-Tucker readout.

    Shapes:
      V: (m, R_i)
      C: (R_o, R_i, R_b)
      U: (d, R_o)

    Default policy (orthogonal_factors=True):
      - V, U: orthogonal columns (rotation-like; preserves variance under matmul)
      - C: small Gaussian (acts like a compressed mixing core; not orthogonal
        because it is 3D and the orthogonality constraint is shape-incompatible)

    The HOSVD warm-start path (see hosvd_warmstart_from_dense) overrides
    these defaults after a few hundred steps of dense pretraining.
    """
    with torch.no_grad():
        if orthogonal_factors:
            nn.init.orthogonal_(V)
            nn.init.orthogonal_(U)
        else:
            nn.init.kaiming_uniform_(V, a=math.sqrt(5))
            nn.init.kaiming_uniform_(U, a=math.sqrt(5))
        # Core: small Gaussian so initial output magnitude is controlled
        # and gradient signal is symmetric across modes.
        R_o, R_i, R_b = C.shape
        sigma_C = 1.0 / math.sqrt(R_i * R_b)
        C.normal_(mean=0.0, std=sigma_C)


def hosvd_warmstart_from_dense(
    W_dense: torch.Tensor,
    R_o: int,
    R_i: int,
    R_b: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HOSVD-initialize Tucker factors (U, V, Q-mode core) from a dense W.

    W_dense: (d, m, L) — the "full storage" spline tensor we want to compress.

    Returns:
      U: (d, R_o)        leading R_o left-singular vectors of mode-0 unfolding
      V: (m, R_i)        leading R_i left-singular vectors of mode-1 unfolding
      core: (R_o, R_i, R_b)  truncated core via mode-wise projection

    Q (basis-mode factor) is handled separately by the caller because in our
    parameterization Q is its own tensor (L, R_b), not contracted into the
    core.  The R_b axis of `core` corresponds to the basis mode.
    """
    assert W_dense.dim() == 3, "W_dense must be (d, m, L)"
    d, m, L = W_dense.shape

    # Mode-0 unfolding: (d, m*L)
    W0 = W_dense.reshape(d, m * L)
    U_full, _, _ = torch.linalg.svd(W0, full_matrices=False)
    U = U_full[:, :R_o].contiguous()

    # Mode-1 unfolding: (m, d*L) — note dim permutation
    W1 = W_dense.permute(1, 0, 2).reshape(m, d * L)
    V_full, _, _ = torch.linalg.svd(W1, full_matrices=False)
    V = V_full[:, :R_i].contiguous()

    # Mode-2 unfolding: (L, d*m) — for the basis-mode SVD
    W2 = W_dense.permute(2, 0, 1).reshape(L, d * m)
    Q_full, _, _ = torch.linalg.svd(W2, full_matrices=False)
    Q_basis = Q_full[:, :R_b].contiguous()  # (L, R_b)

    # Core via Tucker projection: core[a,b,c] = sum_{k,j,i} U[k,a] V[j,b] Q[i,c] W[k,j,i]
    core = torch.einsum("ka,jb,ic,kji->abc", U, V, Q_basis, W_dense)

    return U, V, core, Q_basis  # type: ignore[return-value]


__all__ = [
    "variance_preserving_spline_coef_init",
    "init_mixer",
    "init_tucker_factors",
    "hosvd_warmstart_from_dense",
]
