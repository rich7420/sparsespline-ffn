"""Triton kernel for the B1-spline backward dQ accumulation.

Replaces the slow PyTorch path
    dQ[bin,   :] += (1-t) * dbeta
    dQ[bin+1, :] += t     * dbeta
which lowers to ``aten::_index_put_impl_`` and gets ~97% of FullMix backward
time on RTX 3080 (see ``benchmarks/profile_backward.py``).

Algorithm: tile-and-reduce + atomic-add.
  - Flatten the (N, m) source dimension into E = N*m.
  - Each Triton program owns a tile of BLOCK_E source elements over an
    R_b channel slice of width BLOCK_RB.
  - Accumulate a private (L, BLOCK_RB) buffer in fp32 inside the program.
  - Atomic-add the private buffer into the global dQ[L, R_b] tensor.

Why this is fast: L is tiny (e.g. G=20 -> L=21).  The original PyTorch path
issues ~50M atomic adds against 21*R_b destinations; this kernel issues
~num_blocks * L * R_b atomics instead, and contention drops by >10x.

The kernel is mathematically equivalent to the PyTorch scatter (commutative
sum) but ordering of fp32 accumulations differs, so bit-exact reproducibility
across BLOCK_E choices is not guaranteed.  Numerical agreement is measured by
the correctness suite at fp32 1e-5 / bf16 5e-3.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_E": 64,  "BLOCK_RB": 16}, num_warps=4),
        triton.Config({"BLOCK_E": 128, "BLOCK_RB": 16}, num_warps=4),
        triton.Config({"BLOCK_E": 256, "BLOCK_RB": 16}, num_warps=4),
        triton.Config({"BLOCK_E": 128, "BLOCK_RB": 16}, num_warps=8),
        triton.Config({"BLOCK_E": 256, "BLOCK_RB": 16}, num_warps=8),
        triton.Config({"BLOCK_E": 128, "BLOCK_RB": 8},  num_warps=4),
        triton.Config({"BLOCK_E": 256, "BLOCK_RB": 8},  num_warps=4),
    ],
    key=["E", "R_b", "L"],
    reset_to_zero=["dQ_ptr"],  # critical: dQ is an accumulator output
)
@triton.jit
def _b1_backward_dq_kernel(
    bin_ptr,        # int64*, shape (E,)
    t_ptr,          # fp32/fp16/bf16*, shape (E,)
    dbeta_ptr,      # fp32/fp16/bf16*, shape (E, R_b)  (must be contiguous in R_b)
    dQ_ptr,         # fp32*, shape (L, R_b)  -- ALWAYS fp32 for accumulation
    E: tl.constexpr,
    R_b: tl.constexpr,
    L: tl.constexpr,
    dbeta_stride_e: tl.constexpr,
    dQ_stride_l: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_RB: tl.constexpr,
):
    """One program == one (E_tile, RB_tile) pair.

    Within the program we keep a private (L, BLOCK_RB) fp32 accumulator and
    issue at most L * BLOCK_RB atomic_adds to global dQ at the end.
    """
    pid_e = tl.program_id(0)
    pid_rb = tl.program_id(1)

    e_offsets = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)            # (BLOCK_E,)
    rb_offsets = pid_rb * BLOCK_RB + tl.arange(0, BLOCK_RB)        # (BLOCK_RB,)
    e_mask = e_offsets < E
    rb_mask = rb_offsets < R_b

    # Load this tile's source data.  Out-of-range elements are masked to 0,
    # bin index masked to 0 (does not write because dbeta is also 0).
    bin_e = tl.load(bin_ptr + e_offsets, mask=e_mask, other=0)              # (BLOCK_E,) int64
    t_e = tl.load(t_ptr + e_offsets, mask=e_mask, other=0.0).to(tl.float32) # (BLOCK_E,)
    one_minus_t = 1.0 - t_e

    # dbeta tile: (BLOCK_E, BLOCK_RB) fp32
    dbeta_ptrs = dbeta_ptr + e_offsets[:, None] * dbeta_stride_e + rb_offsets[None, :]
    dbeta_tile = tl.load(
        dbeta_ptrs,
        mask=e_mask[:, None] & rb_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Per-(e) two contributions:
    #   row bin   : (1-t) * dbeta
    #   row bin+1 : t     * dbeta
    contrib_lo = one_minus_t[:, None] * dbeta_tile        # (BLOCK_E, BLOCK_RB)
    contrib_hi = t_e[:, None]         * dbeta_tile

    # Build the private accumulator by reducing across BLOCK_E for each L row.
    # We loop over rows because L is small and known at compile time.
    # For each row l we:
    #   mask_lo = (bin_e == l)
    #   mask_hi = (bin_e+1 == l)  i.e. bin_e == l-1
    #   acc[l] += sum over e of (mask_lo*contrib_lo + mask_hi*contrib_hi)
    #
    # This produces (L, BLOCK_RB) values that we atomic-add to global dQ.
    for l in tl.static_range(L):
        m_lo = (bin_e == l) & e_mask
        m_hi = (bin_e == (l - 1)) & e_mask
        # Pick contributions for this row only.
        row_contrib = tl.where(m_lo[:, None], contrib_lo, 0.0)
        row_contrib += tl.where(m_hi[:, None], contrib_hi, 0.0)
        # Reduce across BLOCK_E -> (BLOCK_RB,)
        acc_row = tl.sum(row_contrib, axis=0)  # (BLOCK_RB,) fp32

        # Atomic-add into dQ[l, rb_offsets].
        out_ptrs = dQ_ptr + l * dQ_stride_l + rb_offsets
        tl.atomic_add(out_ptrs, acc_row, mask=rb_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def b1_backward_dq(
    bin_idx: torch.Tensor,
    t: torch.Tensor,
    dbeta: torch.Tensor,
    *,
    L: int,
) -> torch.Tensor:
    """Accumulate the B1-spline lookup backward into dQ.

    Parameters
    ----------
    bin_idx : (N, m) or (E,) long tensor with values in [0, L-2]
    t       : (N, m) or (E,) tensor (any float dtype)
    dbeta   : (N, m, R_b) or (E, R_b) tensor with the same float dtype as t
              (R_b must be the inner-most contiguous dim)
    L       : grid+1 (B1: L = G + 1).  bin_idx must satisfy bin_idx + 1 < L.

    Returns
    -------
    dQ : (L, R_b) fp32 tensor.  Caller is expected to ``.to(dtype)`` if needed.

    Notes
    -----
    - Input is flattened to E = prod(bin_idx.shape).
    - Output is always fp32 to keep accumulation precision; downstream casts
      back to the parameter dtype.
    - dbeta is made contiguous on the last dim if needed (cost included in
      the wrapper, not the kernel).
    """
    if bin_idx.dim() > 1:
        bin_idx = bin_idx.reshape(-1)
        t_flat = t.reshape(-1)
        dbeta_flat = dbeta.reshape(-1, dbeta.shape[-1])
    else:
        t_flat = t
        dbeta_flat = dbeta
    if dbeta_flat.stride(-1) != 1:
        dbeta_flat = dbeta_flat.contiguous()
    if not bin_idx.is_contiguous():
        bin_idx = bin_idx.contiguous()
    if not t_flat.is_contiguous():
        t_flat = t_flat.contiguous()

    E = bin_idx.numel()
    R_b = dbeta_flat.shape[-1]
    assert dbeta_flat.shape[0] == E, (
        f"dbeta first-dim {dbeta_flat.shape[0]} != E={E}"
    )
    assert bin_idx.dtype == torch.int64, (
        f"bin_idx must be int64 long, got {bin_idx.dtype}"
    )

    device = bin_idx.device
    dQ = torch.zeros((L, R_b), dtype=torch.float32, device=device)

    # Triton autotune picks BLOCK_E / BLOCK_RB; we just give the launch grid.
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(E, meta["BLOCK_E"]),
        triton.cdiv(R_b, meta["BLOCK_RB"]),
    )

    _b1_backward_dq_kernel[grid](
        bin_idx,
        t_flat,
        dbeta_flat,
        dQ,
        E=E,
        R_b=R_b,
        L=L,
        dbeta_stride_e=dbeta_flat.stride(0),
        dQ_stride_l=dQ.stride(0),
    )
    return dQ


__all__ = ["b1_backward_dq"]
