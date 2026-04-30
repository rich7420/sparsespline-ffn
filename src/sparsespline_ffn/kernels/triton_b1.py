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


# ===========================================================================
# Forward kernel (Tier 2): fused gather + lerp.
#
# Replaces the PyTorch path
#     Q0 = Q[bin],  Q1 = Q[bin+1],  beta = (1-t)*Q0 + t*Q1
# which spawns 3 kernel launches (2 index_select + 1 lerp) and reads each
# Q row twice from global memory.  The fused kernel does it in one pass.
# ===========================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_E": 128, "BLOCK_RB": 16}, num_warps=4),
        triton.Config({"BLOCK_E": 256, "BLOCK_RB": 16}, num_warps=4),
        triton.Config({"BLOCK_E": 512, "BLOCK_RB": 16}, num_warps=4),
        triton.Config({"BLOCK_E": 256, "BLOCK_RB": 16}, num_warps=8),
        triton.Config({"BLOCK_E": 512, "BLOCK_RB": 16}, num_warps=8),
        triton.Config({"BLOCK_E": 1024, "BLOCK_RB": 16}, num_warps=8),
    ],
    key=["E", "R_b"],
)
@triton.jit
def _b1_forward_kernel(
    Q_ptr,          # any-float*, shape (L, R_b)
    bin_ptr,        # int64*, shape (E,)
    t_ptr,          # any-float*, shape (E,)
    out_ptr,        # any-float* (same dtype as t/Q), shape (E, R_b)
    E: tl.constexpr,
    R_b: tl.constexpr,
    Q_stride_l: tl.constexpr,
    out_stride_e: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_RB: tl.constexpr,
):
    """One program == one (E_tile, RB_tile)."""
    pid_e = tl.program_id(0)
    pid_rb = tl.program_id(1)

    e_offsets = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    rb_offsets = pid_rb * BLOCK_RB + tl.arange(0, BLOCK_RB)
    e_mask = e_offsets < E
    rb_mask = rb_offsets < R_b

    # Load source.  bin_idx indexes into Q rows; t is per-token weight.
    bin_e = tl.load(bin_ptr + e_offsets, mask=e_mask, other=0)
    t_e = tl.load(t_ptr + e_offsets, mask=e_mask, other=0.0).to(tl.float32)

    # Load Q[bin, rb_tile] and Q[bin+1, rb_tile].
    # Q is (L, R_b) row-major; row stride == Q_stride_l, column stride == 1.
    q0_ptrs = Q_ptr + bin_e[:, None] * Q_stride_l + rb_offsets[None, :]
    q1_ptrs = Q_ptr + (bin_e + 1)[:, None] * Q_stride_l + rb_offsets[None, :]
    Q0 = tl.load(q0_ptrs, mask=e_mask[:, None] & rb_mask[None, :], other=0.0).to(tl.float32)
    Q1 = tl.load(q1_ptrs, mask=e_mask[:, None] & rb_mask[None, :], other=0.0).to(tl.float32)

    # Match PyTorch's torch.lerp exact formula: Q0 + t*(Q1 - Q0).
    # (Algebraically same as (1-t)*Q0 + t*Q1, but fp32 rounding matches
    # PyTorch bit-for-bit, keeping our equivalence audit at strict 1e-7.)
    beta = Q0 + t_e[:, None] * (Q1 - Q0)

    out_ptrs = out_ptr + e_offsets[:, None] * out_stride_e + rb_offsets[None, :]
    tl.store(out_ptrs, beta, mask=e_mask[:, None] & rb_mask[None, :])


def b1_forward(
    Q: torch.Tensor,
    bin_idx: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Fused B1-spline forward: beta = (1-t)*Q[bin] + t*Q[bin+1].

    Parameters
    ----------
    Q       : (L, R_b) tensor (any float dtype)
    bin_idx : (..., m) int64 tensor with values in [0, L-2]
    t       : (..., m) tensor matching Q's dtype (or castable)

    Returns
    -------
    beta : (..., m, R_b) tensor in t's dtype
    """
    out_shape = (*bin_idx.shape, Q.shape[-1])
    flat_bin = bin_idx.reshape(-1).contiguous()
    flat_t = t.reshape(-1).contiguous().to(t.dtype)

    E = flat_bin.numel()
    R_b = Q.shape[-1]
    assert flat_bin.dtype == torch.int64

    # Output dtype follows t (matches the existing torch.lerp behaviour).
    out_flat = torch.empty((E, R_b), dtype=t.dtype, device=Q.device)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(E, meta["BLOCK_E"]),
        triton.cdiv(R_b, meta["BLOCK_RB"]),
    )
    _b1_forward_kernel[grid](
        Q,
        flat_bin,
        flat_t,
        out_flat,
        E=E,
        R_b=R_b,
        Q_stride_l=Q.stride(0),
        out_stride_e=out_flat.stride(0),
    )
    return out_flat.view(out_shape)


# ===========================================================================
# Tier 3: fused backward producing both dQ and dt in one pass.
#
# Eliminates the need to save Q0, Q1 in the autograd ctx (saves ~384 MB
# activation memory per layer at nanochat scale: 2 * N * m * R_b * 2 bytes
# bf16 with N=8192, m=768, R_b=16 -> 384 MB), and folds the
# ``((Q1-Q0) * dbeta).sum(-1)`` ops into the same launch as dQ.
#
# Each program owns BLOCK_E tokens with the FULL R_b channel slice.  We
# keep R_b == BLOCK_RB so dt is fully computed within one program (no
# cross-program reduction needed).
# ===========================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_E": 64},  num_warps=4),
        triton.Config({"BLOCK_E": 128}, num_warps=4),
        triton.Config({"BLOCK_E": 256}, num_warps=4),
        triton.Config({"BLOCK_E": 128}, num_warps=8),
        triton.Config({"BLOCK_E": 256}, num_warps=8),
    ],
    key=["E", "R_b", "L"],
    reset_to_zero=["dQ_ptr"],  # dQ is an accumulator
)
@triton.jit
def _b1_backward_dq_dt_kernel(
    Q_ptr,          # (L, R_b)
    bin_ptr,        # (E,) int64
    t_ptr,          # (E,)
    dbeta_ptr,      # (E, R_b)
    dQ_ptr,         # (L, R_b) fp32 accumulator
    dt_ptr,         # (E,) per-token gradient (output, no atomic needed)
    E: tl.constexpr,
    R_b: tl.constexpr,
    L: tl.constexpr,
    Q_stride_l: tl.constexpr,
    dbeta_stride_e: tl.constexpr,
    dQ_stride_l: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_RB: tl.constexpr,
):
    pid_e = tl.program_id(0)

    e_offsets = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    rb_offsets = tl.arange(0, BLOCK_RB)
    e_mask = e_offsets < E
    rb_mask = rb_offsets < R_b

    bin_e = tl.load(bin_ptr + e_offsets, mask=e_mask, other=0)
    t_e = tl.load(t_ptr + e_offsets, mask=e_mask, other=0.0).to(tl.float32)
    one_minus_t = 1.0 - t_e

    # Load dbeta tile (BLOCK_E, BLOCK_RB) in fp32.
    dbeta_ptrs = (dbeta_ptr + e_offsets[:, None] * dbeta_stride_e
                  + rb_offsets[None, :])
    dbeta_tile = tl.load(
        dbeta_ptrs,
        mask=e_mask[:, None] & rb_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Load Q[bin] and Q[bin+1] for both dt computation and dQ contributions.
    Q0_ptrs = Q_ptr + bin_e[:, None] * Q_stride_l + rb_offsets[None, :]
    Q1_ptrs = Q_ptr + (bin_e + 1)[:, None] * Q_stride_l + rb_offsets[None, :]
    Q0 = tl.load(Q0_ptrs, mask=e_mask[:, None] & rb_mask[None, :], other=0.0).to(tl.float32)
    Q1 = tl.load(Q1_ptrs, mask=e_mask[:, None] & rb_mask[None, :], other=0.0).to(tl.float32)

    # ---- dt: per-token (no cross-program reduction) ------------------------
    # dt[e] = sum_c (Q1[e,c] - Q0[e,c]) * dbeta[e,c]
    diff = Q1 - Q0
    dt_partial = tl.sum(diff * dbeta_tile, axis=1)  # (BLOCK_E,)
    tl.store(dt_ptr + e_offsets, dt_partial, mask=e_mask)

    # ---- dQ: atomic scatter-add as before ----------------------------------
    contrib_lo = one_minus_t[:, None] * dbeta_tile
    contrib_hi = t_e[:, None]         * dbeta_tile

    for l in tl.static_range(L):
        m_lo = (bin_e == l) & e_mask
        m_hi = (bin_e == (l - 1)) & e_mask
        row_contrib = tl.where(m_lo[:, None], contrib_lo, 0.0)
        row_contrib += tl.where(m_hi[:, None], contrib_hi, 0.0)
        acc_row = tl.sum(row_contrib, axis=0)
        out_ptrs = dQ_ptr + l * dQ_stride_l + rb_offsets
        tl.atomic_add(out_ptrs, acc_row, mask=rb_mask)


def b1_backward_dq_dt(
    Q: torch.Tensor,
    bin_idx: torch.Tensor,
    t: torch.Tensor,
    dbeta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused backward: produce both dQ and dt in one Triton launch.

    Returns
    -------
    dQ : (L, R_b) fp32
    dt : matching ``bin_idx`` shape, fp32 (caller can cast to t's dtype).
    """
    L, R_b = Q.shape
    if bin_idx.dim() > 1:
        flat_bin = bin_idx.reshape(-1).contiguous()
        flat_t = t.reshape(-1).contiguous()
        flat_dbeta = dbeta.reshape(-1, R_b)
    else:
        flat_bin = bin_idx.contiguous()
        flat_t = t.contiguous()
        flat_dbeta = dbeta
    if flat_dbeta.stride(-1) != 1:
        flat_dbeta = flat_dbeta.contiguous()

    E = flat_bin.numel()
    assert flat_bin.dtype == torch.int64

    device = Q.device
    dQ = torch.zeros((L, R_b), dtype=torch.float32, device=device)
    dt_flat = torch.empty(E, dtype=torch.float32, device=device)

    # BLOCK_RB == R_b: enforce single-RB-tile so dt has no cross-program reduce.
    # Round R_b up to the next power of 2 for Triton constexpr requirements.
    BLOCK_RB = 1
    while BLOCK_RB < R_b:
        BLOCK_RB *= 2

    grid = lambda meta: (triton.cdiv(E, meta["BLOCK_E"]),)  # noqa: E731
    _b1_backward_dq_dt_kernel[grid](
        Q,
        flat_bin,
        flat_t,
        flat_dbeta,
        dQ,
        dt_flat,
        E=E,
        R_b=R_b,
        L=L,
        Q_stride_l=Q.stride(0),
        dbeta_stride_e=flat_dbeta.stride(0),
        dQ_stride_l=dQ.stride(0),
        BLOCK_RB=BLOCK_RB,
    )
    return dQ, dt_flat.view(bin_idx.shape)


__all__ = ["b1_backward_dq", "b1_backward_dq_dt", "b1_forward"]
