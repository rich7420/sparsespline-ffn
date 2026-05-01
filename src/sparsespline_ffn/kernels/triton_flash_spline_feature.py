"""FlashSplineFeature forward Triton kernel — RL-Spline-KV (v7 Phase B2.1).

Spec: docs/THEORY_v7_RL_SPLINE_KV.md §R.4.1, §R.4.2.

Inputs:
    z         : [N, h]  (post-key activation)
    C         : [h, L, r]  (local code table; L = G + 2 for B2)

Output of the *delta kernel* (this file):
    delta     : [N, r]
        delta[n, c] = sum_j sum_{b in active(z[n,j])} B_b(z[n,j]) * C[j, b, c]

The wrapper ``flash_spline_feature_forward`` produces the full feature
vector::

    f = [phi(z); lambda * delta]   shape [N, h+r]

via two cheap launches: one Triton kernel for delta + one PyTorch
elementwise for phi(z).  Per v7 §R.4.4 a 3-launch end-to-end FFN is
the eventual goal but 4-launch is acceptable for Phase 2 microbench
and easier to validate.

Per v7 §R.3.3.5: kernel accumulates in fp32 internally, casts back
to bf16/fp16 only at output.

This file contains the **forward** kernel.  The backward kernel
(``flash_spline_feature_backward``) is task B2.4; until then the
autograd path uses reference recomputation.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _autotune_fwd_configs():
    """Autotune configs for the forward kernel.

    Search space chosen for typical RL-Spline-KV shapes:
      h ∈ {512, 768, 1024}
      r ∈ {32, 64}
      L ∈ {16, 22}
      N (= B*T) ∈ {256, 512, 2048, 4096}

    BLOCK_R = r is forced (single program along r, no redundant
    bin/tau recomputation across r tiles).  We sweep BLOCK_N and
    BLOCK_J jointly because the inner-loop pipelining trade-off is
    BLOCK_J vs register pressure from BLOCK_N * BLOCK_R accumulator.
    """
    cfgs = []
    # Targeted set (was 180; now ~16) — focused on configs that did well
    # in earlier microbench: BN=64 with deeper BJ unrolling.
    for BN in [32, 64]:
        for BJ in [4, 8, 16]:
            for nw in [4, 8]:
                cfgs.append(triton.Config(
                    {"BLOCK_N": BN, "BLOCK_J": BJ},
                    num_warps=nw, num_stages=2,
                ))
    # A few deep-pipeline alternatives
    for BJ in [8, 16]:
        cfgs.append(triton.Config(
            {"BLOCK_N": 64, "BLOCK_J": BJ},
            num_warps=4, num_stages=3,
        ))
    return cfgs


@triton.autotune(configs=_autotune_fwd_configs(), key=["N", "h", "r", "L"])
@triton.jit
def _flash_spline_feature_delta_fwd_v2(
    z_ptr,                    # [N, h]
    C_ptr,                    # [h, L, r]
    delta_ptr,                # [N, r]
    grid_lo, scale,
    G_max,
    N: tl.constexpr,
    h: tl.constexpr,
    r: tl.constexpr,
    L: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """v2: BLOCK_J>1 unrolls the inner key loop for ILP; autotuned configs.

    Layout: (pid_n, pid_r) tiles output.  For r ≤ BLOCK_R the grid is
    1D in r-direction (single program per r-tile) so no redundant
    bin/tau recomputation.

    Inner loop processes BLOCK_J keys at a time with @tl.static_range
    over the basis index k ∈ {0,1,2}.  This makes the compiler unroll
    everything and pipeline loads across the 3·BLOCK_J access stream.
    """
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    n_mask = n_offs < N
    r_mask = r_offs < r

    acc = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)

    for j_start in range(0, h, BLOCK_J):
        # Inner static-range so compiler unrolls all BLOCK_J*3 loads
        for jj in tl.static_range(BLOCK_J):
            j = j_start + jj
            j_in = j < h

            z_n = tl.load(
                z_ptr + n_offs * h + j,
                mask=n_mask & j_in, other=0.0,
            ).to(tl.float32)

            u = (z_n - grid_lo) * scale
            in_range = (u >= 0.0) & (u <= G_max)
            u_clip = tl.minimum(tl.maximum(u, 0.0), G_max - 1.0)
            bin_idx = u_clip.to(tl.int32)
            tau = u_clip - bin_idx.to(tl.float32)
            omt = 1.0 - tau
            B0 = 0.5 * omt * omt
            B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
            B2 = 0.5 * tau * tau
            ir_f = in_range.to(tl.float32)
            B0 = B0 * ir_f; B1 = B1 * ir_f; B2 = B2 * ir_f

            base_j = C_ptr + j * (L * r)
            mask_2d = n_mask[:, None] & r_mask[None, :] & j_in

            # k=0
            c_addr = base_j + bin_idx[:, None] * r + r_offs[None, :]
            c_load = tl.load(c_addr, mask=mask_2d, other=0.0).to(tl.float32)
            acc += B0[:, None] * c_load
            # k=1
            c_addr = base_j + (bin_idx + 1)[:, None] * r + r_offs[None, :]
            c_load = tl.load(c_addr, mask=mask_2d, other=0.0).to(tl.float32)
            acc += B1[:, None] * c_load
            # k=2
            c_addr = base_j + (bin_idx + 2)[:, None] * r + r_offs[None, :]
            c_load = tl.load(c_addr, mask=mask_2d, other=0.0).to(tl.float32)
            acc += B2[:, None] * c_load

    out_ptr = delta_ptr + n_offs[:, None] * r + r_offs[None, :]
    tl.store(out_ptr, acc, mask=n_mask[:, None] & r_mask[None, :])


@triton.autotune(configs=_autotune_fwd_configs(), key=["N", "h", "r", "L"])
@triton.jit
def _flash_spline_feature_delta_fwd_v3(
    z_ptr, C_ptr, delta_ptr,
    grid_lo, scale, G_max,
    N: tl.constexpr, h: tl.constexpr, r: tl.constexpr, L: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """v3: single 3-bin contiguous gather per (n, j) — cuts load instruction
    count by ~3× compared to v1/v2.

    For each (n, j) we now do ONE tl.load of shape [3, BLOCK_R] covering
    bins {bin_n,j, bin_n,j+1, bin_n,j+2} contiguously, instead of three
    separate loads.  The hardware can coalesce these addresses since
    they are consecutive in memory.
    """
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    n_mask = n_offs < N
    r_mask = r_offs < r

    acc = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)
    k_offs = tl.arange(0, 4)  # we use [:3]; pad to power of 2 for triton

    for j_start in range(0, h, BLOCK_J):
        for jj in tl.static_range(BLOCK_J):
            j = j_start + jj
            j_in = j < h
            z_n = tl.load(z_ptr + n_offs * h + j,
                          mask=n_mask & j_in, other=0.0).to(tl.float32)

            u = (z_n - grid_lo) * scale
            in_range = (u >= 0.0) & (u <= G_max)
            u_clip = tl.minimum(tl.maximum(u, 0.0), G_max - 1.0)
            bin_idx = u_clip.to(tl.int32)
            tau = u_clip - bin_idx.to(tl.float32)
            omt = 1.0 - tau
            B0 = 0.5 * omt * omt
            B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
            B2 = 0.5 * tau * tau
            ir_f = in_range.to(tl.float32)
            B0 = B0 * ir_f; B1 = B1 * ir_f; B2 = B2 * ir_f

            # Single 3-bin load per token: shape [BLOCK_N, 4, BLOCK_R],
            # but we only use the first 3 in the k dim.
            base_j = C_ptr + j * (L * r)
            # offsets[n, k, c] = (bin_idx[n] + k) * r + r_offs[c]
            # k_offs has shape [4]; we mask k>2.
            row_idx = bin_idx[:, None, None] + k_offs[None, :, None]   # [BLOCK_N, 4, 1]
            addr = base_j + row_idx * r + r_offs[None, None, :]        # [BLOCK_N, 4, BLOCK_R]
            k_mask = k_offs < 3                                          # [4]
            mask3d = (n_mask[:, None, None] & r_mask[None, None, :]
                      & k_mask[None, :, None] & j_in)
            c_3 = tl.load(addr, mask=mask3d, other=0.0).to(tl.float32)  # [BLOCK_N, 4, BLOCK_R]

            # Build B_stack [BLOCK_N, 4] (only first 3 used; 4th is 0)
            B_pad = tl.zeros([BLOCK_N], dtype=tl.float32)
            B_stack = tl.where(k_offs[None, :] == 0, B0[:, None],
                       tl.where(k_offs[None, :] == 1, B1[:, None],
                       tl.where(k_offs[None, :] == 2, B2[:, None], B_pad[:, None])))
            # acc[n, c] += sum_k B_stack[n, k] * c_3[n, k, c]
            acc += tl.sum(B_stack[:, :, None] * c_3, axis=1)

    out_ptr = delta_ptr + n_offs[:, None] * r + r_offs[None, :]
    tl.store(out_ptr, acc, mask=n_mask[:, None] & r_mask[None, :])


def flash_spline_delta_forward_v3(
    z: torch.Tensor, C: torch.Tensor,
    grid_lo: float, grid_hi: float, G: int,
) -> torch.Tensor:
    """v3: single 3-bin contiguous gather per (n, j)."""
    if not z.is_cuda or not C.is_cuda:
        raise RuntimeError("flash_spline_delta_forward_v3 needs CUDA tensors")
    N, h = z.shape
    h_C, L, r = C.shape
    if h != h_C or L != G + 2:
        raise ValueError("shape mismatch")
    z_c = z.contiguous(); C_c = C.contiguous()
    delta = torch.empty((N, r), device=z.device, dtype=torch.float32)
    BLOCK_R = max(16, triton.next_power_of_2(r))
    scale = G / (grid_hi - grid_lo)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),
                         triton.cdiv(r, BLOCK_R))
    _flash_spline_feature_delta_fwd_v3[grid](
        z_c, C_c, delta,
        float(grid_lo), float(scale), float(G),
        N, h, r, L,
        BLOCK_R=BLOCK_R,
    )
    return delta


def _autotune_fwd_v4_configs():
    """v4 autotune: ~24 targeted configs (down from 180 for fast compile).

    BLOCK_H drives SM utilization (cdiv(h, BLOCK_H) programs along h);
    BLOCK_J drives inner-loop unrolling within each chunk.

    Spans 3080 (68 SMs) and H100 (132 SMs) — for h=768 the grid sizes
    are 8 × cdiv(768, BLOCK_H) = 8 × {24, 12, 8} = {192, 96, 64} programs.
    """
    cfgs = []
    for BN, BH in [(64, 32), (64, 64), (64, 96),
                   (32, 64), (32, 96)]:
        for BJ in [4, 8]:
            for nw in [4, 8]:
                cfgs.append(triton.Config(
                    {"BLOCK_N": BN, "BLOCK_H": BH, "BLOCK_J": BJ},
                    num_warps=nw, num_stages=2,
                ))
    # Deep-pipeline alternatives
    for BN, BH, BJ in [(64, 64, 8), (64, 96, 8)]:
        for ns in [3, 4]:
            cfgs.append(triton.Config(
                {"BLOCK_N": BN, "BLOCK_H": BH, "BLOCK_J": BJ},
                num_warps=4, num_stages=ns,
            ))
    return cfgs


@triton.autotune(configs=_autotune_fwd_v4_configs(), key=["N", "h", "r", "L"],
                  reset_to_zero=["delta_ptr"])
@triton.jit
def _flash_spline_feature_delta_fwd_v4(
    z_ptr, C_ptr, delta_ptr,
    grid_lo, scale, G_max,
    N: tl.constexpr, h: tl.constexpr, r: tl.constexpr, L: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """v4: 3-D grid (pid_n, pid_r, pid_h) with atomic_add reduction over h.

    With h split across programs, SM utilization goes from ~12% to ~94%
    on 3080.  Each program produces a partial delta for its h chunk,
    then ``tl.atomic_add`` reduces all chunks into the output buffer.

    The trade-off is atomic-add overhead vs. SM parallelism — generally
    a clear win when total programs (8) was much less than SMs (68).
    """
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_h = tl.program_id(2)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    h_start = pid_h * BLOCK_H
    n_mask = n_offs < N
    r_mask = r_offs < r

    acc = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)

    # Iterate the local h chunk in BLOCK_J-sized inner unroll.
    for j_off in range(0, BLOCK_H, BLOCK_J):
        for jj in tl.static_range(BLOCK_J):
            j = h_start + j_off + jj
            j_in = j < h

            z_n = tl.load(z_ptr + n_offs * h + j,
                          mask=n_mask & j_in, other=0.0).to(tl.float32)
            u = (z_n - grid_lo) * scale
            in_range = (u >= 0.0) & (u <= G_max)
            u_clip = tl.minimum(tl.maximum(u, 0.0), G_max - 1.0)
            bin_idx = u_clip.to(tl.int32)
            tau = u_clip - bin_idx.to(tl.float32)
            omt = 1.0 - tau
            B0 = 0.5 * omt * omt
            B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
            B2 = 0.5 * tau * tau
            ir_f = in_range.to(tl.float32)
            B0 = B0 * ir_f; B1 = B1 * ir_f; B2 = B2 * ir_f

            base_j = C_ptr + j * (L * r)
            mask_2d = n_mask[:, None] & r_mask[None, :] & j_in

            c_addr = base_j + bin_idx[:, None] * r + r_offs[None, :]
            c0 = tl.load(c_addr, mask=mask_2d, other=0.0).to(tl.float32)
            acc += B0[:, None] * c0

            c_addr = base_j + (bin_idx + 1)[:, None] * r + r_offs[None, :]
            c1 = tl.load(c_addr, mask=mask_2d, other=0.0).to(tl.float32)
            acc += B1[:, None] * c1

            c_addr = base_j + (bin_idx + 2)[:, None] * r + r_offs[None, :]
            c2 = tl.load(c_addr, mask=mask_2d, other=0.0).to(tl.float32)
            acc += B2[:, None] * c2

    # Atomic reduce across h-chunks
    out_addr = delta_ptr + n_offs[:, None] * r + r_offs[None, :]
    tl.atomic_add(out_addr, acc, mask=n_mask[:, None] & r_mask[None, :])


def flash_spline_delta_forward_v4(
    z: torch.Tensor, C: torch.Tensor,
    grid_lo: float, grid_hi: float, G: int,
) -> torch.Tensor:
    """v4: h-split with atomic_add — addresses SM-utilization bottleneck."""
    if not z.is_cuda or not C.is_cuda:
        raise RuntimeError("flash_spline_delta_forward_v4 needs CUDA tensors")
    N, h = z.shape
    h_C, L, r = C.shape
    if h != h_C or L != G + 2:
        raise ValueError("shape mismatch")
    z_c = z.contiguous(); C_c = C.contiguous()
    # MUST be zero-initialized for atomic_add to be correct.
    delta = torch.zeros((N, r), device=z.device, dtype=torch.float32)
    BLOCK_R = max(16, triton.next_power_of_2(r))
    scale = G / (grid_hi - grid_lo)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),
                         triton.cdiv(r, BLOCK_R),
                         triton.cdiv(h, meta["BLOCK_H"]))
    _flash_spline_feature_delta_fwd_v4[grid](
        z_c, C_c, delta,
        float(grid_lo), float(scale), float(G),
        N, h, r, L,
        BLOCK_R=BLOCK_R,
    )
    return delta


@triton.autotune(configs=_autotune_fwd_v4_configs(), key=["N", "h", "r", "L"])
@triton.jit
def _flash_spline_feature_delta_fwd_v5(
    z_ptr, C_ptr, delta_partial_ptr,
    grid_lo, scale, G_max,
    N: tl.constexpr, h: tl.constexpr, r: tl.constexpr, L: tl.constexpr,
    H_CHUNKS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """v5: like v4 but writes to a per-chunk partial buffer (no atomics).

    delta_partial has shape [H_CHUNKS, N, r].  Each program writes
    EXACTLY ONCE to its own chunk slice via plain tl.store.  A small
    PyTorch sum() reduces the chunks.

    Eliminates atomic-add contention but pays for one extra reduce
    kernel launch (~5 us) and 4× temp memory.
    """
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_h = tl.program_id(2)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    h_start = pid_h * BLOCK_H
    n_mask = n_offs < N
    r_mask = r_offs < r

    acc = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)

    for j_off in range(0, BLOCK_H, BLOCK_J):
        for jj in tl.static_range(BLOCK_J):
            j = h_start + j_off + jj
            j_in = j < h

            z_n = tl.load(z_ptr + n_offs * h + j,
                          mask=n_mask & j_in, other=0.0).to(tl.float32)
            u = (z_n - grid_lo) * scale
            in_range = (u >= 0.0) & (u <= G_max)
            u_clip = tl.minimum(tl.maximum(u, 0.0), G_max - 1.0)
            bin_idx = u_clip.to(tl.int32)
            tau = u_clip - bin_idx.to(tl.float32)
            omt = 1.0 - tau
            B0 = 0.5 * omt * omt
            B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
            B2 = 0.5 * tau * tau
            ir_f = in_range.to(tl.float32)
            B0 = B0 * ir_f; B1 = B1 * ir_f; B2 = B2 * ir_f

            base_j = C_ptr + j * (L * r)
            mask_2d = n_mask[:, None] & r_mask[None, :] & j_in
            c0 = tl.load(base_j + bin_idx[:, None] * r + r_offs[None, :],
                          mask=mask_2d, other=0.0).to(tl.float32)
            c1 = tl.load(base_j + (bin_idx + 1)[:, None] * r + r_offs[None, :],
                          mask=mask_2d, other=0.0).to(tl.float32)
            c2 = tl.load(base_j + (bin_idx + 2)[:, None] * r + r_offs[None, :],
                          mask=mask_2d, other=0.0).to(tl.float32)
            acc += B0[:, None] * c0 + B1[:, None] * c1 + B2[:, None] * c2

    # Plain store to my own chunk slice — no atomic
    out_ptr = (delta_partial_ptr + pid_h * (N * r)
               + n_offs[:, None] * r + r_offs[None, :])
    tl.store(out_ptr, acc, mask=n_mask[:, None] & r_mask[None, :])


def flash_spline_delta_forward_v5(
    z: torch.Tensor, C: torch.Tensor,
    grid_lo: float, grid_hi: float, G: int,
) -> torch.Tensor:
    """v5: parallel partial buffer + sum-reduce (no atomic_add contention)."""
    if not z.is_cuda or not C.is_cuda:
        raise RuntimeError("v5 needs CUDA tensors")
    N, h = z.shape
    h_C, L, r = C.shape
    if h != h_C or L != G + 2:
        raise ValueError("shape mismatch")
    z_c = z.contiguous(); C_c = C.contiguous()
    BLOCK_R = max(16, triton.next_power_of_2(r))
    scale = G / (grid_hi - grid_lo)

    # Two-stage launch: first determine BLOCK_H from autotune, then size partial
    # The trick: we can't know autotune's BLOCK_H until kernel run.  Use
    # a max-bound buffer (assume worst-case smallest BLOCK_H = 32).
    H_CHUNKS_MAX = triton.cdiv(h, 32)
    delta_partial = torch.zeros((H_CHUNKS_MAX, N, r),
                                  device=z.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),
                         triton.cdiv(r, BLOCK_R),
                         triton.cdiv(h, meta["BLOCK_H"]))
    _flash_spline_feature_delta_fwd_v5[grid](
        z_c, C_c, delta_partial,
        float(grid_lo), float(scale), float(G),
        N, h, r, L,
        H_CHUNKS=H_CHUNKS_MAX,
        BLOCK_R=BLOCK_R,
    )
    # Sum across the actual h-chunks (some slots may stay 0 if BLOCK_H > 32)
    return delta_partial.sum(dim=0)


@triton.jit
def _flash_spline_feature_delta_fwd(
    z_ptr,                    # [N, h]
    C_ptr,                    # [h, L, r]
    delta_ptr,                # [N, r]
    grid_lo, scale,           # scale = G / (grid_hi - grid_lo)
    G_max,                    # = float(G); used to mask out-of-range
    N: tl.constexpr,
    h: tl.constexpr,
    r: tl.constexpr,
    L: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    n_mask = n_offs < N
    r_mask = r_offs < r

    # fp32 accumulator (v7 §R.3.3.5)
    acc = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)

    # Iterate over keys j in chunks of BLOCK_J, inner loop unrolled over
    # the 3 active basis indices for B2.
    for j_start in range(0, h, BLOCK_J):
        for jj in tl.static_range(BLOCK_J):
            j = j_start + jj
            j_in = j < h

            # Load z[BLOCK_N, j] for this single key
            z_n = tl.load(
                z_ptr + n_offs * h + j,
                mask=n_mask & j_in,
                other=0.0,
            ).to(tl.float32)

            # Bin / fractional position
            u = (z_n - grid_lo) * scale            # [BLOCK_N], fp32
            in_range = (u >= 0.0) & (u <= G_max)
            u_clip = tl.minimum(tl.maximum(u, 0.0), G_max - 1.0)
            bin_idx = u_clip.to(tl.int32)
            tau = u_clip - bin_idx.to(tl.float32)
            omt = 1.0 - tau
            B0 = 0.5 * omt * omt
            B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
            B2 = 0.5 * tau * tau
            ir_f = in_range.to(tl.float32)
            B0 = B0 * ir_f
            B1 = B1 * ir_f
            B2 = B2 * ir_f

            # For each active bin k in {0, 1, 2}, gather C[j, bin+k, r_offs]
            # and accumulate B_k * C into acc.  Pointer addresses depend on
            # per-token bin_idx.
            base_j = C_ptr + j * (L * r)

            # k = 0
            c_addr = base_j + bin_idx[:, None] * r + r_offs[None, :]
            c_load = tl.load(
                c_addr,
                mask=n_mask[:, None] & r_mask[None, :] & j_in,
                other=0.0,
            ).to(tl.float32)
            acc += B0[:, None] * c_load

            # k = 1
            c_addr = base_j + (bin_idx + 1)[:, None] * r + r_offs[None, :]
            c_load = tl.load(
                c_addr,
                mask=n_mask[:, None] & r_mask[None, :] & j_in,
                other=0.0,
            ).to(tl.float32)
            acc += B1[:, None] * c_load

            # k = 2
            c_addr = base_j + (bin_idx + 2)[:, None] * r + r_offs[None, :]
            c_load = tl.load(
                c_addr,
                mask=n_mask[:, None] & r_mask[None, :] & j_in,
                other=0.0,
            ).to(tl.float32)
            acc += B2[:, None] * c_load

    # Cast back to delta dtype and store
    out_ptr = delta_ptr + n_offs[:, None] * r + r_offs[None, :]
    tl.store(out_ptr, acc, mask=n_mask[:, None] & r_mask[None, :])


def flash_spline_delta_forward_v2(
    z: torch.Tensor,             # [N, h]
    C: torch.Tensor,             # [h, L, r]
    grid_lo: float,
    grid_hi: float,
    G: int,
) -> torch.Tensor:                # [N, r] fp32
    """v2: autotuned forward with BLOCK_J unrolling.

    Forces a single-program-along-r layout (BLOCK_R = r) so we never pay
    redundant bin/tau recomputation across r-tiles.
    """
    if not z.is_cuda or not C.is_cuda:
        raise RuntimeError("flash_spline_delta_forward_v2 needs CUDA tensors")
    N, h = z.shape
    h_C, L, r = C.shape
    if h != h_C:
        raise ValueError(f"z h={h} != C h={h_C}")
    if L != G + 2:
        raise ValueError(f"L={L} should equal G+2={G + 2} for B2")

    z_c = z.contiguous(); C_c = C.contiguous()
    delta = torch.empty((N, r), device=z.device, dtype=torch.float32)

    BLOCK_R = max(16, triton.next_power_of_2(r))  # one program along r
    scale = G / (grid_hi - grid_lo)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),
                         triton.cdiv(r, BLOCK_R))
    _flash_spline_feature_delta_fwd_v2[grid](
        z_c, C_c, delta,
        float(grid_lo), float(scale), float(G),
        N, h, r, L,
        BLOCK_R=BLOCK_R,
    )
    return delta


def flash_spline_delta_forward(
    z: torch.Tensor,             # [N, h]
    C: torch.Tensor,             # [h, L, r]
    grid_lo: float,
    grid_hi: float,
    G: int,
) -> torch.Tensor:                # [N, r]
    """Triton forward for the delta = sum_{j, b} B_b(z[n,j]) * C[j, b, :].

    z and C must be on CUDA, contiguous, dtype float16/bfloat16/float32.
    Output is float32 (caller may cast).
    """
    if not z.is_cuda or not C.is_cuda:
        raise RuntimeError("flash_spline_delta_forward needs CUDA tensors")
    if z.dim() != 2:
        raise ValueError(f"z must be [N, h], got {tuple(z.shape)}")
    if C.dim() != 3:
        raise ValueError(f"C must be [h, L, r], got {tuple(C.shape)}")

    N, h = z.shape
    h_C, L, r = C.shape
    if h != h_C:
        raise ValueError(f"z h={h} != C h={h_C}")
    if L != G + 2:
        raise ValueError(f"L={L} should equal G+2={G + 2} for B2")

    z_c = z.contiguous()
    C_c = C.contiguous()

    delta = torch.empty((N, r), device=z.device, dtype=torch.float32)

    BLOCK_N = 32 if N >= 32 else max(1, triton.next_power_of_2(N))
    BLOCK_R = min(64, max(16, triton.next_power_of_2(r)))
    BLOCK_J = 1  # one key per inner step (correctness-first; tunable later)

    scale = G / (grid_hi - grid_lo)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(r, BLOCK_R))
    _flash_spline_feature_delta_fwd[grid](
        z_c, C_c, delta,
        float(grid_lo), float(scale), float(G),
        N, h, r, L,
        BLOCK_N=BLOCK_N, BLOCK_R=BLOCK_R, BLOCK_J=BLOCK_J,
    )
    return delta


def flash_spline_feature_forward(
    z: torch.Tensor,             # [N, h]
    C: torch.Tensor,             # [h, L, r]
    grid_lo: float,
    grid_hi: float,
    G: int,
    activation: str = "relu_sq",
    lambda_scale: float = 1.0,
    version: str = "v2",
) -> torch.Tensor:                # [N, h+r]
    """Compute f = [phi(z); lambda * delta(z, C)] using the Triton kernel
    for the delta half and PyTorch elementwise for the phi half.

    Returns f as a single [N, h+r] tensor in the same dtype as z.
    """
    if z.dim() != 2:
        raise ValueError(f"z must be [N, h], got {tuple(z.shape)}")
    if activation == "relu_sq":
        a = torch.where(z > 0, z * z, torch.zeros_like(z))
    elif activation == "gelu":
        a = torch.nn.functional.gelu(z)
    elif activation == "identity":
        a = z
    else:
        raise ValueError(f"unknown activation {activation}")

    if version == "v5":
        delta_fn = flash_spline_delta_forward_v5
    elif version == "v4":
        delta_fn = flash_spline_delta_forward_v4
    elif version == "v3":
        delta_fn = flash_spline_delta_forward_v3
    elif version == "v2":
        delta_fn = flash_spline_delta_forward_v2
    elif version == "v1":
        delta_fn = flash_spline_delta_forward
    else:
        raise ValueError(f"unknown version {version}")
    delta = delta_fn(z, C, grid_lo, grid_hi, G).to(z.dtype)
    if lambda_scale != 1.0:
        delta = delta * float(lambda_scale)
    return torch.cat([a, delta], dim=-1)


def _autotune_bwd_v1_configs():
    """Backward autotune (original tight set, post-revert)."""
    cfgs = []
    for BN, BH in [(32, 64), (32, 96), (64, 64), (64, 96)]:
        for BJ in [4, 8]:
            for nw in [4, 8]:
                cfgs.append(triton.Config(
                    {"BLOCK_N": BN, "BLOCK_H": BH, "BLOCK_J": BJ},
                    num_warps=nw, num_stages=2,
                ))
    for BN, BH, BJ in [(32, 64, 8), (64, 64, 8)]:
        cfgs.append(triton.Config(
            {"BLOCK_N": BN, "BLOCK_H": BH, "BLOCK_J": BJ},
            num_warps=4, num_stages=3,
        ))
    return cfgs


@triton.autotune(configs=_autotune_bwd_v1_configs(), key=["N", "h", "r", "L"],
                  reset_to_zero=["dC_ptr", "dz_ptr"])
@triton.jit
def _flash_spline_feature_delta_bwd_v1(
    z_ptr,                     # [N, h] saved
    C_ptr,                     # [h, L, r] saved
    g_delta_ptr,               # [N, r] upstream gradient
    dC_ptr,                    # [h, L, r] fp32 output (zero-initialized)
    dz_ptr,                    # [N, h] fp32 output (zero-initialized OK; we write all positions)
    grid_lo, scale, G_max,
    N: tl.constexpr, h: tl.constexpr, r: tl.constexpr, L: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """Backward kernel — produces dC (atomic-scatter) and dz (plain store).

    Same 3D grid as v4 fwd: (pid_n, pid_r, pid_h).  Crucial design points:

    * dC[h_chunk, :, :] is owned by exactly one pid_h program.  Within that
      program, multiple tokens can hit the same (j, bin+k, c), so we use
      ``tl.atomic_add`` (fp32).  No contention across pid_h.
    * dz[n_chunk, h_chunk] is owned by exactly one (pid_n, pid_h) program
      *when BLOCK_R = r*.  Plain ``tl.store`` — no atomics.
    * Internal accumulators are fp32 (v7 §R.3.3.5).
    * dB derivative mask is ``u in [0, G-1]`` (matches autograd's clamp
      gradient semantics — see v7 §R.3.3 + bwd_ref derivation).

    Compute reuse with forward: bin/τ/B0/B1/B2 are recomputed (saving
    a separate "saved tensor" trip through HBM that would cost more than
    the recompute).
    """
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_h = tl.program_id(2)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    h_start = pid_h * BLOCK_H
    n_mask = n_offs < N
    r_mask = r_offs < r

    # Load g_delta tile [BLOCK_N, BLOCK_R] once — reused for both dC scatter
    # and dz inner products throughout the inner loop.
    g_addr = g_delta_ptr + n_offs[:, None] * r + r_offs[None, :]
    g_mask = n_mask[:, None] & r_mask[None, :]
    g_delta = tl.load(g_addr, mask=g_mask, other=0.0).to(tl.float32)

    for j_off in range(0, BLOCK_H, BLOCK_J):
        for jj in tl.static_range(BLOCK_J):
            j = h_start + j_off + jj
            j_in = j < h
            # Clip j for address arithmetic — when BLOCK_H > h - h_start we
            # iterate OOB positions; mask should suppress writes but we belt-
            # and-braces with j_safe so any leaked write hits a valid (but
            # zero-contribution) cell instead of corrupting memory.
            j_safe = tl.minimum(j, h - 1)

            # ----- Reconstruct fwd quantities for this key -----
            z_n = tl.load(z_ptr + n_offs * h + j_safe,
                          mask=n_mask & j_in, other=0.0).to(tl.float32)
            u = (z_n - grid_lo) * scale
            in_range = (u >= 0.0) & (u <= G_max)
            # Backward derivative mask: ``u ∈ [0, G-1]`` (clamp gradient zone)
            clamp_active = (u >= 0.0) & (u <= G_max - 1.0)
            u_clip = tl.minimum(tl.maximum(u, 0.0), G_max - 1.0)
            bin_idx = u_clip.to(tl.int32)
            tau = u_clip - bin_idx.to(tl.float32)
            omt = 1.0 - tau

            B0 = 0.5 * omt * omt
            B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
            B2 = 0.5 * tau * tau
            dB0 = -omt
            dB1 = 1.0 - 2.0 * tau
            dB2 = tau

            ir_f = in_range.to(tl.float32)
            cl_f = clamp_active.to(tl.float32)
            # j_in_f forces ALL contributions (B and dB) to zero on OOB iters
            j_in_f = tl.where(j_in, 1.0, 0.0)
            B0 = B0 * ir_f * j_in_f
            B1 = B1 * ir_f * j_in_f
            B2 = B2 * ir_f * j_in_f
            dB0 = dB0 * cl_f * j_in_f
            dB1 = dB1 * cl_f * j_in_f
            dB2 = dB2 * cl_f * j_in_f

            base_j    = C_ptr  + j_safe * (L * r)
            base_dC_j = dC_ptr + j_safe * (L * r)
            mask_2d = n_mask[:, None] & r_mask[None, :] & j_in

            # ----- Load C[j, bin+k, :] for k=0,1,2 (same as fwd) -----
            c0 = tl.load(base_j + bin_idx[:, None] * r + r_offs[None, :],
                         mask=mask_2d, other=0.0).to(tl.float32)
            c1 = tl.load(base_j + (bin_idx + 1)[:, None] * r + r_offs[None, :],
                         mask=mask_2d, other=0.0).to(tl.float32)
            c2 = tl.load(base_j + (bin_idx + 2)[:, None] * r + r_offs[None, :],
                         mask=mask_2d, other=0.0).to(tl.float32)

            # ----- dz contribution -----
            # dz[n, j] += scale * (dB0*<C0,g> + dB1*<C1,g> + dB2*<C2,g>)
            # where <Ck, g> = sum_c c_k[n, c] * g_delta[n, c]
            inner0 = tl.sum(c0 * g_delta, axis=1)   # [BLOCK_N]
            inner1 = tl.sum(c1 * g_delta, axis=1)
            inner2 = tl.sum(c2 * g_delta, axis=1)
            dz_val = scale * (dB0 * inner0 + dB1 * inner1 + dB2 * inner2)
            # Plain store — single owner per (n, j) when BLOCK_R = r.
            # (dB already zeroed for OOB j, so dz_val is 0 there; we still
            # mask the store to avoid touching dz[*, j>=h].)
            tl.store(dz_ptr + n_offs * h + j_safe,
                     dz_val, mask=n_mask & j_in)

            # ----- dC contribution (atomic scatter) -----
            # B_k already zeroed on OOB j → v_k = 0, atomic_add adds 0.
            # Mask still applied as primary defense.
            v0 = B0[:, None] * g_delta
            tl.atomic_add(base_dC_j + bin_idx[:, None] * r + r_offs[None, :],
                          v0, mask=mask_2d)
            v1 = B1[:, None] * g_delta
            tl.atomic_add(base_dC_j + (bin_idx + 1)[:, None] * r + r_offs[None, :],
                          v1, mask=mask_2d)
            v2 = B2[:, None] * g_delta
            tl.atomic_add(base_dC_j + (bin_idx + 2)[:, None] * r + r_offs[None, :],
                          v2, mask=mask_2d)


def _autotune_bwd_v2_configs():
    """v2 bwd: BLOCK_N is FIXED (caller controls partial-buffer size).
    Search over (BLOCK_H, BLOCK_J, num_warps, num_stages).
    """
    cfgs = []
    for BH in [64, 96]:
        for BJ in [4, 8]:
            for nw in [4, 8]:
                cfgs.append(triton.Config(
                    {"BLOCK_H": BH, "BLOCK_J": BJ},
                    num_warps=nw, num_stages=2,
                ))
    for BH, BJ in [(64, 8), (96, 8)]:
        cfgs.append(triton.Config(
            {"BLOCK_H": BH, "BLOCK_J": BJ},
            num_warps=4, num_stages=3,
        ))
    return cfgs


@triton.autotune(configs=_autotune_bwd_v2_configs(),
                  key=["N", "h", "r", "L", "BLOCK_N"],
                  reset_to_zero=["dC_partial_ptr", "dz_ptr"])
@triton.jit
def _flash_spline_feature_delta_bwd_v2(
    z_ptr,                     # [N, h]
    C_ptr,                     # [h, L, r]
    g_delta_ptr,               # [N, r]
    dC_partial_ptr,            # [N_CHUNKS, h, L, r] fp32 (zero-init)
    dz_ptr,                    # [N, h] fp32 (zero-init)
    grid_lo, scale, G_max,
    N: tl.constexpr, h: tl.constexpr, r: tl.constexpr, L: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """v2 backward kernel: writes dC contributions to a per-pid_n slice
    of dC_partial[N_CHUNKS, h, L, r] via plain ``tl.store``, eliminating
    atomic_add contention.

    A second (PyTorch) pass reduces dC_partial.sum(dim=0) → dC.

    Why this is faster than v1:
        v1: every dC[j, b, c] cell has ~32 contributing programs hammering
            it via atomic_add → serialization at hot cells.
        v2: each program writes to its own dC_partial[pid_n, j, b, c]
            slice with plain stores — no atomics, no contention.

    VRAM cost: dC_partial is N_CHUNKS × h × L × r × 4 bytes.  For
    BLOCK_N=128 (typical), N=2048, h=768, L=16, r=32: ~24 MB extra.

    This works because:
        * pid_n owns its token chunk → unique partial slice (plain store)
        * pid_h owns its h chunk → no cross-program contention on h axis
        * BLOCK_R = r → no pid_r contention
        * Within a single program, tokens may share (j, bin+k, c) cells,
          but they accumulate into the SAME partial slice → fp32 add ops
          inside the kernel compose without atomics (just within-warp
          reductions if BLOCK_N tokens collide).

    Wait — within-program collision on (j, bin+k) IS a problem.  For
    BLOCK_N tokens hitting same bin, we need to accumulate. We do this
    via atomic_add WITHIN the program (cheap intra-CTA atomics) — much
    less contention than v1's cross-program atomics.

    For now we keep atomic_add but only across BLOCK_N tokens within
    one program (writes to dC_partial[pid_n, ...]).  No cross-program
    contention.
    """
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_h = tl.program_id(2)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    h_start = pid_h * BLOCK_H
    n_mask = n_offs < N
    r_mask = r_offs < r

    # Base pointer to dC_partial[pid_n, :, :, :]  (shape [h, L, r] for this slot)
    dC_partial_base = dC_partial_ptr + pid_n * (h * L * r)

    # Load g_delta tile [BLOCK_N, BLOCK_R] once
    g_addr = g_delta_ptr + n_offs[:, None] * r + r_offs[None, :]
    g_mask = n_mask[:, None] & r_mask[None, :]
    g_delta = tl.load(g_addr, mask=g_mask, other=0.0).to(tl.float32)

    for j_off in range(0, BLOCK_H, BLOCK_J):
        for jj in tl.static_range(BLOCK_J):
            j = h_start + j_off + jj
            j_in = j < h
            j_safe = tl.minimum(j, h - 1)

            z_n = tl.load(z_ptr + n_offs * h + j_safe,
                          mask=n_mask & j_in, other=0.0).to(tl.float32)
            u = (z_n - grid_lo) * scale
            in_range = (u >= 0.0) & (u <= G_max)
            clamp_active = (u >= 0.0) & (u <= G_max - 1.0)
            u_clip = tl.minimum(tl.maximum(u, 0.0), G_max - 1.0)
            bin_idx = u_clip.to(tl.int32)
            tau = u_clip - bin_idx.to(tl.float32)
            omt = 1.0 - tau

            B0 = 0.5 * omt * omt
            B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
            B2 = 0.5 * tau * tau
            dB0 = -omt
            dB1 = 1.0 - 2.0 * tau
            dB2 = tau

            ir_f = in_range.to(tl.float32)
            cl_f = clamp_active.to(tl.float32)
            j_in_f = tl.where(j_in, 1.0, 0.0)
            B0 = B0 * ir_f * j_in_f
            B1 = B1 * ir_f * j_in_f
            B2 = B2 * ir_f * j_in_f
            dB0 = dB0 * cl_f * j_in_f
            dB1 = dB1 * cl_f * j_in_f
            dB2 = dB2 * cl_f * j_in_f

            base_j        = C_ptr            + j_safe * (L * r)
            base_dC_pj    = dC_partial_base  + j_safe * (L * r)
            mask_2d = n_mask[:, None] & r_mask[None, :] & j_in

            c0 = tl.load(base_j + bin_idx[:, None] * r + r_offs[None, :],
                         mask=mask_2d, other=0.0).to(tl.float32)
            c1 = tl.load(base_j + (bin_idx + 1)[:, None] * r + r_offs[None, :],
                         mask=mask_2d, other=0.0).to(tl.float32)
            c2 = tl.load(base_j + (bin_idx + 2)[:, None] * r + r_offs[None, :],
                         mask=mask_2d, other=0.0).to(tl.float32)

            # ----- dz contribution (plain store, single owner) -----
            inner0 = tl.sum(c0 * g_delta, axis=1)
            inner1 = tl.sum(c1 * g_delta, axis=1)
            inner2 = tl.sum(c2 * g_delta, axis=1)
            dz_val = scale * (dB0 * inner0 + dB1 * inner1 + dB2 * inner2)
            tl.store(dz_ptr + n_offs * h + j_safe,
                     dz_val, mask=n_mask & j_in)

            # ----- dC contribution to PARTIAL slice -----
            # Within a program, multiple BLOCK_N tokens can collide on
            # (bin+k, c).  We still need atomic_add inside the program,
            # but contention is now BLOCK_N-bounded (≤ BLOCK_N collisions
            # per cell, vs N total in v1).  Across programs there is NO
            # contention because each pid_n owns a unique partial slice.
            v0 = B0[:, None] * g_delta
            tl.atomic_add(base_dC_pj + bin_idx[:, None] * r + r_offs[None, :],
                          v0, mask=mask_2d)
            v1 = B1[:, None] * g_delta
            tl.atomic_add(base_dC_pj + (bin_idx + 1)[:, None] * r + r_offs[None, :],
                          v1, mask=mask_2d)
            v2 = B2[:, None] * g_delta
            tl.atomic_add(base_dC_pj + (bin_idx + 2)[:, None] * r + r_offs[None, :],
                          v2, mask=mask_2d)


def flash_spline_delta_backward_v2(
    z: torch.Tensor, C: torch.Tensor, g_delta: torch.Tensor,
    grid_lo: float, grid_hi: float, G: int,
    BLOCK_N: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v2 backward: per-pid_n partial buffer + tree reduce.

    Goal: eliminate cross-program atomic_add contention on dC.
    Trade-off: extra VRAM = N_CHUNKS × h × L × r × 4 bytes.
    For default BLOCK_N=128 at N=2048, h=768, L=16-22, r=32:
        N_CHUNKS = 16, partial buffer ~17-24 MB.
    """
    if not z.is_cuda or not C.is_cuda or not g_delta.is_cuda:
        raise RuntimeError("v2 needs CUDA tensors")
    N, h = z.shape
    h_C, L, r = C.shape
    if h != h_C or L != G + 2:
        raise ValueError("shape mismatch")
    if g_delta.shape != (N, r):
        raise ValueError(f"g_delta shape mismatch")

    z_c = z.contiguous(); C_c = C.contiguous(); g_c = g_delta.contiguous()

    N_CHUNKS = (N + BLOCK_N - 1) // BLOCK_N
    dC_partial = torch.zeros((N_CHUNKS, h, L, r),
                              device=z.device, dtype=torch.float32)
    dz = torch.zeros((N, h), device=z.device, dtype=torch.float32)

    BLOCK_R = max(16, triton.next_power_of_2(r))
    scale = G / (grid_hi - grid_lo)
    grid = lambda meta: (N_CHUNKS, triton.cdiv(r, BLOCK_R),
                         triton.cdiv(h, meta["BLOCK_H"]))
    _flash_spline_feature_delta_bwd_v2[grid](
        z_c, C_c, g_c, dC_partial, dz,
        float(grid_lo), float(scale), float(G),
        N, h, r, L,
        BLOCK_N=BLOCK_N, BLOCK_R=BLOCK_R,
    )
    # Reduce — this is one PyTorch sum, very fast on contiguous fp32.
    dC = dC_partial.sum(dim=0)
    return dC, dz


def _autotune_bwd_v3_configs():
    """v3 backward autotune.  Wider search than before: include smaller
    BLOCK_H (less static_range unroll → less register pressure → maybe
    higher SM occupancy), and BLOCK_N=128 (fewer programs along n →
    less atomic contention on dC).

    Trade-offs autotune resolves per shape:
        BLOCK_H smaller → less register pressure, more programs along h
        BLOCK_N larger  → less atomic contention, but bigger per-program tile
        num_warps higher → more parallel warps, less occupancy per CTA
        num_stages higher → deeper async pipeline (Hopper TMA helps)
    """
    cfgs = []
    for BN in [32, 64, 128, 256]:
        for BH in [4, 8, 16, 32]:
            for nw in [4, 8]:
                cfgs.append(triton.Config(
                    {"BLOCK_N": BN, "BLOCK_H": BH},
                    num_warps=nw, num_stages=2,
                ))
    # Deep-pipeline candidates (Hopper async copy benefits from num_stages>=3)
    for BN, BH in [(32, 8), (32, 16), (64, 8), (64, 16), (128, 8),
                   (128, 16), (256, 4), (256, 8)]:
        for ns in [3, 4, 5]:
            for nw in [4, 8]:
                cfgs.append(triton.Config(
                    {"BLOCK_N": BN, "BLOCK_H": BH},
                    num_warps=nw, num_stages=ns,
                ))
    return cfgs


@triton.autotune(configs=_autotune_bwd_v3_configs(),
                  key=["N", "h", "r", "L", "USE_BF16_DOT"],
                  reset_to_zero=["dC_ptr", "dz_ptr"])
@triton.jit
def _flash_spline_feature_delta_bwd_v3(
    z_ptr, C_ptr, g_delta_ptr,
    dC_ptr, dz_ptr,
    grid_lo, scale, G_max,
    N: tl.constexpr, h: tl.constexpr, r: tl.constexpr, L: tl.constexpr,
    L_PAD: tl.constexpr,        # next pow-2 of L for tl.dot
    USE_BF16_DOT: tl.constexpr, # cast to bf16 for tensor-core dispatch when bf16 input
    BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """v3 backward kernel — uses tl.dot (tensor cores) for dC computation.

    Algorithm
    ---------
    The dC scatter ``dC[j, b, c] += sum_n B_k(τ_n) * g_delta[n, c]``
    (where b is one of {bin_n+0, bin_n+1, bin_n+2}) is rewritten as a
    matrix multiply per j_local:

        W[n, b]    = sum_k B_k(τ_n) * 1{bin_n + k == b}      # [BLOCK_N, L_PAD]
        dC_local[b, c] = sum_n W[n, b] * g_delta[n, c]
                       = (W.T @ g_delta)[b, c]               # [L_PAD, BLOCK_R]

    The matmul uses Triton's tl.dot which dispatches to H100/A100
    tensor cores when shapes are aligned — typically 5-10× faster than
    elementwise atomic scatter.

    Atomic count comparison (per call, h=768, N=2048, L=22, r=32):
        v1 atomic: N × h × 3 × r ≈ 150M atomic_adds
        v3 atomic: cdiv(N, BLOCK_N) × cdiv(h, BLOCK_H) × L × r ≈ 4M atomic_adds
        → 35× reduction → atomic-add stage no longer the bottleneck

    Caveats
    -------
    * dz path still uses irregular per-token gather (no dense reformulation
      because dz[n,j] depends on per-token bin_n,j).  We keep v1's dz
      structure inside this kernel.
    * L_PAD must be next pow-2 of L (Triton tl.dot requires power-of-2 dims).
      Padded rows of W are zero (mask-zeroed), then the corresponding
      dC_local rows are not stored.
    """
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_h = tl.program_id(2)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    h_start = pid_h * BLOCK_H
    n_mask = n_offs < N
    r_mask = r_offs < r

    # Load g_delta tile [BLOCK_N, BLOCK_R] once.  We always keep an fp32
    # copy for the dz inner products (which need precision).  When inputs
    # are bf16 (production path), USE_BF16_DOT=True causes the tl.dot to
    # use bf16 inputs → Hopper/Ampere wmma tensor-core dispatch.  For fp32
    # input tests we keep the matmul in fp32 to preserve correctness.
    g_addr = g_delta_ptr + n_offs[:, None] * r + r_offs[None, :]
    g_mask = n_mask[:, None] & r_mask[None, :]
    g_delta = tl.load(g_addr, mask=g_mask, other=0.0).to(tl.float32)
    # Pre-cast to bf16 ONCE (was redundantly done per-iteration before).
    # On fp32-input tests this object stays unused.
    if USE_BF16_DOT:
        g_delta_bf = g_delta.to(tl.bfloat16)

    b_offs_pad = tl.arange(0, L_PAD)             # [L_PAD]
    # Hoist these constants — they don't depend on j.
    b_in_range = (b_offs_pad < L)                # [L_PAD] bool
    valid_bin_f = b_in_range.to(tl.float32)      # [L_PAD] fp32
    bb = b_offs_pad[None, :]                     # [1, L_PAD]

    # Iterate j_local (one j per static-range step — keeps W in registers)
    for j_local in tl.static_range(BLOCK_H):
        j = h_start + j_local
        j_in = j < h
        j_safe = tl.minimum(j, h - 1)

        # ----- Compute B/dB/bin/τ for [BLOCK_N] tokens -----
        z_n = tl.load(z_ptr + n_offs * h + j_safe,
                      mask=n_mask & j_in, other=0.0).to(tl.float32)
        u = (z_n - grid_lo) * scale
        in_range = (u >= 0.0) & (u <= G_max)
        clamp_active = (u >= 0.0) & (u <= G_max - 1.0)
        u_clip = tl.minimum(tl.maximum(u, 0.0), G_max - 1.0)
        bin_idx = u_clip.to(tl.int32)
        tau = u_clip - bin_idx.to(tl.float32)
        omt = 1.0 - tau

        B0 = 0.5 * omt * omt
        B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
        B2 = 0.5 * tau * tau
        dB0 = -omt
        dB1 = 1.0 - 2.0 * tau
        dB2 = tau

        ir_f = in_range.to(tl.float32)
        cl_f = clamp_active.to(tl.float32)
        j_in_f = tl.where(j_in, 1.0, 0.0)
        B0 = B0 * ir_f * j_in_f
        B1 = B1 * ir_f * j_in_f
        B2 = B2 * ir_f * j_in_f
        dB0 = dB0 * cl_f * j_in_f
        dB1 = dB1 * cl_f * j_in_f
        dB2 = dB2 * cl_f * j_in_f

        # ----- dz path (irregular gather — same as v1) -----
        base_j = C_ptr + j_safe * (L * r)
        mask_2d = n_mask[:, None] & r_mask[None, :] & j_in
        c0 = tl.load(base_j + bin_idx[:, None] * r + r_offs[None, :],
                     mask=mask_2d, other=0.0).to(tl.float32)
        c1 = tl.load(base_j + (bin_idx + 1)[:, None] * r + r_offs[None, :],
                     mask=mask_2d, other=0.0).to(tl.float32)
        c2 = tl.load(base_j + (bin_idx + 2)[:, None] * r + r_offs[None, :],
                     mask=mask_2d, other=0.0).to(tl.float32)
        inner0 = tl.sum(c0 * g_delta, axis=1)
        inner1 = tl.sum(c1 * g_delta, axis=1)
        inner2 = tl.sum(c2 * g_delta, axis=1)
        dz_val = scale * (dB0 * inner0 + dB1 * inner1 + dB2 * inner2)
        tl.store(dz_ptr + n_offs * h + j_safe,
                 dz_val, mask=n_mask & j_in)

        # ----- dC path — DENSE MATMUL via tl.dot -----
        # Build weight matrix W [BLOCK_N, L_PAD]:
        #   W[n, b] = sum_k B_k * 1{bin_n + k == b}
        # Only b ∈ [0, L) are valid; pad columns are zero.
        # bb, valid_bin_f, b_in_range all hoisted out of the loop.
        bin_col = bin_idx[:, None]                    # [BLOCK_N, 1]
        m0 = (bin_col == bb).to(tl.float32)
        m1 = ((bin_col + 1) == bb).to(tl.float32)
        m2 = ((bin_col + 2) == bb).to(tl.float32)
        W = (B0[:, None] * m0
             + B1[:, None] * m1
             + B2[:, None] * m2) * valid_bin_f         # [BLOCK_N, L_PAD]

        # tl.dot expects (M, K) @ (K, N).  W.T @ g_delta:
        # → (L_PAD, BLOCK_N) @ (BLOCK_N, BLOCK_R) = (L_PAD, BLOCK_R)
        if USE_BF16_DOT:
            # Production bf16 path: tensor-core dispatch via bf16 inputs.
            # g_delta_bf was pre-cast once outside the loop (saves 16 casts
            # per program).  W is fp32→bf16 inside loop because W changes
            # per j.  Output accumulator stays fp32.  bf16 inputs naturally
            # use H100/Ampere wmma tensor cores (no explicit precision flag
            # needed — that's only for fp32-input override).
            WT_bf = tl.trans(W).to(tl.bfloat16)
            dC_local = tl.dot(WT_bf, g_delta_bf, out_dtype=tl.float32)
        else:
            # Test/fp32 path: keep fp32 + disable TF32 to match the fp32
            # numeric oracle.  ``allow_tf32=False`` forces full fp32 mantissa.
            WT = tl.trans(W)
            dC_local = tl.dot(WT, g_delta, out_dtype=tl.float32,
                              allow_tf32=False)

        # ----- Atomic add dC[j_safe, :L, :] (row b, all c) -----
        # b_in_range hoisted above; dC_addr computed per j (j varies).
        dC_addr = (dC_ptr
                   + j_safe * (L * r)
                   + b_offs_pad[:, None] * r
                   + r_offs[None, :])                  # [L_PAD, BLOCK_R]
        dC_mask = (b_in_range[:, None]
                   & r_mask[None, :]
                   & j_in)
        tl.atomic_add(dC_addr, dC_local, mask=dC_mask)


def flash_spline_delta_backward_v3(
    z: torch.Tensor, C: torch.Tensor, g_delta: torch.Tensor,
    grid_lo: float, grid_hi: float, G: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v3 backward: tl.dot-based dC matmul (tensor cores) + atomic at-grain.

    Atomic count drops 35× vs v1 (one atomic per (j, b, c) cell instead
    of per-token).  Tensor cores accelerate the matmul portion.
    """
    if not z.is_cuda:
        raise RuntimeError("v3 needs CUDA tensors")
    N, h = z.shape
    h_C, L, r = C.shape
    if h != h_C or L != G + 2:
        raise ValueError("shape mismatch")

    z_c = z.contiguous(); C_c = C.contiguous(); g_c = g_delta.contiguous()
    dC = torch.zeros((h, L, r), device=z.device, dtype=torch.float32)
    dz = torch.zeros((N, h), device=z.device, dtype=torch.float32)

    BLOCK_R = max(16, triton.next_power_of_2(r))
    L_PAD = max(16, triton.next_power_of_2(L))
    scale = G / (grid_hi - grid_lo)
    # Use bf16 tl.dot path when input g_delta is bf16/fp16 — engages
    # H100/Ampere wmma tensor cores.  For fp32 inputs (tests) we keep
    # fp32 to match the numeric reference.
    use_bf16_dot = g_delta.dtype in (torch.bfloat16, torch.float16)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),
                         triton.cdiv(r, BLOCK_R),
                         triton.cdiv(h, meta["BLOCK_H"]))
    _flash_spline_feature_delta_bwd_v3[grid](
        z_c, C_c, g_c, dC, dz,
        float(grid_lo), float(scale), float(G),
        N, h, r, L,
        L_PAD=L_PAD, BLOCK_R=BLOCK_R,
        USE_BF16_DOT=use_bf16_dot,
    )
    return dC, dz


def _autotune_bwd_v4_configs():
    """v4 backward: same search space as v3 but with extra constraint
    that BLOCK_R must be ≥ 16 for tl.dot to use tensor cores cleanly.
    """
    cfgs = []
    for BN in [32, 64]:
        for BH in [16, 32]:
            for nw in [4, 8]:
                cfgs.append(triton.Config(
                    {"BLOCK_N": BN, "BLOCK_H": BH},
                    num_warps=nw, num_stages=2,
                ))
    for BN, BH in [(32, 32), (64, 16)]:
        cfgs.append(triton.Config(
            {"BLOCK_N": BN, "BLOCK_H": BH},
            num_warps=4, num_stages=3,
        ))
    return cfgs


@triton.autotune(configs=_autotune_bwd_v4_configs(),
                  key=["N", "h", "r", "L", "USE_BF16_DOT"],
                  reset_to_zero=["dC_ptr", "dz_ptr"])
@triton.jit
def _flash_spline_feature_delta_bwd_v4(
    z_ptr, C_ptr, g_delta_ptr,
    dC_ptr, dz_ptr,
    grid_lo, scale, G_max,
    N: tl.constexpr, h: tl.constexpr, r: tl.constexpr, L: tl.constexpr,
    L_PAD: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """v4 backward — both dC AND dz paths via tl.dot tensor-core matmul.

    Key change vs v3:
        * v3 dz path: 3 irregular per-token gathers from C, then 3 small
          inner-product reductions per j.
        * v4 dz path: 1 contiguous load of C[j, :, :] tile [L_PAD, BLOCK_R],
          1 tl.dot for all (n, b) inner products, then mask-select per k.

    Effect:
        * Compute increases (~10× more MAC ops in dz inner)
        * BUT moves from scalar fp32 ops → bf16 tensor-core matmul
          which is ~10× faster on H100
        * Memory: irregular gather → contiguous tile load (much better
          L2 hit pattern)

    Expected speedup vs v3: 1.5-2× on the dz portion of bwd kernel.
    """
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_h = tl.program_id(2)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    h_start = pid_h * BLOCK_H
    n_mask = n_offs < N
    r_mask = r_offs < r

    # Load g_delta tile once
    g_addr = g_delta_ptr + n_offs[:, None] * r + r_offs[None, :]
    g_mask = n_mask[:, None] & r_mask[None, :]
    g_delta = tl.load(g_addr, mask=g_mask, other=0.0).to(tl.float32)

    b_offs_pad = tl.arange(0, L_PAD)
    b_in_range = (b_offs_pad < L)

    for j_local in tl.static_range(BLOCK_H):
        j = h_start + j_local
        j_in = j < h
        j_safe = tl.minimum(j, h - 1)

        # Compute B/dB/bin/τ for [BLOCK_N] tokens (same as v3)
        z_n = tl.load(z_ptr + n_offs * h + j_safe,
                      mask=n_mask & j_in, other=0.0).to(tl.float32)
        u = (z_n - grid_lo) * scale
        in_range = (u >= 0.0) & (u <= G_max)
        clamp_active = (u >= 0.0) & (u <= G_max - 1.0)
        u_clip = tl.minimum(tl.maximum(u, 0.0), G_max - 1.0)
        bin_idx = u_clip.to(tl.int32)
        tau = u_clip - bin_idx.to(tl.float32)
        omt = 1.0 - tau

        B0 = 0.5 * omt * omt
        B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
        B2 = 0.5 * tau * tau
        dB0 = -omt
        dB1 = 1.0 - 2.0 * tau
        dB2 = tau

        ir_f = in_range.to(tl.float32)
        cl_f = clamp_active.to(tl.float32)
        j_in_f = tl.where(j_in, 1.0, 0.0)
        B0 = B0 * ir_f * j_in_f; B1 = B1 * ir_f * j_in_f; B2 = B2 * ir_f * j_in_f
        dB0 = dB0 * cl_f * j_in_f; dB1 = dB1 * cl_f * j_in_f; dB2 = dB2 * cl_f * j_in_f

        # ----- dz path — DENSE MATMUL approach -----
        # Load full C[j_safe, :L, :] tile (CONTIGUOUS, no irregular gather).
        # We pad to L_PAD by masking; out-of-range bins read garbage but
        # mask zeroes them.
        C_addr_full = (C_ptr + j_safe * (L * r)
                        + b_offs_pad[:, None] * r
                        + r_offs[None, :])                    # [L_PAD, BLOCK_R]
        C_load_mask = (b_in_range[:, None]
                       & r_mask[None, :]
                       & j_in)
        C_full = tl.load(C_addr_full, mask=C_load_mask, other=0.0).to(tl.float32)

        # all_inners[n, b] = sum_c C_full[b, c] * g_delta[n, c]
        # = g_delta @ C_full.T   shape (BLOCK_N, L_PAD)
        if USE_BF16_DOT:
            G_bf = g_delta.to(tl.bfloat16)
            CT_bf = tl.trans(C_full).to(tl.bfloat16)
            all_inners = tl.dot(G_bf, CT_bf, out_dtype=tl.float32)
        else:
            all_inners = tl.dot(g_delta, tl.trans(C_full),
                                 out_dtype=tl.float32, allow_tf32=False)

        # Per-token bin masks: [BLOCK_N, L_PAD]
        bin_col = bin_idx[:, None]
        bb = b_offs_pad[None, :]
        m_k0 = (bin_col == bb).to(tl.float32)
        m_k1 = ((bin_col + 1) == bb).to(tl.float32)
        m_k2 = ((bin_col + 2) == bb).to(tl.float32)

        # inner_k[n] = all_inners[n, bin[n]+k]  (one entry per n)
        inner0 = tl.sum(all_inners * m_k0, axis=1)
        inner1 = tl.sum(all_inners * m_k1, axis=1)
        inner2 = tl.sum(all_inners * m_k2, axis=1)
        dz_val = scale * (dB0 * inner0 + dB1 * inner1 + dB2 * inner2)
        tl.store(dz_ptr + n_offs * h + j_safe,
                 dz_val, mask=n_mask & j_in)

        # ----- dC path — same as v3 (W matrix + tl.dot W.T @ g_delta) -----
        valid_bin = (b_offs_pad < L).to(tl.float32)[None, :]   # [1, L_PAD]
        W = (B0[:, None] * m_k0
             + B1[:, None] * m_k1
             + B2[:, None] * m_k2) * valid_bin                  # [BLOCK_N, L_PAD]

        if USE_BF16_DOT:
            WT_bf = tl.trans(W).to(tl.bfloat16)
            G_bf2 = g_delta.to(tl.bfloat16)
            dC_local = tl.dot(WT_bf, G_bf2, out_dtype=tl.float32)
        else:
            WT = tl.trans(W)
            dC_local = tl.dot(WT, g_delta, out_dtype=tl.float32,
                              allow_tf32=False)

        dC_addr = (dC_ptr
                   + j_safe * (L * r)
                   + b_offs_pad[:, None] * r
                   + r_offs[None, :])
        dC_mask = (b_in_range[:, None] & r_mask[None, :] & j_in)
        tl.atomic_add(dC_addr, dC_local, mask=dC_mask)


def flash_spline_delta_backward_v4(
    z: torch.Tensor, C: torch.Tensor, g_delta: torch.Tensor,
    grid_lo: float, grid_hi: float, G: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v4 backward: both dC and dz via tl.dot tensor-core matmuls.
    Same correctness contract as v3; should be 1.5-2× faster on dz path
    when bf16 inputs allow tensor-core dispatch.
    """
    if not z.is_cuda:
        raise RuntimeError("v4 needs CUDA tensors")
    N, h = z.shape
    h_C, L, r = C.shape
    if h != h_C or L != G + 2:
        raise ValueError("shape mismatch")

    z_c = z.contiguous(); C_c = C.contiguous(); g_c = g_delta.contiguous()
    dC = torch.zeros((h, L, r), device=z.device, dtype=torch.float32)
    dz = torch.zeros((N, h), device=z.device, dtype=torch.float32)

    BLOCK_R = max(16, triton.next_power_of_2(r))
    L_PAD = max(16, triton.next_power_of_2(L))
    scale = G / (grid_hi - grid_lo)
    use_bf16_dot = g_delta.dtype in (torch.bfloat16, torch.float16)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),
                         triton.cdiv(r, BLOCK_R),
                         triton.cdiv(h, meta["BLOCK_H"]))
    _flash_spline_feature_delta_bwd_v4[grid](
        z_c, C_c, g_c, dC, dz,
        float(grid_lo), float(scale), float(G),
        N, h, r, L,
        L_PAD=L_PAD, BLOCK_R=BLOCK_R,
        USE_BF16_DOT=use_bf16_dot,
    )
    return dC, dz


def _autotune_bwd_v5_configs():
    """v5 backward: BLOCK_J chunks of j collapsed into single tl.dot.
    Constraints: BLOCK_N * BLOCK_J * L_PAD must fit in registers (~ 32K fp32 = 128 KB).
    Smaller BLOCK_N (32) is necessary; BLOCK_J ∈ {4, 8} balance matmul size vs reg pressure.
    """
    cfgs = []
    for BN in [16, 32]:
        for BJ in [4, 8]:
            for nw in [4, 8]:
                cfgs.append(triton.Config(
                    {"BLOCK_N": BN, "BLOCK_J_BATCH": BJ},
                    num_warps=nw, num_stages=2,
                ))
    for BN, BJ in [(32, 4), (32, 8)]:
        cfgs.append(triton.Config(
            {"BLOCK_N": BN, "BLOCK_J_BATCH": BJ},
            num_warps=4, num_stages=3,
        ))
    return cfgs


@triton.autotune(configs=_autotune_bwd_v5_configs(),
                  key=["N", "h", "r", "L", "USE_BF16_DOT"],
                  reset_to_zero=["dC_ptr", "dz_ptr"])
@triton.jit
def _flash_spline_feature_delta_bwd_v5(
    z_ptr, C_ptr, g_delta_ptr,
    dC_ptr, dz_ptr,
    grid_lo, scale, G_max,
    N: tl.constexpr, h: tl.constexpr, r: tl.constexpr, L: tl.constexpr,
    L_PAD: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_J_BATCH: tl.constexpr,
):
    """v5 backward — batches BLOCK_J_BATCH j's into ONE big tl.dot.

    Key change vs v3:
        v3: per-j tl.dot of shape (L_PAD=32, BLOCK_N=64, BLOCK_R=32)
            → small tile, undersaturated tensor cores, 16 dots total
        v5: collapse BLOCK_J_BATCH js into one big tl.dot
            (BLOCK_J_BATCH * L_PAD, BLOCK_N, BLOCK_R) e.g. (256, 32, 32)
            → bigger M dim, better wmma utilization, fewer dots

    Trade-offs:
        + Bigger matmul → tensor cores more saturated
        - Higher register pressure: W_3d [BLOCK_N, BLOCK_J_BATCH, L_PAD]
        - More compute per dot but ONE issue (better instruction scheduler)

    dz path stays per-j (irregular gather with c0/c1/c2) — v4 showed
    matmul-ifying dz doesn't help.
    """
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_h = tl.program_id(2)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    h_start = pid_h * BLOCK_H
    n_mask = n_offs < N
    r_mask = r_offs < r

    g_addr = g_delta_ptr + n_offs[:, None] * r + r_offs[None, :]
    g_mask = n_mask[:, None] & r_mask[None, :]
    g_delta = tl.load(g_addr, mask=g_mask, other=0.0).to(tl.float32)
    if USE_BF16_DOT:
        g_delta_bf = g_delta.to(tl.bfloat16)

    b_offs_pad = tl.arange(0, L_PAD)
    b_in_range = (b_offs_pad < L)
    valid_bin_f = b_in_range.to(tl.float32)
    bb = b_offs_pad[None, None, :]                        # [1, 1, L_PAD]

    j_off_arr = tl.arange(0, BLOCK_J_BATCH)              # [BLOCK_J_BATCH]

    # Outer loop processes BLOCK_J_BATCH j's at a time
    for j_block_start in range(0, BLOCK_H, BLOCK_J_BATCH):
        # j_global indices for this batch
        j_offs = h_start + j_block_start + j_off_arr      # [BLOCK_J_BATCH]
        j_in_arr = j_offs < h                             # [BLOCK_J_BATCH] bool
        j_safe_arr = tl.minimum(j_offs, h - 1)            # [BLOCK_J_BATCH]

        # Load z[BLOCK_N, BLOCK_J_BATCH] (batched)
        z_addr = z_ptr + n_offs[:, None] * h + j_safe_arr[None, :]
        z_load_mask = n_mask[:, None] & j_in_arr[None, :]
        z_2d = tl.load(z_addr, mask=z_load_mask, other=0.0).to(tl.float32)
        # Compute bin/B/dB for [BLOCK_N, BLOCK_J_BATCH]
        u = (z_2d - grid_lo) * scale
        in_range = (u >= 0.0) & (u <= G_max)
        clamp_active = (u >= 0.0) & (u <= G_max - 1.0)
        u_clip = tl.minimum(tl.maximum(u, 0.0), G_max - 1.0)
        bin_2d = u_clip.to(tl.int32)                      # [BLOCK_N, BLOCK_J_BATCH]
        tau = u_clip - bin_2d.to(tl.float32)
        omt = 1.0 - tau
        B0 = 0.5 * omt * omt
        B1 = 0.5 * (1.0 + 2.0 * tau - 2.0 * tau * tau)
        B2 = 0.5 * tau * tau
        dB0 = -omt
        dB1 = 1.0 - 2.0 * tau
        dB2 = tau
        ir_f = in_range.to(tl.float32)
        cl_f = clamp_active.to(tl.float32)
        j_in_f = (j_in_arr[None, :]).to(tl.float32)       # [1, BLOCK_J_BATCH]
        B0 = B0 * ir_f * j_in_f; B1 = B1 * ir_f * j_in_f; B2 = B2 * ir_f * j_in_f
        dB0 = dB0 * cl_f * j_in_f; dB1 = dB1 * cl_f * j_in_f; dB2 = dB2 * cl_f * j_in_f

        # ----- dz path (still per-j, irregular gather, same as v3) -----
        # We must do this inner loop per j because c gather is irregular.
        for jj in tl.static_range(BLOCK_J_BATCH):
            j_safe = h_start + j_block_start + jj
            j_safe = tl.minimum(j_safe, h - 1)
            j_in = (h_start + j_block_start + jj) < h

            # Slice the precomputed [BLOCK_N, BLOCK_J_BATCH] arrays at jj
            # Triton requires this be a static index → compile-time constant.
            # We use a mask trick: jj_mask[k] = (k == jj).
            jj_mask_1d = (tl.arange(0, BLOCK_J_BATCH) == jj).to(tl.float32)  # [BLOCK_J_BATCH]
            # bin slice: extract column jj from bin_2d via masked sum
            bin_slice = tl.sum(bin_2d.to(tl.float32) * jj_mask_1d[None, :], axis=1).to(tl.int32)  # [BLOCK_N]
            B0_slice = tl.sum(B0 * jj_mask_1d[None, :], axis=1)              # [BLOCK_N]
            B1_slice = tl.sum(B1 * jj_mask_1d[None, :], axis=1)
            B2_slice = tl.sum(B2 * jj_mask_1d[None, :], axis=1)
            dB0_slice = tl.sum(dB0 * jj_mask_1d[None, :], axis=1)
            dB1_slice = tl.sum(dB1 * jj_mask_1d[None, :], axis=1)
            dB2_slice = tl.sum(dB2 * jj_mask_1d[None, :], axis=1)

            base_j = C_ptr + j_safe * (L * r)
            mask_2d = n_mask[:, None] & r_mask[None, :] & j_in
            c0 = tl.load(base_j + bin_slice[:, None] * r + r_offs[None, :],
                         mask=mask_2d, other=0.0).to(tl.float32)
            c1 = tl.load(base_j + (bin_slice + 1)[:, None] * r + r_offs[None, :],
                         mask=mask_2d, other=0.0).to(tl.float32)
            c2 = tl.load(base_j + (bin_slice + 2)[:, None] * r + r_offs[None, :],
                         mask=mask_2d, other=0.0).to(tl.float32)
            inner0 = tl.sum(c0 * g_delta, axis=1)
            inner1 = tl.sum(c1 * g_delta, axis=1)
            inner2 = tl.sum(c2 * g_delta, axis=1)
            dz_val = scale * (dB0_slice * inner0 + dB1_slice * inner1 + dB2_slice * inner2)
            tl.store(dz_ptr + n_offs * h + j_safe,
                     dz_val, mask=n_mask & j_in)

        # ----- dC path — BATCHED MATMUL across BLOCK_J_BATCH j's -----
        # Build W_3d [BLOCK_N, BLOCK_J_BATCH, L_PAD]:
        #   W_3d[n, j_local, b] = sum_k B_k[n, j_local] * 1{bin[n, j_local]+k == b}
        bin_3d = bin_2d[:, :, None]                       # [BLOCK_N, BLOCK_J_BATCH, 1]
        m0_3d = (bin_3d == bb).to(tl.float32)             # [BLOCK_N, BLOCK_J_BATCH, L_PAD]
        m1_3d = ((bin_3d + 1) == bb).to(tl.float32)
        m2_3d = ((bin_3d + 2) == bb).to(tl.float32)
        W_3d = (B0[:, :, None] * m0_3d
                + B1[:, :, None] * m1_3d
                + B2[:, :, None] * m2_3d) * valid_bin_f[None, None, :]
        # Reshape to W_2d [BLOCK_N, BLOCK_J_BATCH * L_PAD]
        W_2d = tl.reshape(W_3d, (BLOCK_N, BLOCK_J_BATCH * L_PAD))

        # Single big tl.dot:
        #   (BLOCK_J_BATCH * L_PAD, BLOCK_N) @ (BLOCK_N, BLOCK_R) = (BLOCK_J_BATCH * L_PAD, BLOCK_R)
        if USE_BF16_DOT:
            W_2d_T_bf = tl.trans(W_2d).to(tl.bfloat16)    # [BJB*L_PAD, BLOCK_N]
            dC_batch = tl.dot(W_2d_T_bf, g_delta_bf, out_dtype=tl.float32)
        else:
            dC_batch = tl.dot(tl.trans(W_2d), g_delta,
                              out_dtype=tl.float32, allow_tf32=False)
        # dC_batch shape [BLOCK_J_BATCH * L_PAD, BLOCK_R]
        # Reshape into [BLOCK_J_BATCH, L_PAD, BLOCK_R]
        dC_3d = tl.reshape(dC_batch, (BLOCK_J_BATCH, L_PAD, BLOCK_R))

        # Atomic add per j to global dC[j_global, :L, :].
        # Static_range over j_local — each iter does one tl.atomic_add tile.
        for jj in tl.static_range(BLOCK_J_BATCH):
            j_safe = tl.minimum(h_start + j_block_start + jj, h - 1)
            j_in = (h_start + j_block_start + jj) < h
            # Slice dC_3d[jj, :, :] via static index
            jj_mask_1d = (tl.arange(0, BLOCK_J_BATCH) == jj).to(tl.float32)  # [BJB]
            # slice = sum over j-axis with mask
            dC_slice = tl.sum(dC_3d * jj_mask_1d[:, None, None], axis=0)  # [L_PAD, BLOCK_R]

            dC_addr = (dC_ptr + j_safe * (L * r)
                       + b_offs_pad[:, None] * r
                       + r_offs[None, :])
            dC_mask = (b_in_range[:, None] & r_mask[None, :] & j_in)
            tl.atomic_add(dC_addr, dC_slice, mask=dC_mask)


def flash_spline_delta_backward_v5(
    z: torch.Tensor, C: torch.Tensor, g_delta: torch.Tensor,
    grid_lo: float, grid_hi: float, G: int,
    BLOCK_H: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v5 backward: batched-j tl.dot for dC (single big matmul).

    BLOCK_H is fixed (caller-controlled); BLOCK_N and BLOCK_J_BATCH are
    autotuned.  Constraint: BLOCK_J_BATCH must divide BLOCK_H evenly.
    """
    if not z.is_cuda:
        raise RuntimeError("v5 needs CUDA tensors")
    N, h = z.shape
    h_C, L, r = C.shape
    if h != h_C or L != G + 2:
        raise ValueError("shape mismatch")

    z_c = z.contiguous(); C_c = C.contiguous(); g_c = g_delta.contiguous()
    dC = torch.zeros((h, L, r), device=z.device, dtype=torch.float32)
    dz = torch.zeros((N, h), device=z.device, dtype=torch.float32)

    BLOCK_R = max(16, triton.next_power_of_2(r))
    L_PAD = max(16, triton.next_power_of_2(L))
    scale = G / (grid_hi - grid_lo)
    use_bf16_dot = g_delta.dtype in (torch.bfloat16, torch.float16)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),
                         triton.cdiv(r, BLOCK_R),
                         triton.cdiv(h, BLOCK_H))
    _flash_spline_feature_delta_bwd_v5[grid](
        z_c, C_c, g_c, dC, dz,
        float(grid_lo), float(scale), float(G),
        N, h, r, L,
        L_PAD=L_PAD, BLOCK_R=BLOCK_R, BLOCK_H=BLOCK_H,
        USE_BF16_DOT=use_bf16_dot,
    )
    return dC, dz


def flash_spline_delta_backward(
    z: torch.Tensor,             # [N, h]
    C: torch.Tensor,             # [h, L, r]
    g_delta: torch.Tensor,       # [N, r]
    grid_lo: float,
    grid_hi: float,
    G: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """B2.4 backward kernel.  Returns (dC, dz) both in fp32."""
    if not z.is_cuda or not C.is_cuda or not g_delta.is_cuda:
        raise RuntimeError("flash_spline_delta_backward needs CUDA tensors")
    N, h = z.shape
    h_C, L, r = C.shape
    if h != h_C or L != G + 2:
        raise ValueError("shape mismatch")
    if g_delta.shape != (N, r):
        raise ValueError(f"g_delta shape {tuple(g_delta.shape)} != [{N}, {r}]")

    z_c = z.contiguous()
    C_c = C.contiguous()
    g_c = g_delta.contiguous()

    # Outputs MUST be zero-initialized — both dC (atomic_add target) and dz
    # (we write only the `j_in` positions; zero-init protects the masked-out
    # positions at the end if N or h aren't multiples of block sizes).
    dC = torch.zeros((h, L, r), device=z.device, dtype=torch.float32)
    dz = torch.zeros((N, h), device=z.device, dtype=torch.float32)

    BLOCK_R = max(16, triton.next_power_of_2(r))
    scale = G / (grid_hi - grid_lo)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),
                         triton.cdiv(r, BLOCK_R),
                         triton.cdiv(h, meta["BLOCK_H"]))
    _flash_spline_feature_delta_bwd_v1[grid](
        z_c, C_c, g_c, dC, dz,
        float(grid_lo), float(scale), float(G),
        N, h, r, L,
        BLOCK_R=BLOCK_R,
    )
    return dC, dz


__all__ = [
    "flash_spline_delta_forward",
    "flash_spline_delta_forward_v2",
    "flash_spline_delta_forward_v3",
    "flash_spline_delta_forward_v4",
    "flash_spline_delta_forward_v5",
    "flash_spline_delta_backward_v2",
    "flash_spline_delta_backward_v4",
    "flash_spline_feature_forward",
]
