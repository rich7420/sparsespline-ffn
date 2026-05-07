"""JIT-loaded CUDA extensions for FlashSplineFeature backward.

Two implementations:
  * spline_kv_bwd_cuda      — v1: scalar SMEM atomicAdd (slow, kept as ref)
  * spline_kv_bwd_wmma_cuda — v2: bf16 wmma tensor cores (fast)

Compiled lazily on first use.
"""
from __future__ import annotations

from pathlib import Path

import torch

_HERE = Path(__file__).parent
_EXT_V1 = None
_EXT_V2 = None
_EXT_V3 = None
_EXT_V4 = None  # wgmma — sm_90 only
_EXT_V5 = None  # wgmma v2 (output-parallel grid, no dC atomic) — sm_90 only
_EXT_V6 = None  # wgmma v3 (split-N + 2-stage cp.async pipeline) — sm_90 only
_EXT_V7 = None  # wgmma v4 (scratch + reduce, BN=128, no atomic) — sm_90 only
_EXT_V8 = None  # wgmma v5 (register-resident dC + fp16 wgmma) — sm_90 only
_EXT_BWD_V6 = None  # bwd v6 (incremental FA3 pattern; v6.0 = v5 clone) — sm_90 only
_EXT_WGMMA_TMA_TEST = None  # standalone TMA→WGMMA descriptor test — sm_90 only
_EXT_FWD = None  # native CUDA forward (sm_80+)
_EXT_FWD_V3 = None  # v3 forward (Hopper-aligned, single-CTA-per-n-chunk, no atomic) — sm_90
_EXT_FWD_V10 = None  # v10 forward (dense-W wgmma, split-H grid) — sm_90
_EXT_FWD_V11 = None  # v11 forward (v10 + fp16 W for precision) — sm_90


def _load(name: str, source: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    from torch.utils.cpp_extension import load
    return load(
        name=name,
        sources=[str(_HERE / source)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode", "arch=compute_80,code=sm_80",   # Ampere (3080)
            "-gencode", "arch=compute_90,code=sm_90",   # Hopper (H100)
        ],
        verbose=False,
    )


def get_ext_v1():
    global _EXT_V1
    if _EXT_V1 is None:
        _EXT_V1 = _load("spline_kv_bwd_cuda_ext", "spline_kv_bwd.cu")
    return _EXT_V1


def get_ext_v2():
    global _EXT_V2
    if _EXT_V2 is None:
        _EXT_V2 = _load("spline_kv_bwd_wmma_ext", "spline_kv_bwd_wmma.cu")
    return _EXT_V2


def get_ext_v3():
    global _EXT_V3
    if _EXT_V3 is None:
        _EXT_V3 = _load("spline_kv_bwd_hopper_ext", "spline_kv_bwd_hopper.cu")
    return _EXT_V3


def get_ext_v4():
    """Hopper wgmma kernel — sm_90 only."""
    global _EXT_V4
    if _EXT_V4 is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        from torch.utils.cpp_extension import load
        _EXT_V4 = load(
            name="spline_kv_bwd_wgmma_ext",
            sources=[str(_HERE / "spline_kv_bwd_wgmma.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode", "arch=compute_90a,code=sm_90a",  # Hopper + arch-specific (wgmma, TMA)
                "-std=c++17",
                "--extended-lambda",
            ],
            verbose=False,
        )
    return _EXT_V4


def get_ext_v7():
    """Hopper wgmma v4 — scratch + reduce, no global atomic on dC, sm_90 only."""
    global _EXT_V7
    if _EXT_V7 is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        from torch.utils.cpp_extension import load
        _EXT_V7 = load(
            name="spline_kv_bwd_wgmma_v4_ext",
            sources=[str(_HERE / "spline_kv_bwd_wgmma_v4.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "-std=c++17",
                "--extended-lambda",
            ],
            verbose=False,
        )
    return _EXT_V7


def spline_kv_bwd_wgmma_v4_cuda(z, C, g_delta, grid_lo, grid_hi, G):
    """v4 Hopper wgmma backward — scratch + reduce, no global atomic.

    Returns (dC bf16, dz fp32).  Internally:
      - main kernel writes dC contributions to fp32 scratch [N_TILE, H, L, R]
      - reduce kernel sums n_tile dim and casts to bf16 dC.
    """
    if not (z.is_cuda and C.is_cuda and g_delta.is_cuda):
        raise RuntimeError("CUDA-only")
    z, C, g_delta = _coerce(z, C, g_delta)
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError("shape mismatch")
    scale = G / (grid_hi - grid_lo)
    return get_ext_v7().spline_kv_bwd_wgmma_v4_cuda(
        z, C, g_delta, float(grid_lo), float(scale)
    )


def get_ext_v8():
    """Hopper wgmma v5 — register-resident dC + fp16 wgmma, sm_90 only."""
    global _EXT_V8
    if _EXT_V8 is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        from torch.utils.cpp_extension import load
        _EXT_V8 = load(
            name="spline_kv_bwd_wgmma_v5_ext",
            sources=[str(_HERE / "spline_kv_bwd_wgmma_v5.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "-std=c++17",
                "--extended-lambda",
                # Print ptxas register / spill / SMEM info during build —
                # critical for diagnosing whether (128,2) launch_bounds
                # forces register spilling vs (128,1) for v5 optimization.
                # Output appears in Modal stdout when verbose=True.
                "-Xptxas=-v",
                "-lineinfo",
            ],
            verbose=True,
        )
    return _EXT_V8


def spline_kv_bwd_wgmma_v5_cuda(z, C, g_delta, grid_lo, grid_hi, G):
    """v5 Hopper wgmma backward — register-resident dC across chunks +
    fp16 wgmma (precision-corrected, matches v11 fwd's strategy).

    Returns (dC bf16, dz fp32).
    """
    if not (z.is_cuda and C.is_cuda and g_delta.is_cuda):
        raise RuntimeError("CUDA-only")
    z, C, g_delta = _coerce(z, C, g_delta)
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError("shape mismatch")
    scale = G / (grid_hi - grid_lo)
    return get_ext_v8().spline_kv_bwd_wgmma_v5_cuda(
        z, C, g_delta, float(grid_lo), float(scale), int(L)
    )


# =============================================================================
# bwd v6 — FA3-pattern roadmap (TMA + warp-spec); v6.0 = v5-equivalent clone.
# Implementation phases: see top of spline_kv_bwd_v6.cu for the full plan.
# =============================================================================

def get_ext_bwd_v6():
    """v6 — incremental FA3-pattern bwd kernel. v6.1a: TMA load for C_smem.

    Build links against `-lcuda` (CUDA Driver API) for cuTensorMapEncodeTiled.
    """
    global _EXT_BWD_V6
    if _EXT_BWD_V6 is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        from torch.utils.cpp_extension import load
        _EXT_BWD_V6 = load(
            name="spline_kv_bwd_v6_ext",
            sources=[str(_HERE / "spline_kv_bwd_v6.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "-std=c++17",
                "--extended-lambda",
                # ptxas verbose for register/spill/SMEM diagnostics —
                # mandatory while iterating through phases v6.1a..v6.4.
                "-Xptxas=-v",
                "-lineinfo",
            ],
            # v6.1a needs CUDA Driver API for cuTensorMapEncodeTiled.
            # libcuda.so is the user-mode driver lib (not cudart).
            extra_ldflags=["-lcuda"],
            verbose=True,
        )
    return _EXT_BWD_V6


def spline_kv_bwd_v6_cuda(z, C, g_delta, grid_lo, grid_hi, G):
    """v6 Hopper bwd — incremental FA3 pattern. v6.0 = v5-equivalent.

    Phases v6.1a / v6.1b / v6.2 / v6.3 / v6.4 land iteratively. Each phase
    keeps the dC max_rel ≤ 5e-3 parity gate and reports its own ptxas data.
    Returns (dC bf16, dz fp32) — same contract as v5.
    """
    if not (z.is_cuda and C.is_cuda and g_delta.is_cuda):
        raise RuntimeError("CUDA-only")
    z, C, g_delta = _coerce(z, C, g_delta)
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError("shape mismatch")
    scale = G / (grid_hi - grid_lo)
    return get_ext_bwd_v6().spline_kv_bwd_v6_cuda(
        z, C, g_delta, float(grid_lo), float(scale), int(L)
    )


def get_ext_wgmma_tma_test():
    """Standalone TMA→WGMMA descriptor encoding test (v6.1b training wheel)."""
    global _EXT_WGMMA_TMA_TEST
    if _EXT_WGMMA_TMA_TEST is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        from torch.utils.cpp_extension import load
        _EXT_WGMMA_TMA_TEST = load(
            name="wgmma_tma_test_ext",
            sources=[str(_HERE / "wgmma_tma_test.cu")],
            extra_cuda_cflags=[
                "-O3", "--use_fast_math",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "-std=c++17", "--extended-lambda",
                "-Xptxas=-v", "-lineinfo",
            ],
            extra_ldflags=["-lcuda"],
            verbose=True,
        )
    return _EXT_WGMMA_TMA_TEST


def wgmma_tma_test(A, B, variant: int):
    """Run the standalone TMA→WGMMA m64n32k16 test.

    A:       [64, 16] fp16 CUDA tensor
    B:       [16, 32] fp16 CUDA tensor
    variant: 0..5 — see wgmma_tma_test.cu top comment for the variant table.

    Returns (D [64, 32] fp32, B_smem_dump [16, 32] fp16). Compare D to
    torch.matmul(A.float(), B.float()). Variant 5 also fills B_smem_dump
    with the SW64-swizzled SMEM contents for offline layout inspection;
    other variants leave it zero.
    """
    return get_ext_wgmma_tma_test().wgmma_tma_test(A, B, int(variant))


def get_ext_v6():
    """Hopper wgmma v3 — split-N + 2-stage cp.async pipeline, sm_90 only."""
    global _EXT_V6
    if _EXT_V6 is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        from torch.utils.cpp_extension import load
        _EXT_V6 = load(
            name="spline_kv_bwd_wgmma_v3_ext",
            sources=[str(_HERE / "spline_kv_bwd_wgmma_v3.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "-std=c++17",
                "--extended-lambda",
            ],
            verbose=False,
        )
    return _EXT_V6


def spline_kv_bwd_wgmma_v3_cuda(z, C, g_delta, grid_lo, grid_hi, G):
    """v3 Hopper wgmma backward — split-N + 2-stage cp.async pipeline."""
    if not (z.is_cuda and C.is_cuda and g_delta.is_cuda):
        raise RuntimeError("CUDA-only")
    z, C, g_delta = _coerce(z, C, g_delta)
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError("shape mismatch")
    scale = G / (grid_hi - grid_lo)
    return get_ext_v6().spline_kv_bwd_wgmma_v3_cuda(
        z, C, g_delta, float(grid_lo), float(scale)
    )


def get_ext_v5():
    """Hopper wgmma v2 — output-parallel grid, sm_90 only."""
    global _EXT_V5
    if _EXT_V5 is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        from torch.utils.cpp_extension import load
        _EXT_V5 = load(
            name="spline_kv_bwd_wgmma_v2_ext",
            sources=[str(_HERE / "spline_kv_bwd_wgmma_v2.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "-std=c++17",
                "--extended-lambda",
            ],
            verbose=False,
        )
    return _EXT_V5


def _coerce(z, C, g_delta):
    if z.dtype != torch.bfloat16:
        z = z.to(torch.bfloat16)
    if C.dtype != torch.bfloat16:
        C = C.to(torch.bfloat16)
    if g_delta.dtype != torch.bfloat16:
        g_delta = g_delta.to(torch.bfloat16)
    return z.contiguous(), C.contiguous(), g_delta.contiguous()


def spline_kv_bwd_cuda(z, C, g_delta, grid_lo, grid_hi, G):
    """v1 scalar SMEM atomicAdd (kept as correctness reference)."""
    if not (z.is_cuda and C.is_cuda and g_delta.is_cuda):
        raise RuntimeError("CUDA-only")
    z, C, g_delta = _coerce(z, C, g_delta)
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError("shape mismatch")
    scale = G / (grid_hi - grid_lo)
    dC = torch.zeros((H, L, R), device=z.device, dtype=torch.float32)
    dz = torch.zeros((N, H), device=z.device, dtype=torch.float32)
    get_ext_v1().spline_kv_bwd_cuda(
        z, C, g_delta, dC, dz, float(grid_lo), float(scale)
    )
    return dC, dz


def spline_kv_bwd_wmma_cuda(z, C, g_delta, grid_lo, grid_hi, G):
    """v2 wmma tensor-core densified matmul."""
    if not (z.is_cuda and C.is_cuda and g_delta.is_cuda):
        raise RuntimeError("CUDA-only")
    z, C, g_delta = _coerce(z, C, g_delta)
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError("shape mismatch")
    scale = G / (grid_hi - grid_lo)
    dC = torch.zeros((H, L, R), device=z.device, dtype=torch.float32)
    dz = torch.zeros((N, H), device=z.device, dtype=torch.float32)
    get_ext_v2().spline_kv_bwd_wmma_cuda(
        z, C, g_delta, dC, dz, float(grid_lo), float(scale)
    )
    return dC, dz


def spline_kv_bwd_wgmma_cuda(z, C, g_delta, grid_lo, grid_hi, G,
                             g_a=None, activation: str = "relu_sq",
                             fused_post: bool = False):
    """v4 Hopper-only: wgmma m64n32k16 + cp.async + no-SMEM-dC."""
    if not (z.is_cuda and C.is_cuda and g_delta.is_cuda):
        raise RuntimeError("CUDA-only")
    z, C, g_delta = _coerce(z, C, g_delta)
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError("shape mismatch")
    scale = G / (grid_hi - grid_lo)
    if fused_post:
        if g_a is None:
            raise ValueError("fused_post requires g_a")
        if activation not in {"relu_sq", "identity"}:
            raise ValueError(f"unsupported fused activation: {activation}")
        if g_a.dtype != torch.bfloat16:
            g_a = g_a.to(torch.bfloat16)
        g_a = g_a.contiguous()
        activation_id = 0 if activation == "relu_sq" else 2
        return get_ext_v4().spline_kv_bwd_wgmma_cuda_fused(
            z, C, g_delta, g_a, float(grid_lo), float(scale), activation_id
        )
    dC = torch.zeros((H, L, R), device=z.device, dtype=torch.float32)
    dz = torch.zeros((N, H), device=z.device, dtype=torch.float32)
    get_ext_v4().spline_kv_bwd_wgmma_cuda(
        z, C, g_delta, dC, dz, float(grid_lo), float(scale)
    )
    return dC, dz


def spline_kv_bwd_wgmma_v2_cuda(z, C, g_delta, grid_lo, grid_hi, G):
    """v5 Hopper-only: output-parallel grid, no dC atomic-add.

    Returns:
        dC : [H, L, R] bf16 (already cast inside kernel — no fp32 round-trip)
        dz : [N, H]    fp32 (caller must apply lambda_scale + dz_base separately
                              — there is no fused_post path in v2 yet)
    """
    if not (z.is_cuda and C.is_cuda and g_delta.is_cuda):
        raise RuntimeError("CUDA-only")
    z, C, g_delta = _coerce(z, C, g_delta)
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError("shape mismatch")
    scale = G / (grid_hi - grid_lo)
    return get_ext_v5().spline_kv_bwd_wgmma_v2_cuda(
        z, C, g_delta, float(grid_lo), float(scale)
    )


def get_ext_fwd():
    """Native CUDA forward (sm_80+, scalar atomic-add v4 port)."""
    global _EXT_FWD
    if _EXT_FWD is None:
        _EXT_FWD = _load("spline_kv_fwd_cuda_ext", "spline_kv_fwd.cu")
    return _EXT_FWD


def get_ext_fwd_v3():
    """v3 forward (Hopper-aligned, single-CTA-per-n-chunk, no atomic)."""
    global _EXT_FWD_V3
    if _EXT_FWD_V3 is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        from torch.utils.cpp_extension import load
        _EXT_FWD_V3 = load(
            name="spline_kv_fwd_v3_ext",
            sources=[str(_HERE / "spline_kv_fwd_v3.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "-std=c++17",
            ],
            verbose=False,
        )
    return _EXT_FWD_V3


def get_ext_fwd_v10():
    """v10 forward — dense-W wgmma (sm_90 only)."""
    global _EXT_FWD_V10
    if _EXT_FWD_V10 is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        from torch.utils.cpp_extension import load
        _EXT_FWD_V10 = load(
            name="spline_kv_fwd_v10_ext",
            sources=[str(_HERE / "spline_kv_fwd_v10.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "-std=c++17",
            ],
            verbose=False,
        )
    return _EXT_FWD_V10


def spline_kv_fwd_v10_cuda(z, C, grid_lo, grid_hi, G,
                            activation: str = "relu_sq",
                            lambda_scale: float = 1.0):
    """v10 forward — dense-W wgmma kernel (Hopper sm_90)."""
    if not (z.is_cuda and C.is_cuda):
        raise RuntimeError("CUDA-only")
    if z.dtype != torch.bfloat16:
        z = z.to(torch.bfloat16)
    if C.dtype != torch.bfloat16:
        C = C.to(torch.bfloat16)
    z = z.contiguous(); C = C.contiguous()
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError(f"shape mismatch: z={tuple(z.shape)} C={tuple(C.shape)} G={G}")
    if activation not in {"relu_sq", "identity"}:
        raise ValueError(f"unsupported activation: {activation}")
    activation_id = 0 if activation == "relu_sq" else 2
    scale = G / (grid_hi - grid_lo)
    return get_ext_fwd_v10().spline_kv_fwd_v10_cuda(
        z, C, float(grid_lo), float(scale), float(lambda_scale),
        int(activation_id),
    )


def get_ext_fwd_v11():
    """v11 forward — dense-W wgmma f32.f16.f16 (precision-fixed v10)."""
    global _EXT_FWD_V11
    if _EXT_FWD_V11 is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        from torch.utils.cpp_extension import load
        _EXT_FWD_V11 = load(
            name="spline_kv_fwd_v11_ext",
            sources=[str(_HERE / "spline_kv_fwd_v11.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode", "arch=compute_90a,code=sm_90a",
                "-std=c++17",
            ],
            verbose=False,
        )
    return _EXT_FWD_V11


def spline_kv_fwd_v11_cuda(z, C, grid_lo, grid_hi, G,
                            activation: str = "relu_sq",
                            lambda_scale: float = 1.0):
    """v11 forward — fp16 wgmma (B-coefficients keep ~3 more mantissa bits
    than v10's bf16 path; precision parity with triton/v1)."""
    if not (z.is_cuda and C.is_cuda):
        raise RuntimeError("CUDA-only")
    if z.dtype != torch.bfloat16:
        z = z.to(torch.bfloat16)
    # v11 specifically requires fp16 C (kernel uses f32.f16.f16 wgmma).
    # Cast bf16 -> fp16 here; this is exact for |C| < 65504 which is always
    # true for reasonable spline-coefficient magnitudes.
    if C.dtype != torch.float16:
        C = C.to(torch.float16)
    z = z.contiguous(); C = C.contiguous()
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError(f"shape mismatch: z={tuple(z.shape)} C={tuple(C.shape)} G={G}")
    if activation not in {"relu_sq", "identity"}:
        raise ValueError(f"unsupported activation: {activation}")
    activation_id = 0 if activation == "relu_sq" else 2
    scale = G / (grid_hi - grid_lo)
    return get_ext_fwd_v11().spline_kv_fwd_v11_cuda(
        z, C, float(grid_lo), float(scale), float(lambda_scale),
        int(activation_id),
    )


def spline_kv_fwd_v3_cuda(z, C, grid_lo, grid_hi, G,
                            activation: str = "relu_sq",
                            lambda_scale: float = 1.0):
    """v3 forward — Hopper-aligned, single-CTA-per-n-chunk, no atomic.

    Returns f [N, h+r] bf16, with f[:, :h] = activation(z) and
    f[:, h:h+r] = lambda * delta(z, C).
    """
    if not (z.is_cuda and C.is_cuda):
        raise RuntimeError("CUDA-only")
    if z.dtype != torch.bfloat16:
        z = z.to(torch.bfloat16)
    if C.dtype != torch.bfloat16:
        C = C.to(torch.bfloat16)
    z = z.contiguous(); C = C.contiguous()
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError(f"shape mismatch: z={tuple(z.shape)} C={tuple(C.shape)} G={G}")
    if activation not in {"relu_sq", "identity"}:
        raise ValueError(f"unsupported activation: {activation}")
    activation_id = 0 if activation == "relu_sq" else 2
    scale = G / (grid_hi - grid_lo)
    return get_ext_fwd_v3().spline_kv_fwd_v3_cuda(
        z, C, float(grid_lo), float(scale), float(lambda_scale),
        int(activation_id),
    )


def spline_kv_fwd_cuda(z, C, grid_lo, grid_hi, G,
                        activation: str = "relu_sq",
                        lambda_scale: float = 1.0):
    """Native CUDA forward — port of triton flash_spline_feature v4.

    Returns f [N, h+r] bf16, with f[:, :h] = activation(z) and
    f[:, h:h+r] = lambda * delta(z, C). Internally uses cudaMemsetAsync to
    init the fp32 atomic accumulator so the call is CUDA-Graph-safe.
    """
    if not (z.is_cuda and C.is_cuda):
        raise RuntimeError("CUDA-only")
    if z.dtype != torch.bfloat16:
        z = z.to(torch.bfloat16)
    if C.dtype != torch.bfloat16:
        C = C.to(torch.bfloat16)
    z = z.contiguous(); C = C.contiguous()
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError(f"shape mismatch: z={tuple(z.shape)} C={tuple(C.shape)} G={G}")
    if activation not in {"relu_sq", "identity"}:
        raise ValueError(f"unsupported activation: {activation}")
    activation_id = 0 if activation == "relu_sq" else 2
    scale = G / (grid_hi - grid_lo)
    return get_ext_fwd().spline_kv_fwd_cuda(
        z, C, float(grid_lo), float(scale), float(lambda_scale),
        int(activation_id),
    )


def spline_kv_fwd_fused_cuda(z, C, W_out, grid_lo, grid_hi, G,
                              activation: str = "relu_sq",
                              lambda_scale: float = 1.0):
    """Fused forward + W_out matmul.

    Returns y = a @ W_out_a^T + lambda * delta @ W_out_d^T  [N, d_out].
    Eliminates the [N, h+r] f tensor materialization.
    """
    if not (z.is_cuda and C.is_cuda and W_out.is_cuda):
        raise RuntimeError("CUDA-only")
    if z.dtype != torch.bfloat16:
        z = z.to(torch.bfloat16)
    if C.dtype != torch.bfloat16:
        C = C.to(torch.bfloat16)
    if W_out.dtype != torch.bfloat16:
        W_out = W_out.to(torch.bfloat16)
    z = z.contiguous(); C = C.contiguous(); W_out = W_out.contiguous()
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError(f"shape mismatch: z={tuple(z.shape)} C={tuple(C.shape)} G={G}")
    if W_out.shape[1] != H + R:
        raise ValueError(f"W_out cols {W_out.shape[1]} != H+R={H+R}")
    if activation not in {"relu_sq", "identity"}:
        raise ValueError(f"unsupported activation: {activation}")
    activation_id = 0 if activation == "relu_sq" else 2
    scale = G / (grid_hi - grid_lo)
    return get_ext_fwd().spline_kv_fwd_fused_cuda(
        z, C, W_out, float(grid_lo), float(scale), float(lambda_scale),
        int(activation_id),
    )


def spline_kv_bwd_hopper_cuda(z, C, g_delta, grid_lo, grid_hi, G):
    """v3 Hopper: cp.async + no-SMEM-dC + wmma."""
    if not (z.is_cuda and C.is_cuda and g_delta.is_cuda):
        raise RuntimeError("CUDA-only")
    z, C, g_delta = _coerce(z, C, g_delta)
    N, H = z.shape
    H_C, L, R = C.shape
    if H_C != H or L != G + 2:
        raise ValueError("shape mismatch")
    scale = G / (grid_hi - grid_lo)
    dC = torch.zeros((H, L, R), device=z.device, dtype=torch.float32)
    dz = torch.zeros((N, H), device=z.device, dtype=torch.float32)
    get_ext_v3().spline_kv_bwd_hopper_cuda(
        z, C, g_delta, dC, dz, float(grid_lo), float(scale)
    )
    return dC, dz


__all__ = [
    "spline_kv_bwd_cuda",
    "spline_kv_bwd_wmma_cuda",
    "spline_kv_bwd_hopper_cuda",
    "spline_kv_bwd_wgmma_cuda",
    "spline_kv_bwd_wgmma_v5_cuda",
    "spline_kv_bwd_v6_cuda",
    "spline_kv_fwd_cuda",
    "spline_kv_fwd_fused_cuda",
    "spline_kv_fwd_v3_cuda",
    "spline_kv_fwd_v10_cuda",
    "spline_kv_fwd_v11_cuda",
    "get_ext_v1",
    "get_ext_v2",
    "get_ext_v3",
    "get_ext_v4",
    "get_ext_v8",
    "get_ext_bwd_v6",
    "get_ext_fwd",
    "get_ext_fwd_v3",
    "get_ext_fwd_v10",
    "get_ext_fwd_v11",
]
