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


def spline_kv_bwd_wgmma_cuda(z, C, g_delta, grid_lo, grid_hi, G):
    """v4 Hopper-only: wgmma m64n32k16 + cp.async + no-SMEM-dC."""
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
    get_ext_v4().spline_kv_bwd_wgmma_cuda(
        z, C, g_delta, dC, dz, float(grid_lo), float(scale)
    )
    return dC, dz


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
    "get_ext_v1",
    "get_ext_v2",
    "get_ext_v3",
    "get_ext_v4",
]
