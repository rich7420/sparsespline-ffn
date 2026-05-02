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
_EXT_FWD = None  # native CUDA forward (sm_80+)


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


def get_ext_fwd():
    """Native CUDA forward (sm_80+, scalar atomic-add v4 port)."""
    global _EXT_FWD
    if _EXT_FWD is None:
        _EXT_FWD = _load("spline_kv_fwd_cuda_ext", "spline_kv_fwd.cu")
    return _EXT_FWD


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
    "spline_kv_fwd_cuda",
    "spline_kv_fwd_fused_cuda",
    "get_ext_v1",
    "get_ext_v2",
    "get_ext_v3",
    "get_ext_v4",
    "get_ext_fwd",
]
