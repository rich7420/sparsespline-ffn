"""Correctness tests for FlashSplineFeature v4 (h-split + atomic_add)."""
from __future__ import annotations

import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA-only kernel test")


@cuda_only
def test_v4_matches_v1_fp32():
    """v4 (h-split + atomic) must produce same delta as v1 in fp32."""
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward,
    )
    torch.manual_seed(0)
    N, h, r, G = 64, 256, 32, 16
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.float32)
    C = (torch.randn(h, L, r, device="cuda", dtype=torch.float32) * 0.1)

    f_v1 = flash_spline_feature_forward(z, C, grid_lo=-3, grid_hi=3, G=G,
                                         version="v1")
    f_v4 = flash_spline_feature_forward(z, C, grid_lo=-3, grid_hi=3, G=G,
                                         version="v4")
    # atomic_add reduction order is non-deterministic but should be close
    # to fp32 noise from sum-order differences.
    max_diff = (f_v1 - f_v4).abs().max().item()
    rel = (f_v1 - f_v4).pow(2).mean().sqrt().item() / f_v1.pow(2).mean().sqrt().item()
    assert rel < 1e-4, f"v4 vs v1 rel rms = {rel:.3e}, max abs = {max_diff:.3e}"


@cuda_only
def test_v4_C_zero_gives_zero_delta():
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward,
    )
    torch.manual_seed(0)
    N, h, r, G = 64, 256, 32, 16
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(h, L, r, device="cuda", dtype=torch.bfloat16)
    f = flash_spline_feature_forward(z, C, grid_lo=-3, grid_hi=3, G=G,
                                       version="v4")
    delta = f[:, h:]
    assert delta.abs().max().item() == 0.0


@cuda_only
def test_v4_realistic_shape():
    """h=768, r=64, L=22 — production shape."""
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward,
    )
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference,
    )
    torch.manual_seed(0)
    N, h, r, G = 256, 768, 64, 20
    L = G + 2
    # fp32 inputs to suppress bf16 reference noise
    z = torch.randn(N, h, device="cuda", dtype=torch.float32)
    C = (torch.randn(h, L, r, device="cuda", dtype=torch.float32) * 0.1)

    f_ref = flash_spline_feature_reference(z, C, grid_lo=-3, grid_hi=3, G=G)
    f_v4 = flash_spline_feature_forward(z, C, grid_lo=-3, grid_hi=3, G=G,
                                         version="v4")
    rel = (f_v4 - f_ref).pow(2).mean().sqrt().item() / f_ref.pow(2).mean().sqrt().item()
    assert rel < 1e-4, f"v4 vs fp32 ref rel rms = {rel:.3e}"


@cuda_only
def test_v4_repeated_calls_consistent():
    """atomic_add reduction may produce slightly different results across
    calls due to non-deterministic atomic ordering.  Check that variance
    is still within bf16 noise floor."""
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward,
    )
    torch.manual_seed(0)
    N, h, r, G = 128, 512, 64, 18
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
    C = (torch.randn(h, L, r, device="cuda", dtype=torch.bfloat16) * 0.1)

    f0 = flash_spline_feature_forward(z, C, grid_lo=-3, grid_hi=3, G=G, version="v4")
    f1 = flash_spline_feature_forward(z, C, grid_lo=-3, grid_hi=3, G=G, version="v4")
    rel = (f0 - f1).pow(2).mean().sqrt().item() / f0.pow(2).mean().sqrt().item()
    # atomic ordering means small variability across calls.
    assert rel < 1e-3, f"v4 across-call variance = {rel:.3e}"
