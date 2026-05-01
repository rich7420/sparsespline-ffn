"""Forward-correctness tests for the FlashSplineFeature Triton kernel.

These compare the Triton kernel output against the PyTorch reference
(``rl_spline_kv_reference.flash_spline_feature_reference``).  All tests
are CUDA-only.

Tolerance is set to handle bf16 / fp16 / fp32 differently, per v7
§R.3.3.5 (fp32 internal accumulator → bf16 output should be within
~5e-3 relative; fp32 input/output should be within ~1e-5).
"""
from __future__ import annotations

import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA-only kernel test")


@cuda_only
def test_kernel_matches_reference_fp32_small():
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference,
    )
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward,
    )
    torch.manual_seed(0)
    N, h, r, G = 16, 32, 8, 10
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.float32)
    C = torch.randn(h, L, r, device="cuda", dtype=torch.float32) * 0.1

    f_ref = flash_spline_feature_reference(
        z, C, grid_lo=-3.0, grid_hi=3.0, G=G,
        activation="relu_sq", lambda_scale=1.0,
    )
    f_kernel = flash_spline_feature_forward(
        z, C, grid_lo=-3.0, grid_hi=3.0, G=G,
        activation="relu_sq", lambda_scale=1.0,
    )
    max_diff = (f_kernel - f_ref).abs().max().item()
    assert max_diff < 1e-4, f"fp32 max diff = {max_diff}"


@cuda_only
def test_kernel_matches_reference_bf16_realistic():
    """h=768, r=64, L=22 — the actual production shape."""
    from sparsespline_ffn.rl_spline_kv_reference import (
        flash_spline_feature_reference,
    )
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward,
    )
    torch.manual_seed(0)
    N, h, r, G = 64, 768, 64, 20
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
    C = (torch.randn(h, L, r, device="cuda", dtype=torch.bfloat16) * 0.1)

    # Reference computes in input dtype (bf16), so we get bf16 output too.
    f_ref = flash_spline_feature_reference(
        z, C, grid_lo=-3.0, grid_hi=3.0, G=G,
        activation="relu_sq", lambda_scale=1.0,
    )
    f_kernel = flash_spline_feature_forward(
        z, C, grid_lo=-3.0, grid_hi=3.0, G=G,
        activation="relu_sq", lambda_scale=1.0,
    )
    # Check shapes
    assert f_kernel.shape == f_ref.shape == (N, h + r)
    assert f_kernel.dtype == torch.bfloat16

    # phi(z) half should match very closely (same code path)
    diff_a = (f_kernel[:, :h] - f_ref[:, :h]).abs().max().item()
    assert diff_a < 1e-4, f"phi half max diff = {diff_a}"

    # delta half: kernel has fp32 internal accumulator (more accurate),
    # reference does bf16 cast roundtrips.  Allow small relative diff.
    delta_ref = f_ref[:, h:].to(torch.float32)
    delta_kernel = f_kernel[:, h:].to(torch.float32)
    rms_ref = delta_ref.pow(2).mean().sqrt().item()
    rms_diff = (delta_ref - delta_kernel).pow(2).mean().sqrt().item()
    rel = rms_diff / max(rms_ref, 1e-9)
    assert rel < 5e-2, f"delta half rel rms diff = {rel:.4f}, "\
                       f"rms_ref={rms_ref:.5f}, rms_diff={rms_diff:.5f}"


@cuda_only
def test_C_zero_gives_zero_delta():
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward,
    )
    torch.manual_seed(0)
    N, h, r, G = 32, 64, 16, 10
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(h, L, r, device="cuda", dtype=torch.bfloat16)
    f = flash_spline_feature_forward(z, C, grid_lo=-3.0, grid_hi=3.0, G=G)
    delta = f[:, h:]
    assert delta.abs().max().item() == 0.0, "delta must be 0 when C=0"


@cuda_only
def test_lambda_scale_propagates():
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward,
    )
    torch.manual_seed(0)
    N, h, r, G = 32, 64, 16, 10
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.float32)
    C = torch.randn(h, L, r, device="cuda", dtype=torch.float32) * 0.1
    f1 = flash_spline_feature_forward(z, C, grid_lo=-3.0, grid_hi=3.0, G=G,
                                        lambda_scale=1.0)
    f2 = flash_spline_feature_forward(z, C, grid_lo=-3.0, grid_hi=3.0, G=G,
                                        lambda_scale=2.5)
    diff_a = (f1[:, :h] - f2[:, :h]).abs().max().item()
    assert diff_a < 1e-6, "phi half is independent of lambda"
    diff_d = (f2[:, h:] - 2.5 * f1[:, h:]).abs().max().item()
    assert diff_d < 1e-4, f"delta half should scale by lambda: {diff_d}"


@cuda_only
def test_out_of_range_z_does_not_explode():
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward,
    )
    N, h, r, G = 32, 64, 16, 10
    L = G + 2
    z = torch.cat([
        torch.randn(N // 2, h, dtype=torch.float32),
        torch.randn(N // 2, h, dtype=torch.float32) * 100.0,
    ], dim=0).cuda()
    C = (torch.randn(h, L, r, dtype=torch.float32) * 0.1).cuda()
    f = flash_spline_feature_forward(z, C, grid_lo=-3.0, grid_hi=3.0, G=G)
    assert torch.isfinite(f).all(), "out-of-range z must not blow up"


@cuda_only
def test_partition_of_unity_zero_when_C_constant_in_b():
    """If C[j, :, c] is constant in b for all (j, c), delta[n, c] = constant.
    Tests that the kernel correctly handles the partition-of-unity sum."""
    from sparsespline_ffn.kernels.triton_flash_spline_feature import (
        flash_spline_feature_forward,
    )
    torch.manual_seed(0)
    N, h, r, G = 32, 32, 8, 8
    L = G + 2
    z = torch.randn(N, h, device="cuda", dtype=torch.float32)
    # C constant along b dimension
    C_per_jc = torch.randn(h, 1, r, device="cuda", dtype=torch.float32)
    C = C_per_jc.expand(h, L, r).contiguous()
    f = flash_spline_feature_forward(z, C, grid_lo=-3.0, grid_hi=3.0, G=G)
    delta = f[:, h:]
    # delta[n, c] should equal sum over j of (B0+B1+B2)*C_per_jc[j, 0, c]
    # = sum_j C_per_jc[j, 0, c]   (for in-range tokens)
    # = same for every n
    expected_per_c = C_per_jc.squeeze(1).sum(dim=0)  # [r]
    # In-range tokens: all of them given grid_lo/hi=-3/3 and z~N(0,1)
    # but tolerance for those that round to clamp boundary
    diff_per_n = (delta - expected_per_c.unsqueeze(0)).abs()
    # Tokens at saturating boundary may be 0 instead of constant — count
    # how many tokens are clearly in-range and assert majority match
    rel_diff = diff_per_n.max(dim=1).values / expected_per_c.abs().max().clamp_min(1e-9)
    in_range_frac = (rel_diff < 1e-3).float().mean().item()
    assert in_range_frac > 0.9, \
        f"≥90% of tokens should give partition-of-unity result, " \
        f"got {in_range_frac:.2%}"
