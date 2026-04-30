"""Correctness tests for the Triton dQ kernel.

PyTorch reference (the oracle) for backward dQ:
    dQ = zeros(L, R_b)
    dQ.index_add_(0, bin_idx, (1-t).unsqueeze(-1) * dbeta)
    dQ.index_add_(0, bin_idx + 1,    t.unsqueeze(-1) * dbeta)

The Triton kernel ``b1_backward_dq`` must match this within fp32 1e-5
or bf16 5e-3 relative error.

These tests are CUDA-only.
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="kernel tests require CUDA"
)


def _torch_reference_dq(
    bin_idx: torch.Tensor, t: torch.Tensor, dbeta: torch.Tensor, L: int
) -> torch.Tensor:
    """Eager PyTorch reference, fp32 accumulation."""
    bin_flat = bin_idx.reshape(-1).long()
    t_flat = t.reshape(-1).float()
    db_flat = dbeta.reshape(-1, dbeta.shape[-1]).float()
    dQ = torch.zeros((L, db_flat.shape[-1]), dtype=torch.float32, device=db_flat.device)
    dQ.index_add_(0, bin_flat, (1.0 - t_flat).unsqueeze(-1) * db_flat)
    dQ.index_add_(0, bin_flat + 1,        t_flat.unsqueeze(-1) * db_flat)
    return dQ


@cuda_required
def test_kernel_imports():
    from sparsespline_ffn.kernels import HAS_TRITON, b1_backward_dq  # noqa: F401
    assert HAS_TRITON


@cuda_required
def test_single_bin_matches_index_add():
    """Smallest non-trivial: every token lands in bin=3, R_b=4."""
    from sparsespline_ffn.kernels import b1_backward_dq

    torch.manual_seed(0)
    L, R_b, N = 8, 4, 32
    bin_idx = torch.full((N,), 3, dtype=torch.int64, device="cuda")
    t = torch.rand(N, dtype=torch.float32, device="cuda")
    dbeta = torch.randn(N, R_b, dtype=torch.float32, device="cuda")

    dQ_ref = _torch_reference_dq(bin_idx, t, dbeta, L)
    dQ_triton = b1_backward_dq(bin_idx, t, dbeta, L=L)

    rel = (dQ_triton - dQ_ref).norm() / (dQ_ref.norm() + 1e-12)
    assert rel < 1e-5, f"single-bin mismatch rel={rel.item():.3e}"


@cuda_required
def test_random_fp32_matches_reference():
    """Production-shape random fp32 test."""
    from sparsespline_ffn.kernels import b1_backward_dq

    torch.manual_seed(1)
    L, R_b = 21, 16  # G=20, R_b=16
    N, m = 1024, 768
    bin_idx = torch.randint(0, L - 1, (N, m), dtype=torch.int64, device="cuda")
    t = torch.rand(N, m, dtype=torch.float32, device="cuda")
    dbeta = torch.randn(N, m, R_b, dtype=torch.float32, device="cuda")

    dQ_ref = _torch_reference_dq(bin_idx, t, dbeta, L)
    dQ_triton = b1_backward_dq(bin_idx, t, dbeta, L=L)

    rel = (dQ_triton - dQ_ref).norm() / (dQ_ref.norm() + 1e-12)
    assert rel < 1e-5, f"fp32 rel={rel.item():.3e}"


@cuda_required
def test_random_bf16_matches_fp32_reference():
    """bf16 inputs should match fp32 reference within 5e-3."""
    from sparsespline_ffn.kernels import b1_backward_dq

    torch.manual_seed(2)
    L, R_b = 21, 16
    N, m = 512, 768
    bin_idx = torch.randint(0, L - 1, (N, m), dtype=torch.int64, device="cuda")
    t_bf = torch.rand(N, m, dtype=torch.bfloat16, device="cuda")
    db_bf = torch.randn(N, m, R_b, dtype=torch.bfloat16, device="cuda")

    # Reference uses fp32 promoted inputs (matches kernel's internal fp32 accum)
    dQ_ref = _torch_reference_dq(bin_idx, t_bf, db_bf, L)
    dQ_triton = b1_backward_dq(bin_idx, t_bf, db_bf, L=L)

    rel = (dQ_triton - dQ_ref).norm() / (dQ_ref.norm() + 1e-12)
    assert rel < 5e-3, f"bf16 rel={rel.item():.3e}"


@cuda_required
def test_worst_case_all_same_bin():
    """Pathological: every (n,j) lands in the same bin -> max atomic contention.

    This is the hardest case for atomic_add throughput; correctness must
    still match.
    """
    from sparsespline_ffn.kernels import b1_backward_dq

    torch.manual_seed(3)
    L, R_b = 21, 16
    N, m = 1024, 256
    target_bin = 7
    bin_idx = torch.full((N, m), target_bin, dtype=torch.int64, device="cuda")
    t = torch.rand(N, m, dtype=torch.float32, device="cuda")
    dbeta = torch.randn(N, m, R_b, dtype=torch.float32, device="cuda")

    dQ_ref = _torch_reference_dq(bin_idx, t, dbeta, L)
    dQ_triton = b1_backward_dq(bin_idx, t, dbeta, L=L)

    rel = (dQ_triton - dQ_ref).norm() / (dQ_ref.norm() + 1e-12)
    assert rel < 1e-5, f"all-same-bin rel={rel.item():.3e}"

    # Verify only rows {target_bin, target_bin+1} are nonzero
    other_rows = torch.cat([dQ_triton[:target_bin], dQ_triton[target_bin + 2:]], dim=0)
    assert other_rows.abs().max() < 1e-6


@cuda_required
def test_skewed_bin_distribution():
    """80% of tokens in 3 central bins, 20% uniform — typical LM-like shape."""
    from sparsespline_ffn.kernels import b1_backward_dq

    torch.manual_seed(4)
    L, R_b = 21, 16
    N, m = 2048, 512
    E = N * m

    central = torch.randint(9, 12, (int(0.8 * E),), dtype=torch.int64, device="cuda")
    rest = torch.randint(0, L - 1, (E - central.numel(),), dtype=torch.int64, device="cuda")
    bin_idx = torch.cat([central, rest])
    bin_idx = bin_idx[torch.randperm(E, device="cuda")].view(N, m)
    t = torch.rand(N, m, dtype=torch.float32, device="cuda")
    dbeta = torch.randn(N, m, R_b, dtype=torch.float32, device="cuda")

    dQ_ref = _torch_reference_dq(bin_idx, t, dbeta, L)
    dQ_triton = b1_backward_dq(bin_idx, t, dbeta, L=L)

    rel = (dQ_triton - dQ_ref).norm() / (dQ_ref.norm() + 1e-12)
    assert rel < 1e-5, f"skewed rel={rel.item():.3e}"


@cuda_required
def test_non_contiguous_dbeta():
    """dbeta from a transpose / slice is not contiguous on its inner dim.

    Wrapper must handle by calling .contiguous() before launch.
    """
    from sparsespline_ffn.kernels import b1_backward_dq

    torch.manual_seed(5)
    L, R_b = 21, 16
    N, m = 256, 256

    bin_idx = torch.randint(0, L - 1, (N, m), dtype=torch.int64, device="cuda")
    t = torch.rand(N, m, dtype=torch.float32, device="cuda")
    # Build a non-contiguous dbeta by transposing the channel dim into place
    raw = torch.randn(R_b, N, m, dtype=torch.float32, device="cuda")
    dbeta = raw.permute(1, 2, 0)  # (N, m, R_b) but non-contig on last dim
    assert dbeta.stride(-1) != 1

    dQ_ref = _torch_reference_dq(bin_idx, t, dbeta, L)
    dQ_triton = b1_backward_dq(bin_idx, t, dbeta, L=L)

    rel = (dQ_triton - dQ_ref).norm() / (dQ_ref.norm() + 1e-12)
    assert rel < 1e-5, f"non-contig rel={rel.item():.3e}"


@cuda_required
@pytest.mark.parametrize(("N", "m", "R_b", "L"), [
    (1, 1, 1, 2),     # smallest legal shape
    (1, 1, 16, 21),   # tiny token count
    (8192, 1, 4, 5),  # huge N, small everything else
    (4, 64, 32, 41),  # large R_b, smaller other dims
])
def test_shape_edges(N, m, R_b, L):
    from sparsespline_ffn.kernels import b1_backward_dq

    torch.manual_seed(N + m + R_b + L)
    bin_idx = torch.randint(0, L - 1, (N, m), dtype=torch.int64, device="cuda")
    t = torch.rand(N, m, dtype=torch.float32, device="cuda")
    dbeta = torch.randn(N, m, R_b, dtype=torch.float32, device="cuda")

    dQ_ref = _torch_reference_dq(bin_idx, t, dbeta, L)
    dQ_triton = b1_backward_dq(bin_idx, t, dbeta, L=L)

    rel = (dQ_triton - dQ_ref).norm() / (dQ_ref.norm() + 1e-12)
    assert rel < 1e-5, f"shape ({N},{m},{R_b},{L}) rel={rel.item():.3e}"
