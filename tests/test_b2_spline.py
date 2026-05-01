"""Correctness tests for the B2 spline kernels and SimpleSplineMLP."""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="B2 tests require CUDA"
)


def _b2_reference(z: torch.Tensor, Q: torch.Tensor,
                   grid_lo: float, grid_hi: float, G: int) -> torch.Tensor:
    """PyTorch reference for the B2 spline activation (per-channel)."""
    scale = G / (grid_hi - grid_lo)
    u = ((z - grid_lo) * scale).clamp(0.0, float(G - 1))
    bin_idx = u.floor().to(torch.long)
    tau = (u - bin_idx.to(u.dtype)).clamp(0.0, 1.0)
    one_t = 1.0 - tau
    B0 = one_t * one_t * 0.5
    B1 = (1.0 + 2.0 * tau - 2.0 * tau * tau) * 0.5
    B2 = tau * tau * 0.5
    H = Q.shape[0]
    h_idx = torch.arange(H, device=z.device).expand_as(bin_idx)
    Q0 = Q[h_idx, bin_idx]
    Q1 = Q[h_idx, bin_idx + 1]
    Q2 = Q[h_idx, bin_idx + 2]
    return Q0 * B0 + Q1 * B1 + Q2 * B2


@cuda_required
def test_b2_forward_matches_reference():
    from sparsespline_ffn.kernels.triton_b2 import b2_forward
    torch.manual_seed(0)
    H, G = 64, 20
    L = G + 2
    Q = torch.randn(H, L, device="cuda", dtype=torch.float32) * 0.5
    z = torch.randn(256, H, device="cuda", dtype=torch.float32)
    y_ref = _b2_reference(z, Q, -3.0, 3.0, G)
    y_k = b2_forward(z, Q, -3.0, 3.0, G)
    rel = ((y_k - y_ref).norm() / y_ref.norm()).item()
    assert rel < 1e-6, f"fwd fp32 rel={rel:.3e}"


@cuda_required
def test_b2_forward_matches_reference_bf16():
    from sparsespline_ffn.kernels.triton_b2 import b2_forward
    torch.manual_seed(1)
    H, G = 32, 20
    L = G + 2
    Q = torch.randn(H, L, device="cuda", dtype=torch.bfloat16) * 0.5
    z = torch.randn(128, H, device="cuda", dtype=torch.bfloat16)
    y_ref = _b2_reference(z.float(), Q.float(), -3.0, 3.0, G)
    y_k = b2_forward(z, Q, -3.0, 3.0, G)
    rel = ((y_k.float() - y_ref).norm() / y_ref.norm()).item()
    assert rel < 5e-3, f"fwd bf16 rel={rel:.3e}"


@cuda_required
def test_b2_backward_matches_reference():
    from sparsespline_ffn.kernels.b2_autograd import B2SplineActivation
    torch.manual_seed(2)
    H, G = 64, 20
    L = G + 2
    Q_data = torch.randn(H, L, device="cuda", dtype=torch.float32) * 0.5
    z_data = torch.randn(256, H, device="cuda", dtype=torch.float32)
    dy = torch.randn(256, H, device="cuda", dtype=torch.float32)

    # Reference: PyTorch autograd through manual ref
    Q_ref = Q_data.clone().requires_grad_(True)
    z_ref = z_data.clone().requires_grad_(True)
    y_ref = _b2_reference(z_ref, Q_ref, -3.0, 3.0, G)
    y_ref.backward(dy)

    # Kernel: B2SplineActivation
    Q = Q_data.clone().requires_grad_(True)
    z = z_data.clone().requires_grad_(True)
    y_k = B2SplineActivation.apply(z, Q, -3.0, 3.0, G)
    y_k.backward(dy)

    rel_dz = ((z.grad - z_ref.grad).norm() / z_ref.grad.norm()).item()
    rel_dQ = ((Q.grad - Q_ref.grad).norm() / Q_ref.grad.norm()).item()
    assert rel_dz < 1e-5, f"dz rel={rel_dz:.3e}"
    assert rel_dQ < 1e-5, f"dQ rel={rel_dQ:.3e}"


@cuda_required
def test_simple_spline_mlp_kernel_matches_reference():
    from sparsespline_ffn.simple_spline_mlp import (
        SimpleSplineConfig, SimpleSplineMLP,
    )
    torch.manual_seed(3)
    cfg_ref = SimpleSplineConfig(d=128, h_ratio=0.5, G=20, use_kernel=False)
    cfg_kern = SimpleSplineConfig(d=128, h_ratio=0.5, G=20, use_kernel=True)
    torch.manual_seed(3); ref = SimpleSplineMLP(cfg_ref).cuda().float()
    torch.manual_seed(3); kern = SimpleSplineMLP(cfg_kern).cuda().float()
    for p_r, p_k in zip(ref.parameters(), kern.parameters(), strict=True):
        p_k.data.copy_(p_r.data)
    # Perturb W_d (zero by default) so we get non-trivial gradients
    with torch.no_grad():
        ref.W_d.weight.normal_(0.0, 0.1)
        kern.W_d.weight.copy_(ref.W_d.weight)
    x = torch.randn(2, 64, 128, device="cuda", requires_grad=True)

    y_r = ref(x); y_k = kern(x)
    assert ((y_r - y_k).norm() / y_r.norm()).item() < 1e-6

    y_r.pow(2).sum().backward()
    g_ref = {n: p.grad.clone() for n, p in ref.named_parameters()}
    xg_ref = x.grad.clone(); x.grad = None
    for p in ref.parameters(): p.grad = None

    y_k.pow(2).sum().backward()
    rel_x = ((xg_ref - x.grad).norm() / xg_ref.norm()).item()
    assert rel_x < 1e-5, f"x.grad rel={rel_x:.3e}"
    for n, p in kern.named_parameters():
        rel = ((g_ref[n] - p.grad).norm() / (g_ref[n].norm() + 1e-12)).item()
        assert rel < 1e-5, f"param {n!r} grad rel={rel:.3e}"


@cuda_required
def test_simple_spline_mlp_can_train_via_sgd():
    """End-to-end smoke: SimpleSplineMLP fits a per-token nonlinear target."""
    import torch.optim as optim
    from sparsespline_ffn.simple_spline_mlp import (
        SimpleSplineConfig, SimpleSplineMLP,
    )
    torch.manual_seed(4)
    d = 16
    cfg = SimpleSplineConfig(d=d, h_ratio=0.5, G=12, use_kernel=True)
    m = SimpleSplineMLP(cfg).cuda().float()
    # Perturb so output isn't zero
    with torch.no_grad():
        m.W_d.weight.normal_(0.0, 0.1)
    x = torch.randn(64, d, device="cuda")
    target = torch.zeros(64, d, device="cuda")
    target[:, 0] = torch.sin(x[:, 0])
    target[:, 1] = 0.3 * x[:, 1] ** 2

    opt = optim.SGD(m.parameters(), lr=0.05)
    init_mse = float("inf")
    for step in range(150):
        opt.zero_grad()
        y = m(x)
        mse = (y - target).pow(2).mean()
        if step == 0:
            init_mse = mse.item()
        if step == 149:
            final_mse = mse.item()
        mse.backward()
        opt.step()
    assert final_mse < init_mse / 3.0, (
        f"failed to learn: init {init_mse:.4f} -> final {final_mse:.4f}"
    )
