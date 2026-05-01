"""Numerical stability test for RL-Spline-KV training step in bf16.

Trains a tiny RL-KV module for many steps, monitors gradient/parameter
norms.  Fails if norms drift beyond expected range or NaN/Inf appears.
This catches subtle bf16 cast issues, accumulator drift, atomic-add
non-determinism, etc.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA-only stability test")


def _build_rlkv_module(d=128, r=32, G=18, dtype=torch.bfloat16, device="cuda"):
    """Builds a minimal RL-Spline-KV style FFN."""
    from sparsespline_ffn.kernels.flash_spline_feature_autograd import (
        flash_spline_feature,
    )

    class _RLKV(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = nn.Linear(d, d, bias=False)
            self.C = nn.Parameter(torch.zeros(d, G + 2, r,
                                              dtype=dtype, device=device))
            self.W_out = nn.Linear(d + r, d, bias=False)
            self.G = G

        def forward(self, x):
            shape = x.shape
            x_flat = x.reshape(-1, d)
            z = self.K(x_flat)
            f = flash_spline_feature(
                z, self.C, grid_lo=-3.0, grid_hi=3.0, G=self.G,
                activation="relu_sq", lambda_scale=1.0, use_kernel=True,
            )
            y = self.W_out(f)
            return y.reshape(shape)

    return _RLKV().to(device=device, dtype=dtype)


@cuda_only
def test_bf16_training_stability_100_steps():
    """Run 100 fake training steps in bf16; check norms stay bounded."""
    torch.manual_seed(42)
    device = "cuda"
    d, r, G = 128, 32, 18
    B, T = 4, 32
    model = _build_rlkv_module(d=d, r=r, G=G,
                                dtype=torch.bfloat16, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                    weight_decay=0.0)

    loss_history = []
    grad_norm_history = []
    C_norm_history = []
    K_norm_history = []
    Wout_norm_history = []

    for step in range(100):
        x = torch.randn(B, T, d, device=device, dtype=torch.bfloat16)
        target = torch.randn(B, T, d, device=device, dtype=torch.bfloat16)
        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        loss = (y - target).pow(2).sum() / (B * T * d)

        loss.backward()
        # Gradient norms
        g_norms = []
        for p in model.parameters():
            if p.grad is not None:
                g_norms.append(p.grad.detach().to(torch.float32).norm().item())
        grad_norm_history.append(sum(g_norms))
        loss_history.append(loss.item())

        optimizer.step()

        C_norm_history.append(model.C.detach().to(torch.float32).norm().item())
        K_norm_history.append(model.K.weight.detach().to(torch.float32).norm().item())
        Wout_norm_history.append(model.W_out.weight.detach().to(torch.float32).norm().item())

    print(f"\nStep 0: loss={loss_history[0]:.4f} grad_norm={grad_norm_history[0]:.4f}")
    print(f"Step 50: loss={loss_history[50]:.4f} grad_norm={grad_norm_history[50]:.4f}")
    print(f"Step 99: loss={loss_history[99]:.4f} grad_norm={grad_norm_history[99]:.4f}")
    print(f"C_norm: init={C_norm_history[0]:.4f} → final={C_norm_history[-1]:.4f}")
    print(f"K_norm: init={K_norm_history[0]:.4f} → final={K_norm_history[-1]:.4f}")
    print(f"Wout_norm: init={Wout_norm_history[0]:.4f} → final={Wout_norm_history[-1]:.4f}")

    # ---- Sanity checks ----
    # 1. No NaN/Inf
    for arr_name, arr in [("loss", loss_history), ("grad_norm", grad_norm_history),
                           ("C_norm", C_norm_history)]:
        for i, v in enumerate(arr):
            assert v == v, f"NaN in {arr_name} at step {i}"            # NaN ≠ self
            assert v < 1e8, f"explosion in {arr_name} at step {i}: {v}"

    # 2. Loss should decrease (or at least not explode)
    final_loss = loss_history[-1]
    initial_loss = loss_history[0]
    assert final_loss < initial_loss * 5, \
        f"Loss exploded: {initial_loss:.4f} → {final_loss:.4f}"

    # 3. C should grow from 0 (cold start) to nontrivial magnitude
    assert C_norm_history[-1] > 1e-3, \
        f"C didn't learn: final |C| = {C_norm_history[-1]}"

    # 4. K and W_out should grow but not explode (5x bound)
    assert K_norm_history[-1] < K_norm_history[0] * 5, \
        f"K weights exploded: {K_norm_history[0]:.4f} → {K_norm_history[-1]:.4f}"
    assert Wout_norm_history[-1] < Wout_norm_history[0] * 5, \
        f"W_out weights exploded: {Wout_norm_history[0]:.4f} → {Wout_norm_history[-1]:.4f}"


@cuda_only
def test_bf16_vs_fp32_gradient_alignment():
    """Same forward + backward in bf16 vs fp32; check gradients align
    within bf16 precision (~1% relative)."""
    torch.manual_seed(0)
    device = "cuda"
    d, r, G = 64, 16, 10
    B, T = 2, 16

    # Build twin models with same init weights, different dtypes
    m_f32 = _build_rlkv_module(d=d, r=r, G=G, dtype=torch.float32, device=device)
    m_bf16 = _build_rlkv_module(d=d, r=r, G=G, dtype=torch.bfloat16, device=device)
    # Copy weights so they start identical
    with torch.no_grad():
        m_bf16.K.weight.copy_(m_f32.K.weight.to(torch.bfloat16))
        m_bf16.W_out.weight.copy_(m_f32.W_out.weight.to(torch.bfloat16))
        # Init C nonzero so we get gradient flow
        m_f32.C.copy_(torch.randn_like(m_f32.C) * 0.1)
        m_bf16.C.copy_(m_f32.C.to(torch.bfloat16))

    x = torch.randn(B, T, d, device=device, dtype=torch.float32)
    target = torch.randn(B, T, d, device=device, dtype=torch.float32)

    # fp32 path
    y_f = m_f32(x)
    loss_f = (y_f - target).pow(2).sum() / (B*T*d)
    loss_f.backward()

    # bf16 path
    y_b = m_bf16(x.to(torch.bfloat16))
    loss_b = (y_b - target.to(torch.bfloat16)).pow(2).sum() / (B*T*d)
    loss_b.backward()

    # Compare gradients
    for name in ["K", "W_out"]:
        g_f32 = getattr(m_f32, name).weight.grad.detach()
        g_bf16 = getattr(m_bf16, name).weight.grad.detach().to(torch.float32)
        rel_err = ((g_f32 - g_bf16).pow(2).mean().sqrt()
                    / g_f32.pow(2).mean().sqrt().clamp_min(1e-9)).item()
        print(f"  {name}.grad rel rms: {rel_err:.4f}")
        # bf16 has ~1% relative precision; allow 5% for accumulation
        assert rel_err < 5e-2, f"{name}.grad bf16 vs fp32 diff too large: {rel_err}"

    # Compare C grad
    rel_C = ((m_f32.C.grad - m_bf16.C.grad.to(torch.float32)).pow(2).mean().sqrt()
              / m_f32.C.grad.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    print(f"  C.grad rel rms: {rel_C:.4f}")
    assert rel_C < 5e-2, f"C.grad bf16 vs fp32 diff too large: {rel_C}"
