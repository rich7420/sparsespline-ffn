from __future__ import annotations

import pytest
import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN
from sparsespline_ffn.tucker_init import hosvd_warmstart_from_dense


def make_small(**overrides) -> FullMixTuckerFFN:
    cfg_kwargs = dict(d=8, m=8, R_o=4, R_i=4, R_b=2, G=5)
    cfg_kwargs.update(overrides)
    return FullMixTuckerFFN(FullMixTuckerConfig(**cfg_kwargs)).to(torch.float32)


def test_shape_preserved_for_2d_and_3d_inputs() -> None:
    ffn = make_small()
    x2 = torch.randn(3, 8)
    x3 = torch.randn(2, 3, 8)

    assert ffn(x2).shape == x2.shape
    assert ffn(x3).shape == x3.shape


def test_forward_backward_is_finite() -> None:
    ffn = make_small()
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = ffn(x)
    loss = y.square().mean()
    loss.backward()

    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for param in ffn.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_direct_mixer_requires_m_equals_d() -> None:
    with pytest.raises(ValueError, match="requires m == d"):
        FullMixTuckerFFN(
            FullMixTuckerConfig(d=8, m=16, R_o=4, R_i=4, R_b=2, use_mixer=False)
        )


def test_config_rejects_compressive_mixer() -> None:
    with pytest.raises(ValueError, match="input-side bottleneck"):
        FullMixTuckerConfig(d=16, m=8, R_o=4, R_i=4, R_b=2)


def test_dense_reconstruction_matches_reference_sum() -> None:
    torch.manual_seed(0)
    ffn = make_small(use_mixer=False)
    x = torch.randn(3, 8)
    y_ref = ffn(x)

    z = x
    bin_idx, t = ffn._bin_and_frac(z)
    B = torch.zeros(x.shape[0], ffn.cfg.m, ffn._L)
    B.scatter_add_(2, bin_idx.unsqueeze(-1), (1.0 - t).unsqueeze(-1))
    B.scatter_add_(2, (bin_idx + 1).unsqueeze(-1), t.unsqueeze(-1))

    W = ffn.reconstruct_dense_W()
    y_dense = torch.einsum("kji,nji->nk", W, B) * ffn.gamma

    assert torch.allclose(y_ref, y_dense, atol=1e-5, rtol=1e-5)


def test_output_subspace_dim_is_bounded_by_rank() -> None:
    ffn = make_small(R_o=3)
    assert ffn.output_subspace_dim() <= 3


def test_hosvd_warmstart_shapes() -> None:
    W = torch.randn(7, 8, 6)
    U, V, core, Q = hosvd_warmstart_from_dense(W, R_o=3, R_i=4, R_b=2)

    assert U.shape == (7, 3)
    assert V.shape == (8, 4)
    assert core.shape == (3, 4, 2)
    assert Q.shape == (6, 2)
