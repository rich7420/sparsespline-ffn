"""Edge-case + validation tests for ``schedules.py``.

Adds coverage for paths not exercised by ``test_schedules.py``:

  - ``MLPFFN`` with gelu, bias=True, hidden_dim override, mlp_ratio sweeps.
  - ``MLPFFN`` constructor validation (d, hidden_dim, activation).
  - ``should_replace_layer`` validation errors and remaining schedules.
  - ``build_ffn`` with ``ffn_type="mlp"`` (plain MLP path) and the
    ``"fm_b1"`` alias listed in ``VALID_FFN_TYPES``.
  - ``build_ffn`` metadata attributes (``schedule``, ``layer_idx``,
    ``num_layers``, ``ffn_type_requested``, ``is_sparsespline_replacement``).
  - ``build_fullmix_tucker_ffn`` defaults.
"""
from __future__ import annotations

import pytest
import torch

from sparsespline_ffn import (
    MLPFFN,
    FullMixTuckerFFN,
    build_ffn,
    should_replace_layer,
)
from sparsespline_ffn.schedules import (
    VALID_FFN_TYPES,
    VALID_SCHEDULES,
    build_fullmix_tucker_ffn,
)


# ---- MLPFFN: activation, bias, hidden_dim --------------------------------


def test_mlp_gelu_path_runs():
    mlp = MLPFFN(d=16, mlp_ratio=2, activation="gelu")
    x = torch.randn(3, 5, 16)
    y = mlp(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mlp_bias_true_creates_bias_params():
    mlp = MLPFFN(d=16, mlp_ratio=2, bias=True)
    assert mlp.up.bias is not None
    assert mlp.down.bias is not None
    # Forward still works with bias.
    y = mlp(torch.randn(2, 16))
    assert y.shape == (2, 16)


def test_mlp_hidden_dim_override():
    mlp = MLPFFN(d=16, hidden_dim=24, mlp_ratio=4)  # explicit hidden_dim wins
    assert mlp.hidden_dim == 24
    assert mlp.up.weight.shape == (24, 16)
    assert mlp.down.weight.shape == (16, 24)


def test_mlp_rejects_zero_d():
    with pytest.raises(ValueError, match="d must be positive"):
        MLPFFN(d=0)


def test_mlp_rejects_zero_hidden_dim():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        MLPFFN(d=16, hidden_dim=0)


def test_mlp_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation must be"):
        MLPFFN(d=16, activation="silu")


@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 8])
def test_mlp_param_count_matches_2_d_squared_ratio(ratio):
    """No-bias MLP has exactly 2*d*hidden = 2*ratio*d^2 params."""
    d = 32
    mlp = MLPFFN(d=d, mlp_ratio=ratio, bias=False)
    expected = 2 * d * (ratio * d)
    actual = sum(p.numel() for p in mlp.parameters())
    assert actual == expected


# ---- should_replace_layer: validation + remaining schedules ---------------


def test_should_replace_rejects_nonpositive_num_layers():
    with pytest.raises(ValueError, match="num_layers"):
        should_replace_layer(layer_idx=0, num_layers=0)


def test_should_replace_rejects_out_of_range_layer_idx():
    with pytest.raises(ValueError, match="layer_idx"):
        should_replace_layer(layer_idx=12, num_layers=12)
    with pytest.raises(ValueError, match="layer_idx"):
        should_replace_layer(layer_idx=-1, num_layers=12)


def test_should_replace_rejects_unknown_schedule():
    with pytest.raises(ValueError, match="unknown schedule"):
        should_replace_layer(layer_idx=0, num_layers=12, schedule="reverse")


@pytest.mark.parametrize("schedule", list(VALID_SCHEDULES))
def test_should_replace_accepts_all_valid_schedules(schedule):
    """Smoke test: every documented schedule must run for a 12-layer model."""
    selected = [
        should_replace_layer(layer_idx=i, num_layers=12, schedule=schedule)
        for i in range(12)
    ]
    # 'all' selects all 12, 'none' selects 0; everything else is in-between.
    if schedule == "all":
        assert sum(selected) == 12
    elif schedule == "none":
        assert sum(selected) == 0
    else:
        assert 0 < sum(selected) < 12


def test_late_quarter_handles_small_n():
    """``max(1, n // 4)`` ensures a 1-layer model always picks layer 0."""
    assert should_replace_layer(layer_idx=0, num_layers=1, schedule="late_quarter") is True
    assert should_replace_layer(layer_idx=0, num_layers=2, schedule="late_quarter") is False
    assert should_replace_layer(layer_idx=1, num_layers=2, schedule="late_quarter") is True


def test_every2_and_every4_pattern():
    every2 = [should_replace_layer(i, 8, "every2") for i in range(8)]
    every4 = [should_replace_layer(i, 8, "every4") for i in range(8)]
    assert every2 == [True, False] * 4
    assert every4 == [True, False, False, False] * 2


# ---- build_ffn: alternative ffn_types and metadata -----------------------


def test_build_ffn_mlp_type_is_plain_mlp():
    """ffn_type='mlp' should always return an MLPFFN regardless of schedule."""
    module = build_ffn(ffn_type="mlp", d=16, layer_idx=0, num_layers=4,
                       schedule="all")
    assert isinstance(module, MLPFFN)
    assert module.ffn_type_effective == "mlp"
    # The 'mlp' type does NOT carry the is_sparsespline_replacement flag.
    assert not hasattr(module, "is_sparsespline_replacement")


def test_build_ffn_fm_b1_alias_is_recognized():
    """`fm_b1` is the alias listed in VALID_FFN_TYPES; should behave like
    fullmix_tucker for selection purposes."""
    assert "fm_b1" in VALID_FFN_TYPES
    module = build_ffn(ffn_type="fm_b1", d=16, layer_idx=3, num_layers=4,
                       schedule="late", R_o=8, R_i=8, R_b=4, G=6)
    # build_ffn currently routes both fullmix_tucker and fm_b1 through the
    # FullMixTuckerFFN branch when the layer is selected.
    assert isinstance(module, FullMixTuckerFFN)
    assert module.ffn_type_requested == "fm_b1"


def test_build_ffn_metadata_attached():
    module = build_ffn(ffn_type="fullmix_tucker", d=16, layer_idx=2,
                       num_layers=4, schedule="late",
                       R_o=8, R_i=8, R_b=4, G=6)
    assert module.layer_idx == 2
    assert module.num_layers == 4
    assert module.schedule == "late"
    assert module.ffn_type_requested == "fullmix_tucker"
    assert module.is_sparsespline_replacement is True


def test_build_ffn_unselected_layer_records_fallback_metadata():
    module = build_ffn(ffn_type="fullmix_tucker", d=16, layer_idx=0,
                       num_layers=4, schedule="late",
                       R_o=8, R_i=8, R_b=4, G=6)
    assert isinstance(module, MLPFFN)
    assert module.is_sparsespline_replacement is False
    assert module.fallback_from == "fullmix_tucker"
    assert module.layer_idx == 0
    assert module.num_layers == 4


def test_build_ffn_case_insensitive_schedule_and_type():
    module = build_ffn(ffn_type="FullMix_Tucker", d=16, layer_idx=2,
                       num_layers=4, schedule="LATE",
                       R_o=8, R_i=8, R_b=4, G=6)
    assert isinstance(module, FullMixTuckerFFN)
    assert module.schedule == "late"


# ---- build_fullmix_tucker_ffn: defaults ----------------------------------


def test_build_fullmix_tucker_ffn_defaults_m_to_d():
    layer = build_fullmix_tucker_ffn(d=16)  # m omitted
    assert layer.cfg.m == 16


def test_build_fullmix_tucker_ffn_explicit_m():
    layer = build_fullmix_tucker_ffn(d=16, m=24)
    assert layer.cfg.m == 24
