from __future__ import annotations

import pytest
import torch

from sparsespline_ffn import MLPFFN, FullMixTuckerFFN, build_ffn, should_replace_layer


@pytest.mark.parametrize(
    ("schedule", "layer_idx", "selected"),
    [
        ("all", 8, True),
        ("none", 8, False),
        ("late", 8, True),
        ("early", 8, False),
        ("late_quarter", 8, False),
        ("late_quarter", 9, True),
        ("every2", 8, True),
        ("every4", 8, True),
    ],
)
def test_should_replace_layer(schedule: str, layer_idx: int, selected: bool) -> None:
    assert should_replace_layer(layer_idx=layer_idx, num_layers=12, schedule=schedule) is selected


def test_build_ffn_returns_fullmix_on_selected_layer() -> None:
    module = build_ffn(
        ffn_type="fullmix_tucker",
        d=16,
        layer_idx=3,
        num_layers=4,
        schedule="late",
        R_o=8,
        R_i=8,
        R_b=4,
        G=6,
    )
    assert isinstance(module, FullMixTuckerFFN)
    assert module.ffn_type_effective == "fullmix_tucker"
    x = torch.randn(2, 5, 16)
    assert module(x).shape == x.shape


def test_build_ffn_returns_mlp_fallback_on_unselected_layer() -> None:
    module = build_ffn(
        ffn_type="fullmix_tucker",
        d=16,
        layer_idx=0,
        num_layers=4,
        schedule="late",
        R_o=8,
        R_i=8,
        R_b=4,
        G=6,
    )
    assert isinstance(module, MLPFFN)
    assert module.fallback_from == "fullmix_tucker"
    x = torch.randn(2, 5, 16)
    assert module(x).shape == x.shape


def test_build_ffn_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown ffn_type"):
        build_ffn(ffn_type="bad", d=16)
