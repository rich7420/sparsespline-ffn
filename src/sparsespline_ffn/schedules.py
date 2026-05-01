"""FFN factory helpers for transformer integrations.

The factory deliberately exposes replacement schedules as plain layer-index
logic.  Downstream projects can keep their own transformer code and only swap
the FFN module at construction time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsespline_ffn.fullmix_tucker import FullMixTuckerConfig, FullMixTuckerFFN

VALID_SCHEDULES = ("all", "every2", "every4", "early", "late", "late_quarter", "none")
VALID_FFN_TYPES = ("mlp", "fullmix_tucker", "fm_b1")


class MLPFFN(nn.Module):
    """Baseline transformer FFN: ``d -> mlp_ratio*d -> d``."""

    _VALID_ACTIVATIONS = ("gelu", "relu_sq")

    def __init__(
        self,
        d: int,
        hidden_dim: int | None = None,
        *,
        mlp_ratio: int = 4,
        activation: str = "relu_sq",
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        if hidden_dim is None:
            hidden_dim = mlp_ratio * d
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if activation not in self._VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {self._VALID_ACTIVATIONS}, got {activation!r}"
            )
        self.d = int(d)
        self.hidden_dim = int(hidden_dim)
        self.activation = activation
        self.up = nn.Linear(self.d, self.hidden_dim, bias=bias)
        self.down = nn.Linear(self.hidden_dim, self.d, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        if self.activation == "gelu":
            h = F.gelu(h)
        else:
            h = F.relu(h).square()
        return self.down(h)


def should_replace_layer(layer_idx: int, num_layers: int, schedule: str = "all") -> bool:
    """Return whether a zero-based transformer layer should use SparseSpline-FFN.

    Examples
    --------
    >>> [should_replace_layer(i, 12, "late") for i in range(12)]
    [False, False, False, False, False, False, True, True, True, True, True, True]
    >>> [should_replace_layer(i, 8, "every2") for i in range(8)]
    [True, False, True, False, True, False, True, False]
    >>> should_replace_layer(0, 4, "none")
    False
    >>> should_replace_layer(0, 4, "all")
    True
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(
            f"layer_idx must satisfy 0 <= layer_idx < num_layers, got "
            f"layer_idx={layer_idx}, num_layers={num_layers}"
        )
    schedule_key = schedule.lower()
    if schedule_key not in VALID_SCHEDULES:
        raise ValueError(f"unknown schedule: {schedule!r}")
    if schedule_key == "none":
        return False
    if schedule_key == "all":
        return True
    if schedule_key == "every2":
        return layer_idx % 2 == 0
    if schedule_key == "every4":
        return layer_idx % 4 == 0
    if schedule_key == "early":
        return layer_idx < num_layers // 2
    if schedule_key == "late":
        return layer_idx >= num_layers // 2
    if schedule_key == "late_quarter":
        return layer_idx >= num_layers - max(1, num_layers // 4)
    raise AssertionError("unreachable")


def build_fullmix_tucker_ffn(
    *,
    d: int,
    m: int | None = None,
    R_o: int = 96,
    R_i: int = 96,
    R_b: int = 16,
    G: int = 20,
    grid_lo: float = -3.0,
    grid_hi: float = 3.0,
    use_mixer: bool = True,
    bias_in_mixer: bool = False,
    use_kernel: bool | str = False,
) -> FullMixTuckerFFN:
    """Construct the reference FullMix-Tucker SparseSpline-FFN layer.

    ``use_kernel`` follows the tri-state semantics defined on
    :class:`FullMixTuckerConfig`:

      - ``False``     → form-B reference only (default).
      - ``True``      → prefer Triton kernel; silent fallback on CPU /
                        no-Triton.
      - ``"required"`` → demand kernel; raise if it cannot run.
    """
    if m is None:
        m = d
    cfg = FullMixTuckerConfig(
        d=d,
        m=m,
        R_o=R_o,
        R_i=R_i,
        R_b=R_b,
        G=G,
        grid_lo=grid_lo,
        grid_hi=grid_hi,
        use_mixer=use_mixer,
        bias_in_mixer=bias_in_mixer,
        use_kernel=use_kernel,
    )
    return FullMixTuckerFFN(cfg)


def build_ffn(
    *,
    ffn_type: str,
    d: int,
    layer_idx: int = 0,
    num_layers: int = 1,
    schedule: str = "all",
    mlp_ratio: int = 4,
    activation: str = "relu_sq",
    bias: bool = False,
    **fullmix_kwargs,
) -> nn.Module:
    """Build either an MLP fallback or a SparseSpline-FFN replacement.

    ``schedule`` is honored only for SparseSpline-FFN types.  If a layer is not
    selected, the factory returns an ``MLPFFN`` with metadata attached.
    """
    ffn_type_key = ffn_type.lower()
    if ffn_type_key not in VALID_FFN_TYPES:
        raise ValueError(f"unknown ffn_type: {ffn_type!r}")
    # nn.Module.__setattr__ is typed as accepting ``Tensor | Module`` only,
    # so plain attribute assignment of strings/ints fails mypy.  Route
    # metadata through ``setattr`` to bypass the descriptor typing.
    # Ruff B009 ("replace setattr with assignment") is silenced via
    # pyproject.toml's [tool.ruff.lint] ignore list.
    if ffn_type_key == "mlp":
        module: nn.Module = MLPFFN(d=d, mlp_ratio=mlp_ratio, activation=activation, bias=bias)
        setattr(module, "ffn_type_effective", "mlp")
    elif should_replace_layer(layer_idx, num_layers, schedule):
        module = build_fullmix_tucker_ffn(d=d, **fullmix_kwargs)
        setattr(module, "ffn_type_effective", "fullmix_tucker")
        setattr(module, "is_sparsespline_replacement", True)
    else:
        module = MLPFFN(d=d, mlp_ratio=mlp_ratio, activation=activation, bias=bias)
        setattr(module, "ffn_type_effective", "mlp")
        setattr(module, "is_sparsespline_replacement", False)
        setattr(module, "fallback_from", ffn_type_key)

    setattr(module, "ffn_type_requested", ffn_type)
    setattr(module, "schedule", schedule.lower())
    setattr(module, "layer_idx", layer_idx)
    setattr(module, "num_layers", num_layers)
    return module
