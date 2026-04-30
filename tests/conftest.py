"""Shared test fixtures, helpers, and pytest hooks.

Provides:

  - ``make_small_ffn`` / ``make_stack`` helpers used across multiple test
    files (previously duplicated as private ``_make_small`` / ``_stack_layers``);
  - ``residual_stack_forward`` / ``pre_rmsnorm_stack_forward`` — the same
    composition helpers used in stacking tests;
  - ``capture_bin_frac`` context manager that monkey-patches
    ``FullMixTuckerFFN._bin_and_frac`` to capture the spline lookup state;
  - automatic skip for any test marked ``@pytest.mark.cuda`` when CUDA is
    unavailable.

Tests can import these directly:

    from conftest import make_small_ffn, make_stack
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN

# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip ``@pytest.mark.cuda`` tests when CUDA is not available.

    This avoids the ``pytest.skip()`` boilerplate inside every CUDA-only
    test body and keeps the marker semantics consistent with
    ``pyproject.toml``'s registered ``cuda`` marker.
    """
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA not available on this runner")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


# ---------------------------------------------------------------------------
# Module helpers — explicit imports preferred over fixtures so the call sites
# stay readable and don't tie test bodies to pytest dependency injection.
# ---------------------------------------------------------------------------


def make_small_ffn(
    *,
    use_mixer: bool = True,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    **overrides: Any,
) -> FullMixTuckerFFN:
    """Build a small FullMixTuckerFFN suitable for fast CPU tests.

    Defaults: d=m=16, R_o=R_i=8, R_b=4, G=6, grid=[-2, 2].  Any keyword
    argument is forwarded to ``FullMixTuckerConfig`` and overrides the
    default of the same name.
    """
    cfg = FullMixTuckerConfig(
        d=overrides.pop("d", 16),
        m=overrides.pop("m", 16),
        R_o=overrides.pop("R_o", 8),
        R_i=overrides.pop("R_i", 8),
        R_b=overrides.pop("R_b", 4),
        G=overrides.pop("G", 6),
        grid_lo=overrides.pop("grid_lo", -2.0),
        grid_hi=overrides.pop("grid_hi", 2.0),
        use_mixer=use_mixer,
        bias_in_mixer=overrides.pop("bias_in_mixer", False),
        **overrides,
    )
    torch.manual_seed(seed)
    return FullMixTuckerFFN(cfg).to(dtype)


def make_stack(
    K: int, *, seed_offset: int | None = None, **cfg_overrides: Any
) -> tuple[torch.nn.ModuleList, FullMixTuckerConfig]:
    """Build K stacked FullMixTuckerFFN layers with a shared config.

    By default the function does not reset ``torch.manual_seed`` between
    layers — the caller is expected to set the seed externally before
    calling, and the natural RNG progression then gives each layer
    different parameters (which keeps ``[U_1 | ... | U_K]`` full rank for
    the F.5.1 cumulative-coverage check).

    Pass ``seed_offset=N`` to force per-layer seeds ``N, N+1, ...``; use
    only when you need bit-exact reproducibility independent of any
    earlier RNG consumption.
    """
    cfg = FullMixTuckerConfig(
        d=cfg_overrides.pop("d", 64),
        m=cfg_overrides.pop("m", 64),
        R_o=cfg_overrides.pop("R_o", 16),
        R_i=cfg_overrides.pop("R_i", 16),
        R_b=cfg_overrides.pop("R_b", 4),
        G=cfg_overrides.pop("G", 12),
        **cfg_overrides,
    )
    layers: list[FullMixTuckerFFN] = []
    for i in range(K):
        if seed_offset is not None:
            torch.manual_seed(seed_offset + i)
        layers.append(FullMixTuckerFFN(cfg))
    return torch.nn.ModuleList(layers), cfg


def residual_stack_forward(
    layers: torch.nn.ModuleList, x: torch.Tensor
) -> torch.Tensor:
    """Apply layers in residual fashion: x_{l+1} = x_l + FFN(x_l)."""
    h = x
    for ffn in layers:
        h = h + ffn(h)
    return h


def pre_rmsnorm_stack_forward(
    layers: torch.nn.ModuleList, x: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Transformer-style: x_{l+1} = x_l + FFN(RMSNorm(x_l))."""
    h = x
    for ffn in layers:
        rms = h.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        h = h + ffn(h / rms)
    return h


@contextmanager
def capture_bin_frac(ffn: FullMixTuckerFFN):
    """Context manager that captures ``(z, bin_idx, t)`` from the next
    ``ffn`` forward pass into a dict, then restores the original method.

    Usage:
        with capture_bin_frac(ffn) as captured:
            ffn(x)
        assert captured["t"].min() >= 0.0
    """
    captured: dict[str, torch.Tensor] = {}
    original = ffn._bin_and_frac

    def spy(z: torch.Tensor):
        bin_idx, t = original(z)
        captured["z"] = z.detach().clone()
        captured["bin"] = bin_idx.detach().clone()
        captured["t"] = t.detach().clone()
        return bin_idx, t

    ffn._bin_and_frac = spy  # type: ignore[method-assign]
    try:
        yield captured
    finally:
        ffn._bin_and_frac = original  # type: ignore[method-assign]
