"""Smoke tests for the ``python -m sparsespline_ffn`` diagnostic CLI."""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from sparsespline_ffn.__main__ import cmd_config, cmd_info, main


def _capture(callable_, *args, **kwargs) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = callable_(*args, **kwargs)
    return rc, buf.getvalue()


def test_info_subcommand_runs_and_reports_version():
    rc, out = _capture(main, ["info"])
    assert rc == 0
    assert "SparseSpline-FFN" in out
    assert "version" in out
    assert "is_kernel_available" in out


def test_no_subcommand_defaults_to_info():
    rc, out = _capture(main, [])
    assert rc == 0
    assert "SparseSpline-FFN" in out


def test_config_subcommand_prints_param_counts():
    rc, out = _capture(main, [
        "config", "--d", "32", "--R_o", "8", "--R_i", "8",
        "--R_b", "4", "--G", "8",
    ])
    assert rc == 0
    # 32 * 32 (mixer) + 32 * 8 (U) + 32 * 8 (V) + 9 * 4 (Q) + 8*8*4 (C) + 1
    # = 1024 + 256 + 256 + 36 + 256 + 1 = 1829
    assert "params/layer    : 1,829" in out
    assert "vs MLP" in out


def test_check_kernel_returns_int_exit_code():
    """``check-kernel`` returns 0 on success, 1 on failure — never None."""
    rc, _ = _capture(main, ["check-kernel"])
    assert rc in (0, 1), f"unexpected return code {rc}"


@pytest.mark.parametrize("ratio_d", [(2, 64), (1, 16)])  # mixer m vs d
def test_config_runs_at_various_scales(ratio_d):
    """The CLI's ``config`` subcommand handles small and large d cleanly."""
    ratio, d = ratio_d
    rc, out = _capture(main, ["config", "--d", str(d),
                              "--m", str(ratio * d),
                              "--R_o", "4", "--R_i", "4",
                              "--R_b", "2", "--G", "4"])
    assert rc == 0
    assert f"d={d}" in out
    assert f"m={ratio * d}" in out


def test_cmd_info_returns_zero():
    """Direct call to cmd_info also works (it's used as the default)."""
    import argparse
    rc, out = _capture(cmd_info, argparse.Namespace())
    assert rc == 0
    assert "SparseSpline-FFN" in out


def test_cmd_config_returns_zero():
    import argparse
    args = argparse.Namespace(d=16, m=None, R_o=4, R_i=4, R_b=2, G=4)
    rc, _ = _capture(cmd_config, args)
    assert rc == 0
