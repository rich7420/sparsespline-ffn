from __future__ import annotations

from pathlib import Path

import tomllib

import sparsespline_ffn


def test_version_matches_pyproject() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    assert sparsespline_ffn.__version__ == pyproject["project"]["version"]
