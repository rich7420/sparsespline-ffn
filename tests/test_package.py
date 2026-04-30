from __future__ import annotations

import sparsespline_ffn


def test_public_api_exports_core_symbols() -> None:
    assert sparsespline_ffn.__version__ == "0.1.0"
    assert hasattr(sparsespline_ffn, "FullMixTuckerConfig")
    assert hasattr(sparsespline_ffn, "FullMixTuckerFFN")
    assert hasattr(sparsespline_ffn, "build_ffn")
    assert hasattr(sparsespline_ffn, "should_replace_layer")
