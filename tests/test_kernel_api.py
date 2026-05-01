"""User-facing kernel API tests.

Pins the contract that downstream users rely on:

  - ``use_kernel`` accepts ``False`` / ``True`` / ``"required"``;
  - ``"required"`` raises with an actionable message on CPU and on
    no-Triton machines;
  - ``True`` silently falls back to form-B when the kernel is unavailable;
  - numerical equivalence between form-B and the kernel within the K.0.1
    tolerance (CUDA-only test, auto-skipped on CPU);
  - ``is_kernel_available()`` reflects the runtime gate;
  - ``build_fullmix_tucker_ffn`` / ``build_ffn`` forward the flag.
"""
from __future__ import annotations

import pytest
import torch
from conftest import make_small_ffn

from sparsespline_ffn import (
    FullMixTuckerConfig,
    FullMixTuckerFFN,
    build_ffn,
    build_fullmix_tucker_ffn,
    is_kernel_available,
)
from sparsespline_ffn.kernels import HAS_TRITON

# ---- Config validation ----------------------------------------------------


def test_use_kernel_accepts_bool_and_required_string():
    FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2, use_kernel=False)
    FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2, use_kernel=True)
    FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2, use_kernel="required")


@pytest.mark.parametrize("bad_value", ["true", "yes", "auto", 1, 0, None])
def test_use_kernel_rejects_invalid_values(bad_value):
    with pytest.raises(ValueError, match="use_kernel"):
        FullMixTuckerConfig(
            d=8, m=8, R_o=4, R_i=4, R_b=2, use_kernel=bad_value
        )


# ---- Default (use_kernel=False) -------------------------------------------


def test_default_use_kernel_is_false():
    cfg = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2)
    assert cfg.use_kernel is False


def test_form_b_runs_independent_of_triton():
    """Form-B is the permanent reference; it must run regardless of
    Triton availability and regardless of CUDA."""
    ffn = make_small_ffn()
    x = torch.randn(2, ffn.cfg.d)
    y = ffn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert ffn.kernel_will_run(x) is False


# ---- True: silent fallback ------------------------------------------------


def test_use_kernel_true_falls_back_silently_on_cpu():
    """``use_kernel=True`` must NOT crash on CPU; it falls back to form-B."""
    cfg = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2, use_kernel=True)
    ffn = FullMixTuckerFFN(cfg)
    x = torch.randn(2, cfg.d)  # CPU tensor
    y = ffn(x)
    assert torch.isfinite(y).all()
    assert ffn.kernel_will_run(x) is False, (
        "kernel_will_run must report False when running on CPU regardless "
        "of the use_kernel setting"
    )


# ---- "required": loud failure ---------------------------------------------


def test_use_kernel_required_raises_on_cpu_input():
    cfg = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2,
                              use_kernel="required")
    ffn = FullMixTuckerFFN(cfg)
    x = torch.randn(2, cfg.d)  # CPU
    with pytest.raises(RuntimeError, match="not on CUDA"):
        ffn(x)


def test_use_kernel_required_message_points_to_cuda_extras():
    """The failure message must hint at the install fix."""
    cfg = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2,
                              use_kernel="required")
    ffn = FullMixTuckerFFN(cfg)
    x = torch.randn(2, cfg.d)
    with pytest.raises(RuntimeError) as excinfo:
        ffn(x)
    assert "use_kernel=True" in str(excinfo.value), (
        "Error message should suggest the graceful-fallback alternative"
    )


def test_kernel_will_run_does_not_raise_for_required_on_cpu():
    """``kernel_will_run`` is a passive query — must NOT raise even when
    the actual ``forward`` would (because it's documented as introspection)."""
    cfg = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2,
                              use_kernel="required")
    ffn = FullMixTuckerFFN(cfg)
    x = torch.randn(2, cfg.d)
    assert ffn.kernel_will_run(x) is False


# ---- is_kernel_available() top-level helper ------------------------------


def test_is_kernel_available_matches_runtime_gate():
    """``is_kernel_available()`` must agree with the actual layer behavior."""
    expected = HAS_TRITON and torch.cuda.is_available()
    assert is_kernel_available() is expected


def test_is_kernel_available_does_not_raise():
    # Idempotent / never-raising; safe at module import time in user code.
    assert isinstance(is_kernel_available(), bool)


# ---- Factory passthrough --------------------------------------------------


def test_build_fullmix_tucker_ffn_passes_use_kernel():
    layer = build_fullmix_tucker_ffn(
        d=8, R_o=4, R_i=4, R_b=2, G=4, use_kernel=True
    )
    assert layer.cfg.use_kernel is True


def test_build_fullmix_tucker_ffn_passes_required():
    layer = build_fullmix_tucker_ffn(
        d=8, R_o=4, R_i=4, R_b=2, G=4, use_kernel="required"
    )
    assert layer.cfg.use_kernel == "required"


def test_build_ffn_factory_passes_use_kernel():
    """The build_ffn factory routes use_kernel through **fullmix_kwargs."""
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
        use_kernel=True,
    )
    assert isinstance(module, FullMixTuckerFFN)
    assert module.cfg.use_kernel is True


# ---- Numerical equivalence (CUDA-only, auto-skip) ------------------------


@pytest.mark.cuda
def test_kernel_matches_form_b_within_tolerance():
    """The Triton kernel must match form-B at fp32 1e-5 / bf16 5e-3 — the
    K.0.1 contract.  Auto-skipped on CPU and on no-Triton installs."""
    if not HAS_TRITON:
        pytest.skip("Triton not available")
    cfg = FullMixTuckerConfig(d=64, m=64, R_o=16, R_i=16, R_b=8, G=12)
    torch.manual_seed(0)
    ref = FullMixTuckerFFN(cfg).cuda().float()
    cfg_k = FullMixTuckerConfig(
        d=64, m=64, R_o=16, R_i=16, R_b=8, G=12, use_kernel=True
    )
    torch.manual_seed(0)
    kern = FullMixTuckerFFN(cfg_k).cuda().float()

    x = torch.randn(8, cfg.d, device="cuda")
    with torch.no_grad():
        y_ref = ref(x)
        y_kern = kern(x)
    assert kern.kernel_will_run(x) is True
    rel = (y_ref - y_kern).norm() / (y_ref.norm() + 1e-9)
    assert rel < 1e-5, f"fp32 form-B vs kernel rel err {rel.item():.2e}"


@pytest.mark.cuda
def test_kernel_required_runs_on_cuda():
    """``use_kernel='required'`` must work end-to-end on CUDA."""
    if not HAS_TRITON:
        pytest.skip("Triton not available")
    cfg = FullMixTuckerConfig(d=32, m=32, R_o=8, R_i=8, R_b=4, G=8,
                              use_kernel="required")
    ffn = FullMixTuckerFFN(cfg).cuda().float()
    x = torch.randn(4, cfg.d, device="cuda", requires_grad=True)
    y = ffn(x)
    y.pow(2).sum().backward()
    assert ffn.kernel_will_run(x) is True
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---- Checkpoint roundtrip across use_kernel toggle -----------------------


def _state_dict_keys(layer: FullMixTuckerFFN) -> set[str]:
    return set(layer.state_dict().keys())


def test_state_dict_keys_independent_of_use_kernel():
    """The kernel only changes runtime; ``state_dict()`` must contain the
    exact same parameter / buffer keys regardless of ``use_kernel``.
    """
    cfg_off = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2, G=4,
                                  use_kernel=False)
    cfg_on = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2, G=4,
                                 use_kernel=True)
    cfg_req = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2, G=4,
                                  use_kernel="required")
    keys = _state_dict_keys(FullMixTuckerFFN(cfg_off))
    assert keys == _state_dict_keys(FullMixTuckerFFN(cfg_on))
    assert keys == _state_dict_keys(FullMixTuckerFFN(cfg_req))


def test_load_kernel_checkpoint_into_form_b_and_back():
    """Train with use_kernel=True, save state_dict, load into use_kernel=False
    layer and back into another use_kernel=True layer.  Output must match
    the original on both sides — this is the contract that lets users
    train with the kernel and serve with the reference (or vice versa).

    On CPU, ``use_kernel=True`` silently falls back to form-B; the test
    exercises the same code paths but only proves the state_dict layout
    is identical.  On CUDA the test additionally verifies forward
    equivalence within fp32 tolerance.
    """
    cfg_kern = FullMixTuckerConfig(d=16, m=16, R_o=8, R_i=8, R_b=4, G=6,
                                   use_kernel=True)
    cfg_ref = FullMixTuckerConfig(d=16, m=16, R_o=8, R_i=8, R_b=4, G=6,
                                  use_kernel=False)

    torch.manual_seed(0)
    src = FullMixTuckerFFN(cfg_kern)

    # Move some weight to make this not just an init-copy roundtrip.
    with torch.no_grad():
        src.U.data.add_(0.1)
        src.gamma.data.fill_(1.5)

    sd = src.state_dict()

    # Load into a use_kernel=False layer (different cfg).
    dst_ref = FullMixTuckerFFN(cfg_ref)
    missing, unexpected = dst_ref.load_state_dict(sd, strict=True)
    # PyTorch returns named tuples; both lists should be empty in strict mode.
    assert list(missing) == [] and list(unexpected) == []

    # Load back into another use_kernel=True layer.
    dst_kern = FullMixTuckerFFN(cfg_kern)
    dst_kern.load_state_dict(sd, strict=True)

    # Forward must match across the three layers (on CPU all three use form-B
    # because kernel falls back; on CUDA two of them use the Triton path).
    if torch.cuda.is_available():
        src = src.cuda().float()
        dst_ref = dst_ref.cuda().float()
        dst_kern = dst_kern.cuda().float()
        x = torch.randn(4, 16, device="cuda")
    else:
        src = src.float()
        dst_ref = dst_ref.float()
        dst_kern = dst_kern.float()
        x = torch.randn(4, 16)

    with torch.no_grad():
        y_src = src(x)
        y_ref = dst_ref(x)
        y_kern = dst_kern(x)

    rel_ref = (y_src - y_ref).norm() / (y_src.norm() + 1e-9)
    rel_kern = (y_src - y_kern).norm() / (y_src.norm() + 1e-9)
    # form-B vs form-B (same weights) must be bit-identical.
    assert rel_ref.item() < 1e-6, (
        f"form-B reload differed from source: rel {rel_ref.item():.2e}"
    )
    # kernel vs source: bit-identical on CPU (both fall back), 1e-5 on CUDA.
    cap = 1e-5 if torch.cuda.is_available() else 1e-6
    assert rel_kern.item() < cap, (
        f"kernel reload differed from source: rel {rel_kern.item():.2e}"
    )


def test_save_and_load_through_disk_roundtrip(tmp_path):
    """End-to-end: torch.save -> torch.load -> load_state_dict, across the
    use_kernel toggle.  Catches any pickling/buffer surprises."""
    cfg_a = FullMixTuckerConfig(d=12, m=12, R_o=6, R_i=6, R_b=3, G=4,
                                use_kernel=True)
    cfg_b = FullMixTuckerConfig(d=12, m=12, R_o=6, R_i=6, R_b=3, G=4,
                                use_kernel=False)

    torch.manual_seed(2)
    src = FullMixTuckerFFN(cfg_a)
    with torch.no_grad():
        src.gamma.data.fill_(0.7)

    ckpt_path = tmp_path / "fmt.pt"
    torch.save(src.state_dict(), ckpt_path)

    dst = FullMixTuckerFFN(cfg_b)
    dst.load_state_dict(torch.load(ckpt_path, weights_only=True))

    x = torch.randn(3, 12)
    with torch.no_grad():
        y_src = src(x)
        y_dst = dst(x)
    assert torch.allclose(y_src, y_dst, atol=1e-6)


def test_kernel_does_not_register_extra_state():
    """A layer with ``use_kernel=True`` must NOT add hidden buffers /
    parameters that would bloat the checkpoint or break load_state_dict
    against a use_kernel=False layer."""
    cfg_off = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2, G=4,
                                  use_kernel=False)
    cfg_on = FullMixTuckerConfig(d=8, m=8, R_o=4, R_i=4, R_b=2, G=4,
                                 use_kernel=True)
    n_off = sum(p.numel() for p in FullMixTuckerFFN(cfg_off).parameters())
    n_on = sum(p.numel() for p in FullMixTuckerFFN(cfg_on).parameters())
    assert n_off == n_on, (
        f"use_kernel toggle changed parameter count: "
        f"off={n_off}, on={n_on}"
    )

    buffers_off = {n for n, _ in FullMixTuckerFFN(cfg_off).named_buffers()}
    buffers_on = {n for n, _ in FullMixTuckerFFN(cfg_on).named_buffers()}
    assert buffers_off == buffers_on, (
        f"use_kernel toggle changed buffer set: "
        f"only_off={buffers_off - buffers_on}, only_on={buffers_on - buffers_off}"
    )
