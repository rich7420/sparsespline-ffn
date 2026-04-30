"""Numerical drift across dtypes: fp32 / bf16 / fp16.

The K.0.1 contract requires the eventual fused kernel to match the form-B
reference within bf16's ~1e-3 relative tolerance.  This benchmark measures
the reference's own fp32 / bf16 / fp16 drift so we know what part of any
future kernel's drift is "kernel" vs "dtype."

We compute:

  - fwd-only relative error vs fp32 reference, on a synthetic input;
  - fwd+bwd parameter-gradient relative error, again vs fp32;
  - largest single-element absolute error (bf16's worst case).

bf16 is the default training dtype for nanochat, so it gets the most
attention.  fp16 is reported as a curiosity — it has the same exponent
bits as fp32 but only 10 mantissa bits, so it tends to overflow the
beta tensor on large grids.

Auto-detects CUDA (bf16 / fp16 are CUDA-friendly); on CPU only fp32 vs
manual bf16 cast is reported.
"""
from __future__ import annotations

import time

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_pair(cfg, dtype, device):
    torch.manual_seed(0)
    fp32 = FullMixTuckerFFN(cfg).to(device).float()
    other = FullMixTuckerFFN(cfg).to(device=device, dtype=dtype)
    with torch.no_grad():
        for p32, p_o in zip(fp32.parameters(), other.parameters(),
                             strict=True):
            p_o.copy_(p32.detach().to(dtype))
        for b32, b_o in zip(fp32.buffers(), other.buffers(), strict=True):
            b_o.copy_(b32.detach().to(b_o.dtype))
    return fp32, other


def _measure(fp32_model, other_model, x_fp32, dtype, do_bwd: bool):
    x_other = x_fp32.detach().to(dtype)

    if do_bwd:
        x_fp32_g = x_fp32.detach().clone().requires_grad_(True)
        x_oth_g = x_other.detach().clone().requires_grad_(True)
        y_fp32 = fp32_model(x_fp32_g)
        y_oth = other_model(x_oth_g).float()
        y_fp32.pow(2).sum().backward()
        y_oth.pow(2).sum().backward()
        rel_y = ((y_fp32 - y_oth).norm() /
                 (y_fp32.norm() + 1e-9)).item()
        max_abs_y = (y_fp32 - y_oth).abs().max().item()

        rel_grads = []
        for (n32, p32), (no, po) in zip(
            fp32_model.named_parameters(),
            other_model.named_parameters(),
            strict=True,
        ):
            assert n32 == no
            if p32.grad is None or po.grad is None:
                continue
            ref = p32.grad.float()
            cmp = po.grad.float()
            rel = (ref - cmp).norm() / (ref.norm() + 1e-9)
            rel_grads.append(rel.item())
        max_grad_rel = max(rel_grads) if rel_grads else float("nan")
        fp32_model.zero_grad(set_to_none=True)
        other_model.zero_grad(set_to_none=True)
        return rel_y, max_abs_y, max_grad_rel
    else:
        with torch.no_grad():
            y_fp32 = fp32_model(x_fp32)
            y_oth = other_model(x_other).float()
        rel_y = ((y_fp32 - y_oth).norm() /
                 (y_fp32.norm() + 1e-9)).item()
        max_abs_y = (y_fp32 - y_oth).abs().max().item()
        return rel_y, max_abs_y, float("nan")


def main():
    device = _device()
    cfg = FullMixTuckerConfig(d=128, m=128, R_o=64, R_i=64, R_b=8, G=16)

    print("=" * 78)
    print("Numerical drift across dtypes (vs fp32 reference)")
    print(f"device={device}, "
          f"d={cfg.d}, R=({cfg.R_o},{cfg.R_i},{cfg.R_b}), G={cfg.G}")
    print("=" * 78)

    dtypes_to_test = [torch.bfloat16]
    if device.type == "cuda":
        dtypes_to_test.append(torch.float16)

    print(f"\n{'dtype':<10} {'mode':<10} {'rel(y)':>12} {'max|y-y32|':>14} "
          f"{'max rel grad':>14} {'wall(ms)':>10}")
    print("-" * 78)

    torch.manual_seed(123)
    x_fp32 = torch.randn(8, 64, cfg.d, device=device).float()

    for dtype in dtypes_to_test:
        fp32_model, other_model = _build_pair(cfg, dtype, device)
        for mode, do_bwd in [("fwd", False), ("fwd+bwd", True)]:
            t0 = time.perf_counter()
            rel_y, max_abs, rel_g = _measure(
                fp32_model, other_model, x_fp32, dtype, do_bwd
            )
            wall = (time.perf_counter() - t0) * 1000
            label = str(dtype).replace("torch.", "")
            print(f"{label:<10} {mode:<10} {rel_y:>12.4e} {max_abs:>14.4e} "
                  f"{rel_g:>14.4e} {wall:>10.2f}")

    print("\n" + "=" * 78)
    print("Headline:")
    print("  - bf16 fwd rel err <= 5e-2 is the contract any kernel inherits.")
    print("  - bf16 grad rel err is the spec FlashKAT-style backward needs to beat.")
    print("  - fp16 may hit overflow on the beta tensor at large grids; if seen,")
    print("    treat as expected and avoid fp16 for FullMix-Tucker training.")


if __name__ == "__main__":
    main()
