"""Activation-memory analysis: per-layer intermediate footprint retained for backward.

For each FFN type at production scale, we compute the bytes that autograd
must keep alive between forward and backward.

Reference (FullMix-Tucker, form B):
  z       : (N, m)         retained for stage 2's bin/frac
  beta    : (N, m, R_b)    retained for stage 3
  xi      : (N, R_i, R_b)  retained for stage 4
  eta     : (N, R_o)       retained for stage 5

  Dominant term at nanochat scale: beta = N * m * R_b.

MLP (relu_sq):
  h_pre   : (N, r*d)  retained for activation backward
  h_post  : (N, r*d)  output of relu_sq, retained for matmul backward

A separate flag reports torch.utils.checkpoint footprint estimates (rematerialize
beta in backward by re-running stages 1+2).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN


@dataclass(frozen=True)
class ActFootprint:
    name: str
    bytes_per_token: int
    bytes_total: int
    components: dict[str, int]


def fullmix_activation_bytes(
    cfg: FullMixTuckerConfig, N: int, dtype_bytes: int = 2
) -> ActFootprint:
    """Bytes retained between fwd/bwd for the 5-stage reference, in dtype.

    dtype_bytes: 2 = bf16/fp16, 4 = fp32.
    """
    z      = N * cfg.m * dtype_bytes
    beta   = N * cfg.m * cfg.R_b * dtype_bytes
    xi     = N * cfg.R_i * cfg.R_b * dtype_bytes
    eta    = N * cfg.R_o * dtype_bytes
    parts = {"z": z, "beta": beta, "xi": xi, "eta": eta}
    total = sum(parts.values())
    return ActFootprint(
        name="FullMix-Tucker (form B, no checkpoint)",
        bytes_per_token=total // N,
        bytes_total=total,
        components=parts,
    )


def fullmix_with_checkpoint_bytes(
    cfg: FullMixTuckerConfig, N: int, dtype_bytes: int = 2
) -> ActFootprint:
    """If we wrap the FFN in torch.utils.checkpoint, only the input is saved
    and stages 1-5 are re-run in backward.  Worst case keeps just the input.
    """
    x_in = N * cfg.d * dtype_bytes
    return ActFootprint(
        name="FullMix-Tucker (form B + checkpoint)",
        bytes_per_token=x_in // N,
        bytes_total=x_in,
        components={"x_in": x_in},
    )


def mlp_activation_bytes(d: int, mlp_ratio: int, N: int, dtype_bytes: int = 2) -> ActFootprint:
    h_pre  = N * mlp_ratio * d * dtype_bytes
    h_post = N * mlp_ratio * d * dtype_bytes
    parts = {"h_pre": h_pre, "h_post": h_post}
    total = sum(parts.values())
    return ActFootprint(
        name="MLPFFN (relu_sq)",
        bytes_per_token=total // N,
        bytes_total=total,
        components=parts,
    )


def _fmt_bytes(n: int) -> str:
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a >= 1024**3:
        return f"{sign}{a / 1024**3:.2f} GB"
    if a >= 1024**2:
        return f"{sign}{a / 1024**2:.2f} MB"
    if a >= 1024:
        return f"{sign}{a / 1024:.2f} KB"
    return f"{sign}{a} B"


def measure_actual_peak_memory(module: torch.nn.Module, x: torch.Tensor) -> float:
    """Peak resident bytes during fwd+bwd, measured via tracemalloc-equivalent.

    On CPU we use torch.cuda.max_memory_allocated() if CUDA available, otherwise
    we approximate via torch.profiler / by counting saved tensors.  Since this
    is CPU-only here, we rely on the analytical formula and only run a tiny
    forward to verify the formula's tensor shapes."""
    module(x)  # smoke run only
    return -1.0


def main() -> None:
    print("=" * 78)
    print("Activation memory analysis: per-layer footprint retained for backward")
    print("=" * 78)

    d = 768
    cfg = FullMixTuckerConfig(d=d, m=d, R_o=96, R_i=96, R_b=16, G=20)
    mlp_ratio = 4

    for label, B, T in [
        ("seq=512 single-batch",  1,  512),
        ("nanochat micro-batch",  4, 2048),
        ("nanochat full-batch",  16, 2048),
    ]:
        N = B * T
        print(f"\n>>> {label}: B={B}, T={T} -> N={N} tokens, dtype=bf16")
        fm = fullmix_activation_bytes(cfg, N, dtype_bytes=2)
        fm_ckp = fullmix_with_checkpoint_bytes(cfg, N, dtype_bytes=2)
        mlp = mlp_activation_bytes(d, mlp_ratio, N, dtype_bytes=2)

        print(f"\n  {mlp.name}:")
        for k, v in mlp.components.items():
            print(f"    {k:8s}: {_fmt_bytes(v):>10}")
        print(f"    {'TOTAL':8s}: {_fmt_bytes(mlp.bytes_total):>10}  "
              f"({_fmt_bytes(mlp.bytes_per_token)}/token)")

        print(f"\n  {fm.name}:")
        for k, v in fm.components.items():
            print(f"    {k:8s}: {_fmt_bytes(v):>10}")
        print(f"    {'TOTAL':8s}: {_fmt_bytes(fm.bytes_total):>10}  "
              f"({_fmt_bytes(fm.bytes_per_token)}/token)")

        print(f"\n  {fm_ckp.name}:")
        for k, v in fm_ckp.components.items():
            print(f"    {k:8s}: {_fmt_bytes(v):>10}")
        print(f"    {'TOTAL':8s}: {_fmt_bytes(fm_ckp.bytes_total):>10}  "
              f"({_fmt_bytes(fm_ckp.bytes_per_token)}/token)")

        print("\n  Ratios per layer:")
        print(f"    FullMix vs MLP        : "
              f"{fm.bytes_total / mlp.bytes_total:.2f}x  "
              f"({'larger' if fm.bytes_total > mlp.bytes_total else 'smaller'})")
        print(f"    FullMix+ckp vs MLP    : "
              f"{fm_ckp.bytes_total / mlp.bytes_total:.2f}x  smaller")

    print("\n" + "-" * 78)
    print("K-layer stack budget (Pattern A+ K=6, Pattern Full K=12):")
    print("  Assumes B=16, T=2048 (32K tokens), bf16, no checkpoint.")
    N = 16 * 2048
    fm_total = fullmix_activation_bytes(cfg, N, dtype_bytes=2).bytes_total
    fm_ckp_total = fullmix_with_checkpoint_bytes(cfg, N, dtype_bytes=2).bytes_total
    mlp_total = mlp_activation_bytes(d, mlp_ratio, N, dtype_bytes=2).bytes_total
    for K in [6, 12]:
        print(f"\n  K={K} layers replaced:")
        print(f"    {K} x MLP                      : {_fmt_bytes(K * mlp_total):>10}")
        print(f"    {K} x FullMix (no ckp)         : {_fmt_bytes(K * fm_total):>10}  "
              f"(delta {_fmt_bytes(K * (fm_total - mlp_total)):>10})")
        print(f"    {K} x FullMix (checkpoint)     : {_fmt_bytes(K * fm_ckp_total):>10}  "
              f"(delta {_fmt_bytes(K * (fm_ckp_total - mlp_total)):>10})")

    print("\n" + "-" * 78)
    print("R_b sweep (R_b dominates beta, the largest term):")
    print(f"{'R_b':>4} {'beta_bytes':>14} {'total_per_layer':>18} {'vs_MLP':>10}")
    N = 16 * 2048
    mlp_per = mlp_activation_bytes(d, mlp_ratio, N, dtype_bytes=2).bytes_total
    for R_b in [4, 8, 16, 24, 32]:
        cfg_rb = FullMixTuckerConfig(d=d, m=d, R_o=96, R_i=96, R_b=R_b, G=20)
        fm_rb = fullmix_activation_bytes(cfg_rb, N, dtype_bytes=2)
        beta = fm_rb.components["beta"]
        print(f"{R_b:>4} {_fmt_bytes(beta):>14} "
              f"{_fmt_bytes(fm_rb.bytes_total):>18} "
              f"{fm_rb.bytes_total / mlp_per:>10.2f}x")

    # Sanity: actually run a small forward on CPU to confirm shapes match.
    print("\n" + "-" * 78)
    print("Empirical shape check (small d=64 to keep CPU run cheap):")
    small = FullMixTuckerConfig(d=64, m=64, R_o=16, R_i=16, R_b=8, G=10)
    ffn = FullMixTuckerFFN(small)
    x = torch.randn(4, 16, 64)
    captured: dict[str, tuple] = {}
    orig = ffn._bin_and_frac

    def spy(z):  # noqa: ANN001
        bin_idx, t = orig(z)
        captured["z_shape"] = tuple(z.shape)
        captured["bin_shape"] = tuple(bin_idx.shape)
        return bin_idx, t

    ffn._bin_and_frac = spy  # type: ignore[method-assign]
    y = ffn(x)
    print(f"  input  : {tuple(x.shape)}")
    print(f"  z      : {captured['z_shape']}  (expected (N, m)={4*16, small.m})")
    print(f"  output : {tuple(y.shape)}")
    print(f"  numerically finite: {bool(torch.isfinite(y).all())}")


if __name__ == "__main__":
    main()
