"""Diagnostic CLI: ``python -m sparsespline_ffn``.

Prints version, package paths, kernel availability, and a quick MAC /
parameter count for a configurable FullMix-Tucker layer.  Intended for
sanity-checking installs and bug reports — not for benchmarking
(see ``benchmarks/`` for that).
"""
from __future__ import annotations

import argparse
import platform
import sys
from importlib import metadata

import torch

import sparsespline_ffn
from sparsespline_ffn import (
    FullMixTuckerConfig,
    FullMixTuckerFFN,
    is_kernel_available,
)


def _safe_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def cmd_info(args: argparse.Namespace) -> int:
    """Print runtime + install info."""
    print("SparseSpline-FFN")
    print(f"  version            : {sparsespline_ffn.__version__}")
    print(f"  package path       : {sparsespline_ffn.__file__}")
    print()
    print("Runtime")
    print(f"  python             : {platform.python_version()}  "
          f"({sys.executable})")
    print(f"  torch              : {torch.__version__}")
    print(f"  cuda available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda device        : {torch.cuda.get_device_name(0)}")
        print(f"  cuda capability    : "
              f"{'.'.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"  triton             : {_safe_version('triton')}")
    print(f"  is_kernel_available: {is_kernel_available()}")
    return 0


def cmd_check_kernel(args: argparse.Namespace) -> int:
    """Build a small layer with ``use_kernel='required'`` and run a forward.

    Exit code 0 if the kernel actually runs end-to-end; 1 otherwise.  A
    quick health check for ``pip install sparsespline-ffn[cuda]`` setups.
    """
    if not is_kernel_available():
        print("kernel unavailable (need CUDA + triton).")
        print(f"  cuda available     : {torch.cuda.is_available()}")
        print(f"  triton             : {_safe_version('triton')}")
        return 1

    cfg = FullMixTuckerConfig(d=64, m=64, R_o=16, R_i=16, R_b=8, G=12,
                              use_kernel="required")
    ffn = FullMixTuckerFFN(cfg).cuda().float()
    x = torch.randn(8, cfg.d, device="cuda", requires_grad=True)
    y = ffn(x)
    y.pow(2).sum().backward()
    if not torch.isfinite(y).all():
        print("FAIL: forward produced non-finite outputs.")
        return 1
    print("OK: kernel forward + backward ran on CUDA "
          f"(d={cfg.d}, R_o={cfg.R_o}, R_b={cfg.R_b}).")
    print(f"  kernel_will_run    : {ffn.kernel_will_run(x)}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Print parameter / MAC counts for a configurable layer.

    Useful for capacity planning before changing rank: ``python -m
    sparsespline_ffn config --d 768 --R_o 96 --R_i 96 --R_b 16``.
    """
    cfg = FullMixTuckerConfig(
        d=args.d, m=args.m or args.d,
        R_o=args.R_o, R_i=args.R_i, R_b=args.R_b, G=args.G,
    )
    ffn = FullMixTuckerFFN(cfg)
    L = cfg.G + 1
    params = sum(p.numel() for p in ffn.parameters())
    macs = (
        cfg.d * cfg.m
        + cfg.m * cfg.R_b
        + cfg.m * cfg.R_i * cfg.R_b
        + cfg.R_o * cfg.R_i * cfg.R_b
        + cfg.d * cfg.R_o
    )
    mlp_params = 2 * cfg.d * (4 * cfg.d)
    mlp_macs = 2 * cfg.d * (4 * cfg.d) + 2 * (4 * cfg.d) * cfg.d
    print(f"FullMix-Tucker layer: d={cfg.d}, m={cfg.m}, "
          f"R=({cfg.R_o},{cfg.R_i},{cfg.R_b}), G={cfg.G}, L={L}")
    print(f"  params/layer    : {params:,}")
    print(f"  forward MACs/tok: {macs:,}")
    print("  vs MLP (4d, no bias):")
    print(f"    MLP params    : {mlp_params:,}  "
          f"(FullMix is {mlp_params / max(params, 1):.2f}x smaller)")
    print(f"    MLP MACs/tok  : {mlp_macs:,}  "
          f"(FullMix is {mlp_macs / max(macs, 1):.2f}x cheaper)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sparsespline_ffn",
        description="SparseSpline-FFN diagnostic CLI.",
    )
    sub = ap.add_subparsers(dest="cmd")

    p_info = sub.add_parser("info", help="print version + runtime info "
                                          "(default if no subcommand given)")
    p_info.set_defaults(func=cmd_info)

    p_check = sub.add_parser("check-kernel",
                             help="run the Triton kernel end-to-end")
    p_check.set_defaults(func=cmd_check_kernel)

    p_cfg = sub.add_parser("config",
                           help="print parameter / MAC counts for a config")
    p_cfg.add_argument("--d", type=int, default=768)
    p_cfg.add_argument("--m", type=int, default=None)
    p_cfg.add_argument("--R_o", type=int, default=96)
    p_cfg.add_argument("--R_i", type=int, default=96)
    p_cfg.add_argument("--R_b", type=int, default=16)
    p_cfg.add_argument("--G", type=int, default=20)
    p_cfg.set_defaults(func=cmd_config)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        return cmd_info(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
