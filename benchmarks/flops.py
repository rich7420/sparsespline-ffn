"""Closed-form per-token FLOP comparison: FullMix-Tucker vs MLPFFN.

Counts multiply-add-as-2-FLOPs for both forwards.  This is a static analysis
(no torch run) so it is deterministic and runs in milliseconds.

Per-token FLOPs (forward only, B1 spline => 2 active basis per scalar):

  MLP (d -> r*d -> d, no bias):
      F_mlp = 2 * d * (r * d)  +  r * d  +  2 * (r * d) * d
            = 4 * r * d^2  +  r * d            (activation: r*d adds)

  FullMix-Tucker (5-stage, B1 lookup):
      stage 1 (mixer A: d -> m):              2 * m * d
      stage 2 (B1 lookup, m scalars * R_b):   3 * m * R_b   (1 mul + 1 sub + 1 mul)
      stage 3 (V^T beta: (m, R_b) -> (R_i, R_b)): 2 * m * R_i * R_b
      stage 4 (core C contract: -> R_o):      2 * R_o * R_i * R_b
      stage 5 (U eta: R_o -> d):              2 * d * R_o
      stage 5b (gamma scalar mul):            d
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FullMixDims:
    d: int
    m: int
    R_o: int
    R_i: int
    R_b: int


def mlp_flops_per_token(d: int, mlp_ratio: int = 4) -> int:
    r = mlp_ratio
    matmul = 2 * d * (r * d) + 2 * (r * d) * d
    activation = r * d
    return matmul + activation


def fullmix_flops_per_token(dims: FullMixDims) -> dict[str, int]:
    d, m, R_o, R_i, R_b = dims.d, dims.m, dims.R_o, dims.R_i, dims.R_b
    stages = {
        "1_mixer": 2 * m * d,
        "2_b1_lookup": 3 * m * R_b,
        "3_input_contract": 2 * m * R_i * R_b,
        "4_core_contract": 2 * R_o * R_i * R_b,
        "5_readout": 2 * d * R_o,
        "5b_gamma": d,
    }
    stages["total"] = sum(stages.values())
    return stages


def fullmix_param_count(dims: FullMixDims, G: int = 20) -> int:
    L = G + 1
    return (
        dims.d * dims.m
        + dims.d * dims.R_o
        + dims.m * dims.R_i
        + L * dims.R_b
        + dims.R_o * dims.R_i * dims.R_b
        + 1
    )


def mlp_param_count(d: int, mlp_ratio: int = 4) -> int:
    return 2 * d * (mlp_ratio * d)


def _format_int(n: int) -> str:
    return f"{n:,}"


def _ratio(a: int, b: int) -> str:
    if b == 0:
        return "inf"
    return f"{a / b:.2f}x"


def main() -> None:
    print("=" * 78)
    print("FLOP analysis: FullMix-Tucker vs MLPFFN (forward, per token)")
    print("=" * 78)

    nanochat = FullMixDims(d=768, m=768, R_o=96, R_i=96, R_b=16)
    G = 20

    fm = fullmix_flops_per_token(nanochat)
    mlp = mlp_flops_per_token(nanochat.d, mlp_ratio=4)
    fm_p = fullmix_param_count(nanochat, G=G)
    mlp_p = mlp_param_count(nanochat.d, mlp_ratio=4)

    print(f"\nNanochat scale: d={nanochat.d}, m={nanochat.m}, "
          f"R_o={nanochat.R_o}, R_i={nanochat.R_i}, R_b={nanochat.R_b}, G={G}")
    print("\n  MLPFFN (d -> 4d -> d, relu_sq, no bias):")
    print(f"    params/layer       : {_format_int(mlp_p)}")
    print(f"    forward FLOPs/token: {_format_int(mlp)}")

    print("\n  FullMix-Tucker (5-stage):")
    print(f"    params/layer       : {_format_int(fm_p)}")
    print("    forward FLOPs/token (per stage):")
    for k in ["1_mixer", "2_b1_lookup", "3_input_contract",
              "4_core_contract", "5_readout", "5b_gamma"]:
        print(f"      {k:20s}: {_format_int(fm[k])}")
    print(f"    forward FLOPs/token: {_format_int(fm['total'])}")

    print("\n  Ratios (MLP / FullMix-Tucker):")
    print(f"    params       : {_ratio(mlp_p, fm_p)} smaller for FullMix")
    print(f"    forward FLOPs: {_ratio(mlp, fm['total'])} fewer for FullMix")

    print("\n" + "-" * 78)
    print("R_o sweep (other ranks fixed at R_i=96, R_b=16, m=d=768):")
    print(f"{'R_o':>6} {'params':>14} {'flops/token':>16} {'param_ratio':>12} {'flop_ratio':>12}")
    for R_o in [32, 48, 64, 96, 128, 192, 256]:
        dims = FullMixDims(d=768, m=768, R_o=R_o, R_i=96, R_b=16)
        p = fullmix_param_count(dims, G=G)
        f = fullmix_flops_per_token(dims)["total"]
        print(f"{R_o:>6} {_format_int(p):>14} {_format_int(f):>16} "
              f"{_ratio(mlp_p, p):>12} {_ratio(mlp, f):>12}")

    print("\n" + "-" * 78)
    print("MLP ratio sweep (compare FullMix to wider MLPs at d=768):")
    print(f"{'mlp_r':>6} {'mlp_params':>14} {'mlp_flops':>16} "
          f"{'vs_FM_params':>14} {'vs_FM_flops':>14}")
    for r in [2, 3, 4, 6, 8]:
        p = mlp_param_count(nanochat.d, mlp_ratio=r)
        f = mlp_flops_per_token(nanochat.d, mlp_ratio=r)
        print(f"{r:>6} {_format_int(p):>14} {_format_int(f):>16} "
              f"{_ratio(p, fm_p):>14} {_ratio(f, fm['total']):>14}")

    print("\nNote: B1 spline lookup is gather-bound, not FLOP-bound.  In "
          "practice the dominant kernel cost is memory traffic on the "
          "(N, m, R_b) beta tensor, not the 3*m*R_b adds counted here.")


if __name__ == "__main__":
    main()
