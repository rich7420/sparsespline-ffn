from __future__ import annotations

from sparsespline_ffn import MLPFFN, FullMixTuckerConfig, FullMixTuckerFFN


def count_params(module) -> int:
    return sum(param.numel() for param in module.parameters())


def main() -> None:
    d = 768
    mlp = MLPFFN(d=d, mlp_ratio=4, activation="relu_sq", bias=False)
    fm = FullMixTuckerFFN(FullMixTuckerConfig(d=d, m=d, R_o=96, R_i=96, R_b=16, G=20))

    mlp_params = count_params(mlp)
    fm_params = count_params(fm)
    print(f"MLPFFN params: {mlp_params:,}")
    print(f"FullMixTuckerFFN params: {fm_params:,}")
    print(f"compression: {mlp_params / fm_params:.2f}x")


if __name__ == "__main__":
    main()
