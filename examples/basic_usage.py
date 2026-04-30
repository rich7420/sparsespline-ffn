from __future__ import annotations

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN, build_ffn


def main() -> None:
    torch.manual_seed(0)

    cfg = FullMixTuckerConfig(d=32, m=32, R_o=16, R_i=16, R_b=4, G=8)
    ffn = FullMixTuckerFFN(cfg)
    x = torch.randn(2, 5, 32)
    y = ffn(x)
    print(f"direct module: {tuple(x.shape)} -> {tuple(y.shape)}")

    scheduled = build_ffn(
        ffn_type="fullmix_tucker",
        d=32,
        layer_idx=4,
        num_layers=6,
        schedule="late",
        R_o=16,
        R_i=16,
        R_b=4,
        G=8,
    )
    y2 = scheduled(x)
    print(f"scheduled module: {scheduled.ffn_type_effective}, {tuple(y2.shape)}")


if __name__ == "__main__":
    main()
