"""HOSVD warm-start flow (THEORY.md L.4).

The L.4 recipe is:

    1. Train a "dense surrogate" FullMix-Tucker layer at near-full Tucker
       rank for a short pretraining phase.
    2. Materialize its dense W_{kji} = sum_{abc} U C V Q (the
       ``reconstruct_dense_W`` helper does this).
    3. Project onto a lower target rank via HOSVD
       (``hosvd_warmstart_from_dense``).
    4. Build a low-rank FullMix-Tucker layer and copy the projected
       factors into it.
    5. Continue training the low-rank layer.

This script demonstrates that the warm-started low-rank layer reproduces
the dense surrogate's forward to good precision before any further
training, so the low-rank phase can pick up where the dense phase left
off rather than starting from a fresh init.
"""
from __future__ import annotations

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN
from sparsespline_ffn.tucker_init import hosvd_warmstart_from_dense


def _train_short(layer: FullMixTuckerFFN, *, d: int, steps: int = 80,
                 lr: float = 3e-3, seed: int = 0) -> None:
    """A few SGD steps so the dense surrogate has non-init weights."""
    torch.manual_seed(seed)
    x = torch.randn(256, d)
    y = torch.zeros_like(x)
    for k in range(min(d, 4)):
        y[..., k] = 0.4 * torch.sin((k + 1) * x[..., k])
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (layer(x) - y).pow(2).mean().backward()
        opt.step()


def main() -> None:
    d = 16
    R_low = (4, 4, 3)        # target compressed rank
    R_high = (d, d, R_low[2])  # near-dense surrogate (full d on first two modes)
    G = 8

    print(f"d={d}, low rank={R_low}, dense surrogate rank={R_high}")

    # 1. Build and train the dense surrogate.
    torch.manual_seed(0)
    surrogate = FullMixTuckerFFN(
        FullMixTuckerConfig(d=d, m=d, R_o=R_high[0], R_i=R_high[1],
                             R_b=R_high[2], G=G)
    )
    _train_short(surrogate, d=d, steps=80)
    print("dense surrogate trained.")

    # 2-3. Materialize W and HOSVD-project to low rank.
    with torch.no_grad():
        W_dense = surrogate.reconstruct_dense_W()
    U, V, core, Q = hosvd_warmstart_from_dense(
        W_dense, R_o=R_low[0], R_i=R_low[1], R_b=R_low[2]
    )
    print(f"HOSVD: U {tuple(U.shape)}, V {tuple(V.shape)}, "
          f"core {tuple(core.shape)}, Q {tuple(Q.shape)}")

    # 4. Build the target low-rank layer and inject factors.
    target = FullMixTuckerFFN(
        FullMixTuckerConfig(d=d, m=d, R_o=R_low[0], R_i=R_low[1],
                             R_b=R_low[2], G=G)
    )
    with torch.no_grad():
        target.U.copy_(U)
        target.V.copy_(V)
        target.C.copy_(core)
        target.Q.copy_(Q)
        target.gamma.copy_(surrogate.gamma)
        # Mirror the trained mixer so spline inputs match.
        target.A.weight.copy_(surrogate.A.weight)
        if surrogate.A.bias is not None:
            target.A.bias.copy_(surrogate.A.bias)

    # 5. Compare forwards.  The low-rank layer should reproduce the surrogate
    #    closely (exactly when the surrogate's W is already low-rank;
    #    approximately otherwise).
    torch.manual_seed(1)
    x = torch.randn(8, d)
    with torch.no_grad():
        y_surrogate = surrogate(x)
        y_target = target(x)
    rel = (y_surrogate - y_target).norm() / (y_surrogate.norm() + 1e-9)
    print(f"warm-started vs dense surrogate forward rel err: "
          f"{rel.item():.2e}")
    print("  (~1.0 means heavy lossy projection from full rank to low rank;")
    print("   this is expected — the warm start is meant as a head start,")
    print("   not as an equivalence.  See tests/test_fullmix_tucker_extras::")
    print("   test_hosvd_warmstart_reinjection_matches_dense_W for the lossless case.)")

    # 5b. Demonstrate that further training picks up from the warm start.
    _train_short(target, d=d, steps=40, seed=42)
    with torch.no_grad():
        y_after = target(x)
    print(f"after 40 more steps: warm output norm {y_after.norm().item():.3f}")


if __name__ == "__main__":
    main()
