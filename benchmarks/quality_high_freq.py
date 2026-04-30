"""Quality benchmark: 1-D high-frequency function fitting.

Theoretical claim: locally-supported B1 splines (G grid intervals on a fixed
domain) can fit functions with frequency up to about G / (grid_hi - grid_lo)
per unit cycle, while a smooth-activation MLP needs O(width * depth) units
to match the same frequency.

This is the "splines should win at high frequency" benchmark.  We use d=1
(repeated to fill the configured d) so the target is genuinely 1-D and the
mixer A degenerates into a learnable identity-ish projection.

Setup:
  input  : x[..., 0] ~ Uniform[-2, 2]
  target : y_target[..., 0] = sum_k a_k * sin(omega_k * x[..., 0])
  rest of x and y are random Gaussian / zero respectively (they cost both
  models equally).

We sweep omega_max in {2, 4, 8, 16} to find each model's frequency cliff.
"""
from __future__ import annotations

import statistics
import time

import torch

from sparsespline_ffn import MLPFFN, FullMixTuckerConfig, FullMixTuckerFFN

# -- Targets --------------------------------------------------------------


def make_target(omegas: list[float], coefs: list[float]):
    def f(x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        z = x[..., 0]
        s = torch.zeros_like(z)
        for om, c in zip(omegas, coefs, strict=True):
            s = s + c * torch.sin(om * z)
        y[..., 0] = s
        return y
    return f


# -- Models ---------------------------------------------------------------


def build_fullmix(d: int, R_o: int = None, R_i: int = None, R_b: int = 8, G: int = 24) -> FullMixTuckerFFN:
    if R_o is None:
        R_o = d
    if R_i is None:
        R_i = d
    cfg = FullMixTuckerConfig(
        d=d, m=d, R_o=R_o, R_i=R_i, R_b=R_b, G=G,
        grid_lo=-2.5, grid_hi=2.5,
    )
    return FullMixTuckerFFN(cfg)


def build_matched_mlp(d: int, target_params: int) -> MLPFFN:
    best_r = 1
    for r in [1, 2, 3, 4, 6, 8, 12, 16]:
        p = 2 * d * (r * d)
        if p <= target_params:
            best_r = r
    return MLPFFN(d=d, mlp_ratio=best_r)


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# -- Train ----------------------------------------------------------------


def train(
    model: torch.nn.Module, target_fn, *, d: int, steps: int, lr: float, seed: int,
    n_train: int = 2048, n_eval: int = 1024, device: torch.device,
) -> dict:
    model = model.to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    x_train = torch.empty(n_train, d, device=device).uniform_(
        -2.0, 2.0, generator=g
    )
    x_eval = torch.empty(n_eval, d, device=device).uniform_(
        -2.0, 2.0, generator=g
    )
    y_train = target_fn(x_train)
    y_eval = target_fn(x_eval)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        pred = model(x_train)
        # Only the first channel matters; mask others to zero MSE on target=0.
        mse = (pred[..., 0] - y_train[..., 0]).pow(2).mean()
        if step % max(1, steps // 10) == 0:
            history.append(mse.item())
        mse.backward()
        opt.step()
    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        eval_mse = (model(x_eval)[..., 0] - y_eval[..., 0]).pow(2).mean().item()
    return {"history": history, "eval_mse": eval_mse, "wall_s": elapsed}


# -- Driver ---------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = 16
    seeds = [0, 1, 2]
    steps = 1500
    coefs = [0.6, 0.3, 0.15, 0.1]  # decreasing amplitudes

    fm_template = build_fullmix(d, R_b=8, G=24)
    fm_params = count_params(fm_template)
    mlp_template = build_matched_mlp(d, fm_params)
    mlp_params = count_params(mlp_template)

    print("=" * 78)
    print("Quality benchmark: 1-D high-frequency fitting")
    print(f"device={device}, d={d}, steps={steps}, seeds={seeds}")
    print(f"FullMix params : {fm_params:,}  (R_b=8, G=24, grid=[-2.5,2.5])")
    print(f"MLP params     : {mlp_params:,}  (mlp_ratio chosen for parity)")
    print(f"param ratio    : MLP/FullMix = {mlp_params / fm_params:.3f}")
    print("=" * 78)
    print(f"\n{'omega_max':>10} {'model':<12} {'eval_mse':>22} {'wall(s)':>10}")
    print("-" * 60)

    for omega_max in [2.0, 4.0, 8.0, 16.0, 32.0]:
        omegas = [omega_max / (2 ** k) for k in range(len(coefs))]
        target_fn = make_target(omegas, coefs)
        fm_evals: list[float] = []
        mlp_evals: list[float] = []
        fm_walls: list[float] = []
        mlp_walls: list[float] = []
        for seed in seeds:
            torch.manual_seed(seed)
            fm = build_fullmix(d, R_b=8, G=24)
            r_fm = train(fm, target_fn, d=d, steps=steps, lr=3e-3,
                         seed=seed, device=device)
            fm_evals.append(r_fm["eval_mse"])
            fm_walls.append(r_fm["wall_s"])

            torch.manual_seed(seed)
            mlp = build_matched_mlp(d, fm_params)
            r_mlp = train(mlp, target_fn, d=d, steps=steps, lr=3e-3,
                          seed=seed, device=device)
            mlp_evals.append(r_mlp["eval_mse"])
            mlp_walls.append(r_mlp["wall_s"])

        wins = sum(1 for a, b in zip(fm_evals, mlp_evals, strict=True) if a < b)
        fm_str = (f"{statistics.mean(fm_evals):.4e} ± "
                  f"{statistics.stdev(fm_evals) if len(fm_evals) > 1 else 0:.2e}")
        mlp_str = (f"{statistics.mean(mlp_evals):.4e} ± "
                   f"{statistics.stdev(mlp_evals) if len(mlp_evals) > 1 else 0:.2e}")
        print(f"{omega_max:>10.1f} {'FullMix':<12} {fm_str:>22} "
              f"{statistics.mean(fm_walls):>10.2f}  win={wins}/{len(seeds)}")
        print(f"{'':>10} {'MLP':<12} {mlp_str:>22} "
              f"{statistics.mean(mlp_walls):>10.2f}")
        print()

    print("Notes:")
    print("- omega_max is the highest frequency in the target sum.")
    print("- B1 splines on G=24 grid spanning [-2.5,2.5] resolve up to ~G/range = 4.8 cycles/unit.")
    print("- Theoretically FullMix's spline can outpace a same-budget smooth MLP at omega>=8.")


if __name__ == "__main__":
    main()
