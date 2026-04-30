"""Quality benchmark: convergence speed.

For each (target, model) pair, fit until convergence and report:
  - steps_to_X_pct : steps needed to reach X% of best-loss-seen
  - best_mse       : best train MSE seen during training
  - eval_mse       : held-out eval MSE at the end

This separates *can it learn at all* from *how fast does it learn*.

Targets:
  spline_friendly : sum_j sin(omega_j * x[..., j])  (separable in input dims)
  product         : x[..., 0] * x[..., 1] * tanh(x[..., 2])  (low-rank cross)
  square_sum      : sum_j x[..., j]^2  (rotationally symmetric)
"""
from __future__ import annotations

import statistics
import time

import torch

from sparsespline_ffn import MLPFFN, FullMixTuckerConfig, FullMixTuckerFFN

# -- Targets --------------------------------------------------------------


def target_spline_friendly(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    omegas = [1.0, 2.0, 3.0, 4.0]
    for j, om in enumerate(omegas):
        if j < x.shape[-1]:
            y[..., 0] = y[..., 0] + 0.3 * torch.sin(om * x[..., j])
    return y


def target_product(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    if x.shape[-1] >= 3:
        y[..., 0] = x[..., 0] * x[..., 1] * torch.tanh(x[..., 2])
    return y


def target_square_sum(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    y[..., 0] = 0.1 * x.pow(2).sum(dim=-1)
    return y


TARGETS = {
    "spline_friendly": target_spline_friendly,
    "product":         target_product,
    "square_sum":      target_square_sum,
}


# -- Models ---------------------------------------------------------------


def build_fullmix(d: int) -> FullMixTuckerFFN:
    cfg = FullMixTuckerConfig(d=d, m=d, R_o=d // 2, R_i=d // 2, R_b=8, G=20,
                              grid_lo=-3.0, grid_hi=3.0)
    return FullMixTuckerFFN(cfg)


def build_matched_mlp(d: int, target_params: int) -> MLPFFN:
    best_r = 1
    for r in [1, 2, 3, 4]:
        if 2 * d * (r * d) <= target_params:
            best_r = r
    return MLPFFN(d=d, mlp_ratio=best_r)


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# -- Train ----------------------------------------------------------------


def train_with_history(
    model: torch.nn.Module, target_fn, *, d: int, steps: int, lr: float,
    n_train: int, n_eval: int, seed: int, device: torch.device,
) -> dict:
    model = model.to(device).train()
    g = torch.Generator(device=device).manual_seed(seed + 11)
    x_train = torch.randn(n_train, d, device=device, generator=g)
    x_eval = torch.randn(n_eval, d, device=device, generator=g)
    y_train = target_fn(x_train)
    y_eval = target_fn(x_eval)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_history: list[float] = []
    t0 = time.perf_counter()
    for _step in range(steps):
        opt.zero_grad()
        pred = model(x_train)
        mse = (pred - y_train).pow(2).mean()
        train_history.append(mse.item())
        mse.backward()
        opt.step()
    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        eval_mse = (model(x_eval) - y_eval).pow(2).mean().item()
    return {
        "history": train_history,
        "eval_mse": eval_mse,
        "best_mse": min(train_history),
        "wall_s": elapsed,
    }


def steps_to_reach(history: list[float], threshold: float) -> int:
    for i, v in enumerate(history):
        if v <= threshold:
            return i
    return len(history)


# -- Driver ---------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = 32
    seeds = [0, 1, 2]
    steps = 1500

    fm_template = build_fullmix(d)
    fm_p = count_params(fm_template)
    mlp_template = build_matched_mlp(d, fm_p)
    mlp_p = count_params(mlp_template)

    print("=" * 78)
    print("Quality benchmark: convergence speed")
    print(f"device={device}, d={d}, steps={steps}, seeds={seeds}")
    print(f"FullMix params : {fm_p:,}  (R_o=R_i=d/2, R_b=8, G=20)")
    print(f"MLP params     : {mlp_p:,}")
    print("=" * 78)

    print(f"\n{'target':<18} {'model':<10} {'best_train_mse':>16} "
          f"{'eval_mse':>14} {'steps@best/2':>14} {'steps@best/10':>14}")
    print("-" * 90)

    for tname, tfn in TARGETS.items():
        for mname, builder in [
            ("FullMix", lambda: build_fullmix(d)),
            ("MLP", lambda: build_matched_mlp(d, fm_p)),
        ]:
            best_list: list[float] = []
            eval_list: list[float] = []
            steps_half: list[int] = []
            steps_tenth: list[int] = []
            for seed in seeds:
                torch.manual_seed(seed)
                model = builder()
                r = train_with_history(
                    model, tfn, d=d, steps=steps, lr=3e-3,
                    n_train=1024, n_eval=512, seed=seed, device=device,
                )
                best_list.append(r["best_mse"])
                eval_list.append(r["eval_mse"])
                steps_half.append(steps_to_reach(r["history"], r["best_mse"] * 2))
                steps_tenth.append(steps_to_reach(r["history"], r["best_mse"] * 10))

            print(f"{tname:<18} {mname:<10} "
                  f"{statistics.mean(best_list):>16.4e} "
                  f"{statistics.mean(eval_list):>14.4e} "
                  f"{int(statistics.mean(steps_half)):>14d} "
                  f"{int(statistics.mean(steps_tenth)):>14d}")
        print()

    print("Notes:")
    print("- best_train_mse: lowest MSE seen during training (capacity proxy).")
    print("- eval_mse: MSE on held-out inputs at the end.")
    print("- steps@best/2: first step where train MSE <= 2 * best_train_mse "
          "(i.e., \"close to convergence\"). Lower = converges faster.")
    print("- steps@best/10: first step where MSE <= 10 * best (\"on track\").")


if __name__ == "__main__":
    main()
