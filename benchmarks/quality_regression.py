"""Quality benchmark: parity-budget regression.

For a fair comparison we match parameter counts: pick a FullMix-Tucker config,
then choose an MLP mlp_ratio so MLP params <= FullMix params.  Both fit the
same synthetic targets via Adam.  Track final MSE across seeds.

Targets (per-token, deterministic functions of x):
  smooth     : sin(x[0]) + 0.5 * cos(x[1])
  multimodal : tanh(2x[0])*tanh(2x[1]) + 0.3 * x[2]^2
  highfreq   : sum_k a_k sin(omega_k * x[0])  with omega in {1, 3, 5, 7}
  piecewise  : x[0] if x[0] > 0 else -x[0]^2  (kink at 0)

For each target we run 3 seeds and report mean +/- std final MSE for both
models, plus the win-rate (FullMix MSE < MLP MSE).
"""
from __future__ import annotations

import statistics
import time

import torch

from sparsespline_ffn import MLPFFN, FullMixTuckerConfig, FullMixTuckerFFN

# -- Targets --------------------------------------------------------------


def target_smooth(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    y[..., 0] = torch.sin(x[..., 0]) + 0.5 * torch.cos(x[..., 1])
    return y


def target_multimodal(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    y[..., 0] = torch.tanh(2 * x[..., 0]) * torch.tanh(2 * x[..., 1]) + 0.3 * x[..., 2] ** 2
    return y


def target_highfreq(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    omegas = [1.0, 3.0, 5.0, 7.0]
    coefs = [0.5, 0.3, 0.2, 0.1]
    for om, c in zip(omegas, coefs, strict=True):
        y[..., 0] = y[..., 0] + c * torch.sin(om * x[..., 0])
    return y


def target_piecewise(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    x0 = x[..., 0]
    y[..., 0] = torch.where(x0 > 0, x0, -x0 ** 2)
    return y


TARGETS = {
    "smooth": target_smooth,
    "multimodal": target_multimodal,
    "highfreq": target_highfreq,
    "piecewise": target_piecewise,
}


# -- Models ---------------------------------------------------------------


def build_fullmix(d: int) -> FullMixTuckerFFN:
    cfg = FullMixTuckerConfig(d=d, m=d, R_o=d, R_i=d, R_b=4, G=12)
    return FullMixTuckerFFN(cfg)


def build_matched_mlp(d: int, target_params: int) -> MLPFFN:
    """Pick the largest mlp_ratio so MLP params <= target_params."""
    best_r = 1
    for r in [1, 2, 3, 4, 6, 8]:
        p = 2 * d * (r * d)
        if p <= target_params:
            best_r = r
    return MLPFFN(d=d, mlp_ratio=best_r)


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# -- Training loop --------------------------------------------------------


def train_to_target(
    model: torch.nn.Module,
    target_fn,
    *,
    d: int,
    n_train: int = 512,
    n_eval: int = 256,
    steps: int = 600,
    lr: float = 3e-3,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    x_train = torch.randn(n_train, d, device=device, generator=g)
    x_eval = torch.randn(n_eval, d, device=device, generator=g)
    y_train = target_fn(x_train)
    y_eval = target_fn(x_eval)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    initial_train_mse = float("inf")
    history: list[float] = []
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        pred = model(x_train)
        mse = (pred - y_train).pow(2).mean()
        if step == 0:
            initial_train_mse = mse.item()
        if step % max(1, steps // 20) == 0:
            history.append(mse.item())
        mse.backward()
        opt.step()
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        eval_mse = (model(x_eval) - y_eval).pow(2).mean().item()
    return {
        "initial_train_mse": initial_train_mse,
        "final_train_mse": history[-1] if history else float("nan"),
        "eval_mse": eval_mse,
        "history": history,
        "wall_s": elapsed,
    }


# -- Driver ---------------------------------------------------------------


def _summary(metric: list[float]) -> str:
    if len(metric) <= 1:
        return f"{metric[0]:.4e}"
    return f"{statistics.mean(metric):.4e} ± {statistics.stdev(metric):.2e}"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = 32
    seeds = [0, 1, 2]
    steps = 600

    fm_template = build_fullmix(d)
    fm_params = count_params(fm_template)
    mlp_template = build_matched_mlp(d, fm_params)
    mlp_params = count_params(mlp_template)

    print("=" * 78)
    print("Quality benchmark: parity-budget regression on synthetic targets")
    print(f"device={device}, d={d}, steps={steps}, seeds={seeds}")
    print(f"FullMix params : {fm_params:,}  (R_o=R_i={d}, R_b=4, G=12)")
    print(f"MLP params     : {mlp_params:,}  (mlp_ratio chosen for parity)")
    print(f"param ratio (MLP/FM): {mlp_params / fm_params:.3f}")
    print("=" * 78)

    print(f"\n{'target':<12} {'model':<12} "
          f"{'final_train_mse':>22} {'eval_mse':>22} {'wall(s)':>10} {'win':>6}")
    print("-" * 90)

    for tname, tfn in TARGETS.items():
        fm_evals: list[float] = []
        mlp_evals: list[float] = []
        fm_walls: list[float] = []
        mlp_walls: list[float] = []
        fm_finals: list[float] = []
        mlp_finals: list[float] = []
        for seed in seeds:
            torch.manual_seed(seed)
            fm = build_fullmix(d)
            r_fm = train_to_target(
                fm, tfn, d=d, steps=steps, seed=seed, device=device
            )
            fm_evals.append(r_fm["eval_mse"])
            fm_walls.append(r_fm["wall_s"])
            fm_finals.append(r_fm["final_train_mse"])

            torch.manual_seed(seed)
            mlp = build_matched_mlp(d, fm_params)
            r_mlp = train_to_target(
                mlp, tfn, d=d, steps=steps, seed=seed, device=device
            )
            mlp_evals.append(r_mlp["eval_mse"])
            mlp_walls.append(r_mlp["wall_s"])
            mlp_finals.append(r_mlp["final_train_mse"])

        wins = sum(1 for a, b in zip(fm_evals, mlp_evals, strict=True) if a < b)
        print(f"{tname:<12} {'FullMix':<12} {_summary(fm_finals):>22} "
              f"{_summary(fm_evals):>22} {statistics.mean(fm_walls):>10.2f} {wins:>3}/{len(seeds)}")
        print(f"{'':<12} {'MLP-matched':<12} {_summary(mlp_finals):>22} "
              f"{_summary(mlp_evals):>22} {statistics.mean(mlp_walls):>10.2f}")
        print()

    print("Notes:")
    print("- 'win' counts seeds where FullMix eval MSE < MLP eval MSE (lower better).")
    print("- Both models use Adam with lr=3e-3, identical schedule, matched params.")
    print("- 'highfreq' is where locally-supported splines should have an edge.")
    print("- 'piecewise' tests fitting a non-smooth kink at zero.")


if __name__ == "__main__":
    main()
