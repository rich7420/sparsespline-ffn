"""Quality benchmark: T_direct (no mixer) vs T_mixer at parity.

THEORY.md M.5 lists "mixer-matters" as the primary diagnostic ablation.
The claim (Part I.1) is that T_direct fires splines on canonical input
dims while T_mixer fires them on learned linear combinations of inputs
(MLP's "learned half-space" inductive bias).  T_mixer should help on
tasks whose target depends on linear combinations of inputs, and tie
T_direct on tasks that are already separable per-dim.

Two contrasting targets:
  separable: y[k] = sin(x[k]) — separable in canonical coordinates.
              T_direct should match T_mixer.
  rotational: y[0] = sin(0.7*x[0] + 0.7*x[1]) — needs a learned mixer
              direction.  T_mixer should beat T_direct.

Auto-detects CUDA; bf16 on CUDA, fp32 on CPU.
"""
from __future__ import annotations

import statistics
import time

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN


def _device_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def target_separable(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    d = x.shape[-1]
    for k in range(min(d, 4)):
        y[..., k] = torch.sin(x[..., k])
    return y


def target_rotational(x: torch.Tensor) -> torch.Tensor:
    """Target depends on linear combinations of inputs — direct-d KAN
    cannot fire knots along this direction without a mixer."""
    y = torch.zeros_like(x)
    if x.shape[-1] >= 2:
        y[..., 0] = torch.sin(0.7 * x[..., 0] + 0.7 * x[..., 1])
        y[..., 1] = torch.sin(0.7 * x[..., 0] - 0.7 * x[..., 1])
    if x.shape[-1] >= 4:
        y[..., 2] = torch.tanh(0.5 * x[..., 2] + 0.5 * x[..., 3])
    return y


TARGETS = {
    "separable":  target_separable,
    "rotational": target_rotational,
}


def train(model, target_fn, *, d, steps, lr, seed, device, dtype):
    model = model.to(device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(2048, d, device=device, dtype=dtype, generator=g)
    y = target_fn(x).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (model(x) - y).pow(2).mean().backward()
        opt.step()
    with torch.no_grad():
        x_eval = torch.randn(1024, d, device=device, dtype=dtype, generator=g)
        y_eval = target_fn(x_eval).to(dtype)
        eval_mse = (model(x_eval) - y_eval).pow(2).mean().item()
    return eval_mse


def main():
    device, dtype = _device_dtype()
    d = 16
    seeds = [0, 1, 2]
    steps = 700
    lr = 3e-3

    print("=" * 78)
    print("Quality benchmark: M.5 mixer-matters ablation (T_direct vs T_mixer)")
    print(f"device={device}, dtype={dtype}, d={d}, steps={steps}, "
          f"seeds={seeds}")
    print("=" * 78)

    # Same per-layer rank for both topologies; only difference is use_mixer.
    R_o = R_i = 8
    R_b = 4
    G = 12

    for tname, tfn in TARGETS.items():
        print(f"\n--- target = {tname} ---")
        results: dict[str, list[float]] = {"T_mixer": [], "T_direct": []}
        for use_mixer in (True, False):
            label = "T_mixer" if use_mixer else "T_direct"
            t0 = time.perf_counter()
            for seed in seeds:
                torch.manual_seed(seed)
                cfg = FullMixTuckerConfig(
                    d=d, m=d, R_o=R_o, R_i=R_i, R_b=R_b, G=G,
                    use_mixer=use_mixer,
                )
                fm = FullMixTuckerFFN(cfg)
                ev = train(fm, tfn, d=d, steps=steps, lr=lr, seed=seed,
                           device=device, dtype=dtype)
                results[label].append(ev)
            wall = time.perf_counter() - t0
            params = sum(p.numel() for p in fm.parameters())
            ev_m = statistics.mean(results[label])
            ev_s = statistics.stdev(results[label]) if len(results[label]) > 1 else 0.0
            print(f"  {label:<10} params={params:,}  "
                  f"eval_mse={ev_m:.4e} ± {ev_s:.2e}  wall={wall:.2f}s")

        m = statistics.mean(results["T_mixer"])
        d_ = statistics.mean(results["T_direct"])
        print(f"  ratio T_direct / T_mixer : {d_ / max(m, 1e-12):.2f}")
        if tname == "rotational":
            if m < d_:
                print("  => T_mixer beats T_direct on rotational target  "
                      "(M.5 mixer-matters CONFIRMED)")
            else:
                print("  => T_direct ties or wins — surprising, suggests")
                print("     mixer is not pulling weight at this scale")
        else:  # separable
            if d_ <= m * 1.10:
                print("  => T_direct ties T_mixer on separable target  "
                      "(expected per Part I.1)")
            else:
                print("  => T_direct loses to T_mixer even on separable target —")
                print("     direct-d KAN may have an init disadvantage")


if __name__ == "__main__":
    main()
