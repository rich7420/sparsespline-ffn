"""Quality benchmark: L.4 HOSVD warm-start vs cold-start.

THEORY.md L.4 recommends a "train dense W -> SVD-init Tucker factors ->
switch to factored kernel" sequence to de-risk Tucker's non-convex
optimization.  This benchmark measures whether the warm-start actually
helps in practice.

Procedure:
  COLD: build a FullMixTuckerFFN with the standard variance-preserving
        init, train for `steps_total` SGD steps.
  WARM: build a "dense" surrogate (a FullMixTuckerFFN at high rank, near
        full Tucker), train it for `steps_pretrain` steps; use HOSVD on
        its dense W to initialize a low-rank FullMixTuckerFFN; train the
        low-rank one for the remaining (steps_total - steps_pretrain) steps.

We compare final eval_mse and convergence trajectory (loss at fixed
checkpoints).  Hypothesis (L.4): warm-start gives a head start (lower
loss after the same total steps).

Auto-detects CUDA; runs in fp32 because HOSVD via torch.linalg.svd is
fp32-friendly.
"""
from __future__ import annotations

import statistics
import time

import torch

from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN
from sparsespline_ffn.tucker_init import hosvd_warmstart_from_dense


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def target(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    d = x.shape[-1]
    for k in range(min(d, 6)):
        y[..., k] = 0.4 * torch.sin((k + 1) * x[..., k])
    return y


def train_one(model, *, d, steps, lr, seed, device, log_every: int = 100):
    model = model.to(device).float()
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1024, d, device=device, generator=g)
    y = target(x)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[tuple[int, float]] = []
    for step in range(steps):
        opt.zero_grad()
        mse = (model(x) - y).pow(2).mean()
        if step % log_every == 0 or step == steps - 1:
            history.append((step, mse.item()))
        mse.backward()
        opt.step()
    with torch.no_grad():
        x_eval = torch.randn(1024, d, device=device, generator=g)
        y_eval = target(x_eval)
        eval_mse = (model(x_eval) - y_eval).pow(2).mean().item()
    return history, eval_mse


def main():
    device = _device()
    d = 16
    R_o_low = 4
    R_i_low = 4
    R_b_low = 4
    R_o_high = d  # near-dense surrogate
    R_i_high = d
    R_b_high = R_b_low  # match basis-rank so HOSVD truncates only spatial dims
    G = 10
    seeds = [0, 1, 2]
    steps_total = 800
    steps_pretrain = 200
    lr = 3e-3

    print("=" * 78)
    print("Quality benchmark: L.4 HOSVD warm-start vs cold-start")
    print(f"device={device}, d={d}")
    print(f"low rank (target compressed FFN) : "
          f"R=({R_o_low},{R_i_low},{R_b_low})")
    print(f"high rank (dense pre-train)      : "
          f"R=({R_o_high},{R_i_high},{R_b_high})")
    print(f"total steps={steps_total}, pretrain steps for WARM={steps_pretrain}")
    print("=" * 78)

    cold_evals: list[float] = []
    warm_evals: list[float] = []
    cold_history: list[list[tuple[int, float]]] = []
    warm_history: list[list[tuple[int, float]]] = []

    t0 = time.perf_counter()
    for seed in seeds:
        torch.manual_seed(seed)
        cfg_low = FullMixTuckerConfig(
            d=d, m=d, R_o=R_o_low, R_i=R_i_low, R_b=R_b_low, G=G
        )
        # ---- COLD ----
        cold = FullMixTuckerFFN(cfg_low)
        ch, ce = train_one(cold, d=d, steps=steps_total, lr=lr,
                           seed=seed, device=device, log_every=100)
        cold_evals.append(ce)
        cold_history.append(ch)

        # ---- WARM ----
        torch.manual_seed(seed + 1000)
        cfg_high = FullMixTuckerConfig(
            d=d, m=d, R_o=R_o_high, R_i=R_i_high, R_b=R_b_high, G=G
        )
        dense_surrogate = FullMixTuckerFFN(cfg_high)
        _ph, _pe = train_one(dense_surrogate, d=d, steps=steps_pretrain,
                             lr=lr, seed=seed, device=device, log_every=100)

        with torch.no_grad():
            W_dense = dense_surrogate.reconstruct_dense_W().float().cpu()
        U, V, core, Q = hosvd_warmstart_from_dense(
            W_dense, R_o=R_o_low, R_i=R_i_low, R_b=R_b_low
        )

        warm = FullMixTuckerFFN(cfg_low)
        with torch.no_grad():
            warm.U.copy_(U.to(warm.U.device))
            warm.V.copy_(V.to(warm.V.device))
            warm.C.copy_(core.to(warm.C.device))
            warm.Q.copy_(Q.to(warm.Q.device))
            # Carry the trained mixer too -- otherwise we throw away half of
            # what the dense surrogate learned.
            warm.A.weight.copy_(dense_surrogate.A.weight)
            if dense_surrogate.A.bias is not None and warm.A.bias is not None:
                warm.A.bias.copy_(dense_surrogate.A.bias)
            warm.gamma.copy_(dense_surrogate.gamma)

        wh, we = train_one(warm, d=d, steps=steps_total - steps_pretrain,
                           lr=lr, seed=seed, device=device, log_every=100)
        warm_evals.append(we)
        warm_history.append(wh)

    elapsed = time.perf_counter() - t0

    print(f"\n{'config':<14} {'eval_mse_mean':>16} {'eval_mse_std':>14} "
          f"{'wall(s)':>10}")
    print("-" * 60)
    print(f"{'COLD':<14} {statistics.mean(cold_evals):>16.4e} "
          f"{statistics.stdev(cold_evals):>14.2e} {elapsed:>10.2f}")
    print(f"{'WARM (HOSVD)':<14} {statistics.mean(warm_evals):>16.4e} "
          f"{statistics.stdev(warm_evals):>14.2e}")

    # WARM history is shifted: pretrain spent steps_pretrain + cold ran from 0.
    # Compare last point.
    print("\nFinal eval_mse comparison:")
    if statistics.mean(warm_evals) < statistics.mean(cold_evals):
        speedup = statistics.mean(cold_evals) / max(
            statistics.mean(warm_evals), 1e-12
        )
        print(f"  WARM beats COLD by {speedup:.2f}x final eval_mse  "
              "=> L.4 thesis SUPPORTED at this scale")
    else:
        print("  WARM does not beat COLD final eval_mse — could be that")
        print("  the cold init is already strong at this small scale or that")
        print("  the surrogate's trained dense W did not generalize well.")

    print("\nLoss trajectories (mean across seeds, log_every=100):")

    def avg_history(hs: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
        n_pts = min(len(h) for h in hs)
        out = []
        for i in range(n_pts):
            step = hs[0][i][0]
            mean = statistics.mean(h[i][1] for h in hs)
            out.append((step, mean))
        return out

    cold_avg = avg_history(cold_history)
    warm_avg = avg_history(warm_history)
    print(f"  {'step (cold)':>14} {'cold_mse':>14}    "
          f"{'step (warm)':>14} {'warm_mse':>14}")
    n = max(len(cold_avg), len(warm_avg))
    for i in range(n):
        c = cold_avg[i] if i < len(cold_avg) else None
        w = warm_avg[i] if i < len(warm_avg) else None
        c_step = f"{c[0]:>14}" if c else " " * 14
        c_mse = f"{c[1]:>14.4e}" if c else " " * 14
        w_step = f"{w[0]+steps_pretrain:>14}" if w else " " * 14  # absolute
        w_mse = f"{w[1]:>14.4e}" if w else " " * 14
        print(f"  {c_step} {c_mse}    {w_step} {w_mse}")


if __name__ == "__main__":
    main()
