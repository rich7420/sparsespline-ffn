"""Quality benchmark: distillation from a wide MLP teacher.

A *random* MLP teacher (with mlp_ratio = 4) defines the target mapping x -> y.
We distill it into:
  - FullMix-Tucker student (matched-or-fewer params)
  - MLP student (matched params, smaller mlp_ratio)

If FullMix's Tucker compression is recoverable representation, it should
distill the teacher at MSE comparable to the matched MLP.  If FullMix has
*better* representational fit per parameter, it should beat the matched MLP.

This is the closest empirical analogue to "can FullMix represent what MLP
represents" without doing a full LM pretraining run.

Setup:
  d=64 (small enough for fast Adam, large enough for nontrivial randomness)
  teacher: MLPFFN(d=64, mlp_ratio=4) random init  -> 32,768 params
  student_FullMix: chosen R_o, R_i, R_b, G        -> ~params
  student_MLP    : MLPFFN with mlp_ratio s.t. params <= teacher params
"""
from __future__ import annotations

import statistics
import time

import torch

from sparsespline_ffn import MLPFFN, FullMixTuckerConfig, FullMixTuckerFFN


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def make_teacher(d: int, seed: int) -> MLPFFN:
    torch.manual_seed(1000 + seed)
    return MLPFFN(d=d, mlp_ratio=4)


def make_student_fullmix(d: int) -> FullMixTuckerFFN:
    cfg = FullMixTuckerConfig(d=d, m=d, R_o=d // 2, R_i=d // 2, R_b=8, G=20)
    return FullMixTuckerFFN(cfg)


def make_student_mlp_matched(d: int, target_params: int) -> MLPFFN:
    """Largest mlp_ratio that fits within target_params."""
    best_r = 1
    for r in [1, 2, 3, 4]:
        if 2 * d * (r * d) <= target_params:
            best_r = r
    return MLPFFN(d=d, mlp_ratio=best_r)


def make_student_mlp_smaller(d: int, target_params: int) -> MLPFFN:
    """For comparison: an MLP that uses *exactly* the same param count we'd
    spend on FullMix, with mlp_ratio capped at 1."""
    return MLPFFN(d=d, mlp_ratio=1)


def distill(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    *,
    d: int,
    steps: int,
    lr: float,
    n_train: int,
    n_eval: int,
    seed: int,
    device: torch.device,
) -> dict:
    teacher = teacher.to(device).eval()
    student = student.to(device).train()
    for p in teacher.parameters():
        p.requires_grad_(False)

    g = torch.Generator(device=device).manual_seed(seed + 7)
    x_train = torch.randn(n_train, d, device=device, generator=g)
    x_eval = torch.randn(n_eval, d, device=device, generator=g)
    with torch.no_grad():
        y_train = teacher(x_train)
        y_eval = teacher(x_eval)

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    history: list[float] = []
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        pred = student(x_train)
        mse = (pred - y_train).pow(2).mean()
        if step % max(1, steps // 10) == 0:
            history.append(mse.item())
        mse.backward()
        opt.step()
    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        eval_mse = (student(x_eval) - y_eval).pow(2).mean().item()
    return {"history": history, "eval_mse": eval_mse, "wall_s": elapsed}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = 64
    seeds = [0, 1, 2]
    steps = 1500
    n_train = 4096
    n_eval = 1024

    # Get the teacher param count for context
    teacher_demo = make_teacher(d, 0)
    teacher_p = count_params(teacher_demo)

    fm_demo = make_student_fullmix(d)
    fm_p = count_params(fm_demo)
    mlp_matched_demo = make_student_mlp_matched(d, fm_p)
    mlp_matched_p = count_params(mlp_matched_demo)

    print("=" * 78)
    print("Quality benchmark: distillation from random MLP teacher")
    print(f"device={device}, d={d}, seeds={seeds}, steps={steps}")
    print(f"teacher params  : {teacher_p:,}  (MLPFFN, mlp_ratio=4)")
    print(f"FullMix student : {fm_p:,}  (R_o=R_i=d/2, R_b=8, G=20)")
    print(f"MLP-matched student : {mlp_matched_p:,}")
    print(f"compression FullMix : {teacher_p / fm_p:.2f}x vs teacher")
    print("=" * 78)

    fm_results: dict[str, list[float]] = {"eval_mse": [], "wall": []}
    mlp_results: dict[str, list[float]] = {"eval_mse": [], "wall": []}

    for seed in seeds:
        teacher = make_teacher(d, seed)
        torch.manual_seed(seed)
        fm = make_student_fullmix(d)
        r_fm = distill(teacher, fm, d=d, steps=steps, lr=3e-3,
                       n_train=n_train, n_eval=n_eval, seed=seed, device=device)
        fm_results["eval_mse"].append(r_fm["eval_mse"])
        fm_results["wall"].append(r_fm["wall_s"])

        torch.manual_seed(seed)
        mlp_s = make_student_mlp_matched(d, fm_p)
        r_mlp = distill(teacher, mlp_s, d=d, steps=steps, lr=3e-3,
                        n_train=n_train, n_eval=n_eval, seed=seed, device=device)
        mlp_results["eval_mse"].append(r_mlp["eval_mse"])
        mlp_results["wall"].append(r_mlp["wall_s"])

    print(f"\n{'student':<14} {'eval_mse':>22} {'wall(s)':>10}")
    print("-" * 50)

    def _fmt(vals: list[float]) -> str:
        return (f"{statistics.mean(vals):.4e} ± "
                f"{statistics.stdev(vals) if len(vals) > 1 else 0:.2e}")

    print(f"{'FullMix':<14} {_fmt(fm_results['eval_mse']):>22} "
          f"{statistics.mean(fm_results['wall']):>10.2f}")
    print(f"{'MLP-matched':<14} {_fmt(mlp_results['eval_mse']):>22} "
          f"{statistics.mean(mlp_results['wall']):>10.2f}")

    wins = sum(
        1 for a, b in zip(fm_results["eval_mse"], mlp_results["eval_mse"], strict=True) if a < b
    )
    print(f"\nFullMix wins {wins}/{len(seeds)} seeds.")
    print("\nNotes:")
    print("- Teacher = random init MLP (mlp_ratio=4, 32K params at d=64).")
    print("- 'eval_mse' = student MSE against teacher's output on held-out inputs.")
    print("- A FullMix win here means tucker compression captures MLP teachers")
    print("  more parameter-efficiently than narrowing the MLP.")


if __name__ == "__main__":
    main()
