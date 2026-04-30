"""End-to-end LM-style training curve: MLP vs FullMix-Tucker.

Trains the prototype TinyTransformerLM on a synthetic copy/shift task
with cross-entropy loss for ~500 steps, comparing:

  - All-MLP baseline,
  - All-FullMix replacement (Pattern Full),
  - Late-half replacement (Pattern A+ analog at small K).

Reports the loss curve at fixed checkpoints plus final eval CE.  This is
the closest cheap analog to the paper's nanochat experiment: it tests
whether the FFN swap actually trains end-to-end, not just whether one
isolated layer fits a regression.

Synthetic task: predict the previous token (causal copy with shift = 1).
With a vocabulary of ~64 and a small block size, the optimal CE is near
zero and a healthy model gets there in a few hundred steps.

Auto-detects CUDA; bf16 on CUDA, fp32 on CPU.
"""
from __future__ import annotations

import statistics
import time

import torch

from integrations.nanochat.adapter import replace_mlp_with_sparsespline
from integrations.tiny_transformer import TinyConfig, TinyTransformerLM
from sparsespline_ffn import MLPFFN


def _device_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def _mlp_factory(d: int):
    def make(*, layer_idx: int) -> torch.nn.Module:
        return MLPFFN(d=d, mlp_ratio=2)
    return make


def make_data(B: int, T: int, vocab: int, device, seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    idx = torch.randint(0, vocab, (B, T), generator=g, device=device)
    # Targets: shift right by 1, fill last with random.
    targets = torch.roll(idx, shifts=-1, dims=1)
    targets[:, -1] = torch.randint(0, vocab, (B,), generator=g, device=device)
    return idx, targets


def train_curve(model, *, B, T, vocab, steps, lr, seed, device, dtype,
                log_every: int):
    model = model.to(device=device, dtype=dtype)
    idx, targets = make_data(B, T, vocab, device, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[tuple[int, float]] = []
    for step in range(steps):
        opt.zero_grad()
        _logits, loss = model(idx, targets)
        if step % log_every == 0 or step == steps - 1:
            history.append((step, loss.item()))
        loss.backward()
        opt.step()
    return history


def main():
    device, dtype = _device_dtype()
    vocab = 64
    d = 64
    n_layer = 4
    block_size = 32
    B = 4
    seeds = [0, 1, 2]
    steps = 400
    lr = 3e-3
    log_every = 50

    print("=" * 78)
    print("End-to-end LM-style training: MLP vs FullMix-Tucker")
    print(f"device={device}, dtype={dtype}")
    print(f"vocab={vocab}, d={d}, n_layer={n_layer}, "
          f"block_size={block_size}, B={B}")
    print(f"steps={steps}, seeds={seeds}, log_every={log_every}")
    print("=" * 78)

    configs = [
        ("all_mlp",       None,    None),
        ("all_fullmix",   "all",   dict(R_o=d // 2, R_i=d // 2, R_b=4, G=8)),
        ("late_fullmix",  "late",  dict(R_o=d // 2, R_i=d // 2, R_b=4, G=8)),
    ]

    results: dict[str, list[list[tuple[int, float]]]] = {
        cfg[0]: [] for cfg in configs
    }
    walls: dict[str, float] = {cfg[0]: 0.0 for cfg in configs}

    for label, schedule, ffn_kwargs in configs:
        for seed in seeds:
            torch.manual_seed(seed)
            cfg = TinyConfig(vocab_size=vocab, d=d, n_head=4,
                             n_layer=n_layer, block_size=block_size)
            model = TinyTransformerLM(cfg, _mlp_factory(d))
            if schedule is not None:
                replace_mlp_with_sparsespline(
                    model, schedule=schedule, **ffn_kwargs
                )
            t0 = time.perf_counter()
            history = train_curve(
                model, B=B, T=block_size, vocab=vocab,
                steps=steps, lr=lr, seed=seed,
                device=device, dtype=dtype, log_every=log_every,
            )
            walls[label] += time.perf_counter() - t0
            results[label].append(history)

    # Print headline table.
    print(f"\n{'config':<14} {'final_ce_mean':>16} {'final_ce_std':>14} "
          f"{'wall(s)':>10}")
    print("-" * 60)
    for label in results:
        finals = [h[-1][1] for h in results[label]]
        em = statistics.mean(finals)
        es = statistics.stdev(finals) if len(finals) > 1 else 0.0
        print(f"{label:<14} {em:>16.4f} {es:>14.4f} "
              f"{walls[label]:>10.2f}")

    # Print averaged loss curves.
    print("\nAveraged CE loss curves:")
    n_ckp = len(results[configs[0][0]][0])
    header = "  step      " + "".join(f"{c[0]:>16}" for c in configs)
    print(header)
    for i in range(n_ckp):
        step = results[configs[0][0]][0][i][0]
        row = f"  {step:<8d}"
        for label in [c[0] for c in configs]:
            mean = statistics.mean(h[i][1] for h in results[label])
            row += f"{mean:>16.4f}"
        print(row)

    print("\nNotes:")
    print("- Synthetic copy-shift task; optimal CE is near 0.")
    print("- FullMix variants should match or beat MLP given enough steps.")
    print("- 'late' replaces the upper half of layers; 'all' replaces all.")


if __name__ == "__main__":
    main()
