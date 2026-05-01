# SparseSpline-FFN

SparseSpline-FFN is a transformer FFN replacement built around locally-supported
B1 spline activations and a Tucker readout.  The current package ships the
permanent PyTorch reference implementation first; fused Triton kernels should be
added behind the same API after the quality gate passes.

## Install

```bash
cd sparsespline-ffn
python -m pip install -e ".[dev]"
```

CUDA/Triton development:

```bash
python -m pip install -e ".[dev,cuda]"
```

## Quick Start

```python
import torch
from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN

cfg = FullMixTuckerConfig(d=768, m=768, R_o=96, R_i=96, R_b=16, G=20)
ffn = FullMixTuckerFFN(cfg)

x = torch.randn(2, 128, 768)
y = ffn(x)
assert y.shape == x.shape
```

For layer-by-layer transformer integration:

```python
from sparsespline_ffn import build_ffn

ffn = build_ffn(
    ffn_type="fullmix_tucker",
    d=768,
    layer_idx=8,
    num_layers=12,
    schedule="late",
    R_o=96,
    R_i=96,
    R_b=16,
    G=20,
)
```

## Using the Triton kernel

The PyTorch reference (`use_kernel=False`, default) is the permanent oracle —
correct everywhere, slow on backward.  The optional Triton kernel
accelerates the B1-spline lookup and its backward dQ scatter; it must
match the reference within fp32 1e-5 / bf16 5e-3 (the K.0.1 contract).

Install with the optional `[cuda]` extras to bring in Triton:

```bash
python -m pip install -e ".[dev,cuda]"
```

Opt in via `use_kernel`, which has tri-state semantics:

| value         | behavior                                                         |
|---------------|------------------------------------------------------------------|
| `False`       | reference path always (default).                                 |
| `True`        | prefer kernel; **silent fallback** to reference on CPU / no-Triton.|
| `"required"`  | demand kernel; **raise RuntimeError** if it cannot run.           |

```python
from sparsespline_ffn import (
    FullMixTuckerConfig, FullMixTuckerFFN,
    build_ffn, is_kernel_available,
)

if is_kernel_available():
    cfg = FullMixTuckerConfig(d=768, m=768, R_o=96, R_i=96, R_b=16, G=20,
                              use_kernel=True)
else:
    cfg = FullMixTuckerConfig(d=768, m=768, R_o=96, R_i=96, R_b=16, G=20)

ffn = FullMixTuckerFFN(cfg)
# At forward time you can introspect the actual path:
#   ffn.kernel_will_run(x)  -> True iff the Triton path will be taken
```

Use `"required"` in production training where you do **not** want to
silently lose the speedup if Triton is missing or the model is on CPU:

```python
ffn = build_ffn(
    ffn_type="fullmix_tucker", d=768, layer_idx=6, num_layers=12,
    schedule="late", R_o=96, R_i=96, R_b=16, G=20,
    use_kernel="required",
)
```

Saving / loading checkpoints does not depend on the kernel — `state_dict`
contains parameters only, so a model trained with `use_kernel=True` can
be loaded into a layer with `use_kernel=False` and vice versa without
any conversion.

## CLI: `python -m sparsespline_ffn`

A small diagnostic CLI is provided for sanity-checking installs, bug
reports, and capacity planning.  Three subcommands:

```bash
python -m sparsespline_ffn               # version + runtime + kernel availability
python -m sparsespline_ffn check-kernel  # actually runs the kernel end-to-end
python -m sparsespline_ffn config --d 768 --R_o 96 --R_i 96 --R_b 16
                                          # prints parameter / MAC counts vs MLP
```

`check-kernel` exits 0 if the Triton path runs end-to-end on this machine
and 1 otherwise — useful in CI and bug reports.

## Recommended Project Boundary

This repository should own:

- the SparseSpline-FFN modules and initialization logic;
- reference-vs-kernel numerical contracts;
- transformer replacement schedules and integration helpers;
- focused unit tests, examples, and benchmark harnesses;
- documentation needed by users who do not know the original paper repo.

The original `pal-kan` repository should keep paper receipts, large experiment
logs, and historical phase closeouts.  Those can cite this package rather than
housing production code.

## Current Status

Implemented:

- `FullMixTuckerFFN`: five-stage PyTorch reference path
  (mixer → B1 spline lookup → V → core C → readout U·γ);
- `FullMixTuckerConfig`: validation for rank/grid/mixer choices, including
  the non-compressive `m >= d` mixer guard;
- `build_ffn`: MLP fallback plus replacement schedules
  (`all`, `every2`, `every4`, `early`, `late`, `late_quarter`, `none`);
- HOSVD warm-start helpers (`hosvd_warmstart_from_dense`);
- nanochat / nanoGPT adapter at `integrations.nanochat.adapter`
  (`replace_mlp_with_sparsespline`, `summarize_replacement`);
- tiny prototype transformer at `integrations.tiny_transformer` for plumbing
  smoke tests;
- benchmark suite covering FLOPs, activation memory, latency (fwd / fwd+bwd
  split), parameter count, invariant audit at production scale, and quality
  on synthetic regression / high-frequency / Jacobian / distillation /
  convergence / R_o sweep / asymmetric-rank / placement-K / HOSVD warm-start /
  mixer ablation / grid-resolution / init-sensitivity / subspace-diversity
  workloads;
- 128-test CPU+CUDA test suite covering shape, autograd, output-rank bound
  (F.4.b), cumulative subspace coverage (F.5.1), variance-preserving init
  (L.4), HOSVD round-trip and re-injection, distributional invariants,
  K=12 stacking, asymmetric ranks, and `torch.utils.checkpoint` integration.

Planned:

- fused Triton forward path behind `use_kernel` and matching backward
  (FlashKAT-style coefficient-tiled gradient);
- reference/kernel equivalence tests at fp32 and bf16 tolerances;
- end-to-end nanochat training receipts published outside this package.

## Development

```bash
pytest
ruff check --no-cache src tests examples benchmarks
python examples/basic_usage.py
```

Or use the project targets:

```bash
make install-dev
make check
python benchmarks/run_all.py             # full benchmark sweep
python benchmarks/param_count.py          # quick analytical sanity
```

The reference implementation is intentionally kept readable.  Do not remove it
after adding kernels; it is the oracle for correctness and the CPU fallback.
