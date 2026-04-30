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

- `FullMixTuckerFFN`: five-stage PyTorch reference path;
- `FullMixTuckerConfig`: validation for rank/grid/mixer choices;
- `build_ffn`: MLP fallback plus replacement schedules;
- HOSVD warm-start helpers;
- CPU tests for shape, gradients, schedules, dense-equivalence, and packaging.

Planned:

- fused Triton forward path behind `use_kernel`;
- reference/kernel equivalence tests at fp32 and bf16 tolerances;
- nanochat adapter package with no upstream fork required;
- benchmark scripts for parameter count, activation memory, and step latency.

## Development

```bash
pytest
ruff check src tests examples
python examples/basic_usage.py
```

The reference implementation is intentionally kept readable.  Do not remove it
after adding kernels; it is the oracle for correctness and the CPU fallback.
