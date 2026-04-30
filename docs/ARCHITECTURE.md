# Architecture

## Package Layout

```text
src/sparsespline_ffn/
  fullmix_tucker.py   # permanent PyTorch reference implementation
  tucker_init.py      # variance-preserving and HOSVD initialization helpers
  schedules.py        # MLP baseline, layer schedules, integration factory
```

## Separation From pal-kan

`pal-kan` currently mixes paper artifacts, benchmark receipts, sparse KAN
kernels, nanochat adapters, and new SparseSpline-FFN code.  The independent
project should draw a hard line:

- library code lives under `src/sparsespline_ffn`;
- tests live under `tests`;
- runnable examples live under `examples`;
- external framework adapters live under `integrations`;
- paper-specific phase logs stay in `pal-kan`.

## Public API

The stable surface should stay small:

- `FullMixTuckerConfig`
- `FullMixTuckerFFN`
- `build_ffn`
- `should_replace_layer`
- `hosvd_warmstart_from_dense`

Everything kernel-specific should remain internal until the reference/kernel
contract is locked.

## Reference and Kernel Contract

The PyTorch reference path is the source of truth.  Any future fused kernel must:

- preserve output shape for arbitrary leading dimensions;
- match the reference within fp32 and bf16 tolerances;
- preserve gradients against the reference on small tensors;
- fall back to the reference when CUDA/Triton is unavailable.

The expected future entry point is a `use_kernel: bool | str` option, where
`False` means reference, `True` means auto-select kernel if available, and
`"required"` raises if the kernel cannot run.
