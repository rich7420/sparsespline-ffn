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
- `build_ffn` / `build_fullmix_tucker_ffn`
- `should_replace_layer`
- `hosvd_warmstart_from_dense`
- `is_kernel_available` (runtime check for the optional Triton kernel)

Everything kernel-specific stays inside `sparsespline_ffn.kernels`; users
opt into it via `use_kernel` on the config (see Reference and Kernel
Contract below).

## Reference and Kernel Contract

The PyTorch reference path is the source of truth.  Any future fused kernel must:

- preserve output shape for arbitrary leading dimensions;
- match the reference within fp32 and bf16 tolerances;
- preserve gradients against the reference on small tensors;
- fall back to the reference when CUDA/Triton is unavailable.

The user-facing entry point is `FullMixTuckerConfig.use_kernel: bool | str`,
implemented in `FullMixTuckerFFN._will_use_kernel`:

- `False` → form-B reference always (default).
- `True`  → prefer kernel; silently fall back to form-B when CUDA or
  Triton is unavailable.
- `"required"` → demand kernel; raise `RuntimeError` if the kernel cannot
  run for the given input.

Downstream code can pre-flight via the top-level helper
`is_kernel_available()` (which checks both `torch.cuda.is_available()`
and Triton importability) and per-input via
`FullMixTuckerFFN.kernel_will_run(x)`.

`use_kernel` is forwarded by `build_fullmix_tucker_ffn` and through
`build_ffn`'s `**fullmix_kwargs`, so the nanoGPT/nanochat adapter picks
it up without extra plumbing.

Checkpoint compatibility: `state_dict` contains only parameters, so a
model trained with `use_kernel=True` can be loaded into a layer with
`use_kernel=False` and vice versa without any conversion.
