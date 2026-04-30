# nanochat Integration

Thin adapter around karpathy/nanochat — and any nanoGPT-shaped model — that
swaps selected `block.mlp` modules for SparseSpline-FFN layers without
forking the upstream model code.

The implementation lives in [`adapter.py`](adapter.py).  The two entry
points are:

- `replace_mlp_with_sparsespline(model, schedule="late", **ffn_kwargs)` —
  walks `model.transformer.h` (or any iterable passed via `layers=`) and
  replaces each `block.<mlp_attr>` (default `"mlp"`) with the result of
  `sparsespline_ffn.build_ffn(...)`.  Layers that the schedule does not
  select keep an MLPFFN fallback so the model shape is preserved.
- `summarize_replacement(model, mlp_attr="mlp")` — returns
  `{"n_layers": int, "swapped": [int, ...], "kept_mlp": [int, ...]}` so you
  can sanity-check which layers actually became SparseSpline-FFN.

## Auto-detection conventions

The adapter looks for the layer list and embedding dim using nanoGPT /
nanochat names by default:

- layer list: `model.transformer.h`, falling back to `model.layers`;
- embedding dim: `model.config.n_embd`, falling back to `d`,
  `hidden_size`, or `model_dim`.

If your fork uses different names, pass `layers=...` and `d=...` directly,
or set `mlp_attr="ffn"` if the FFN slot is named `ffn`.

## Example

```python
from sparsespline_ffn import build_ffn  # noqa: F401  -- adapter pulls this in
from integrations.nanochat.adapter import (
    replace_mlp_with_sparsespline,
    summarize_replacement,
)

# `model` is a nanochat / nanoGPT-style model whose blocks live at
# model.transformer.h and whose config exposes n_embd.
replace_mlp_with_sparsespline(
    model,
    schedule="late",        # see sparsespline_ffn.should_replace_layer
    R_o=96, R_i=96, R_b=16, # FullMix-Tucker rank
    G=20,
)
print(summarize_replacement(model))
# {'n_layers': 12, 'swapped': [6, 7, 8, 9, 10, 11], 'kept_mlp': [0, 1, 2, 3, 4, 5]}
```

For training receipts and end-to-end nanochat scripts, prefer to keep the
launcher code outside this directory so the library can be installed
without nanochat as a dependency.

## Tests

Adapter behavior is pinned in `tests/test_integrations.py`, which exercises:

- `late`, `all`, and `none` schedules on a tiny prototype transformer;
- forward and backward through the swapped model;
- `summarize_replacement` agreement with the schedule;
- error paths (no layer list, no `d`, missing `mlp` attribute, empty
  layer list);
- `mlp_attr="ffn"` for forks that name the FFN slot differently.
