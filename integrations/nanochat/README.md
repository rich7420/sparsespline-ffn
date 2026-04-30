# nanochat Integration

This directory is reserved for a thin adapter around karpathy/nanochat.  The
adapter should not fork nanochat; it should expose helper functions that replace
selected MLP blocks with `sparsespline_ffn.build_ffn(...)`.

Expected shape:

```python
from sparsespline_ffn import build_ffn

def replace_mlp_with_sparsespline(model, schedule="late", **ffn_kwargs):
    for layer_idx, block in enumerate(model.transformer.h):
        block.mlp = build_ffn(
            ffn_type="fullmix_tucker",
            d=model.config.n_embd,
            layer_idx=layer_idx,
            num_layers=model.config.n_layer,
            schedule=schedule,
            **ffn_kwargs,
        )
    return model
```

Keep benchmark launchers outside the package import path so users can install
the library without nanochat as a dependency.
