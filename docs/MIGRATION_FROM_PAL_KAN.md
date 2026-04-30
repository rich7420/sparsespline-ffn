# Migration From pal-kan

## Copied Now

- `JHCG_REDESIGN_THEORY.md` -> `docs/THEORY.md`
- `sparsefuse/fullmix_tucker.py` -> `src/sparsespline_ffn/fullmix_tucker.py`
- `sparsefuse/tucker_init.py` -> `src/sparsespline_ffn/tucker_init.py`
- MIT license

## Renamed Imports

Use:

```python
from sparsespline_ffn import FullMixTuckerConfig, FullMixTuckerFFN
```

instead of:

```python
from sparsefuse.fullmix_tucker import FullMixTuckerConfig, FullMixTuckerFFN
```

## Not Copied Yet

The broader `sparsefuse` KAN layers and legacy JHCG modules are not copied into
the public API of this first split.  They are valuable historical baselines, but
they make the package boundary unclear.  If they are needed for benchmarks,
import them from `pal-kan` or vendor them later under `benchmarks/legacy`.

## Suggested Extraction Order

1. Keep this directory in `pal-kan` until tests and examples are stable.
2. Create a new remote repository named `sparsespline-ffn`.
3. Move this directory to the new repository root.
4. Run `python -m pip install -e ".[dev]"` and `pytest`.
5. Add CI for Python 3.10, 3.11, and 3.12.
6. Add the fused kernel only after the reference quality gate is meaningful.
