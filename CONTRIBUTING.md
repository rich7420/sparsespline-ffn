# Contributing

## Local Setup

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 -m ruff check src tests examples
```

## Engineering Rules

- Keep `FullMixTuckerFFN` as the permanent reference implementation.
- New fused kernels must compare against the reference before becoming default.
- Public API additions should be small and documented in `README.md`.
- Do not add benchmark receipts or paper phase logs to the library package.

## Before Opening a PR

```bash
scripts/check_project.sh
```

