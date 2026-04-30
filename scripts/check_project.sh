#!/usr/bin/env bash
set -euo pipefail

export COVERAGE_FILE="${COVERAGE_FILE:-/tmp/sparsespline_ffn.coverage}"

python3 -m pip install -e ".[dev]"
python3 -m ruff check --no-cache src tests examples benchmarks
python3 -m mypy
python3 -m pytest \
    --cov=sparsespline_ffn \
    --cov-report=term-missing \
    --cov-fail-under=80
python3 benchmarks/run_all.py --smoke
python3 -m build
python3 -m pip install --force-reinstall --no-deps dist/sparsespline_ffn-*.whl
python3 -c "import sparsespline_ffn; print(sparsespline_ffn.__version__)"
