#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 -m ruff check --no-cache src tests examples benchmarks
python3 -m build
