.PHONY: install install-dev lint typecheck test build smoke-dist check bench bench-smoke clean clean-runs

PYTHON ?= python3
COVERAGE_FILE ?= /tmp/sparsespline_ffn.coverage

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check --no-cache src tests examples benchmarks

typecheck:
	$(PYTHON) -m mypy

test:
	COVERAGE_FILE=$(COVERAGE_FILE) $(PYTHON) -m pytest \
	  --cov=sparsespline_ffn \
	  --cov-report=term-missing \
	  --cov-fail-under=80

build:
	$(PYTHON) -m build

smoke-dist: build
	$(PYTHON) -m pip install --force-reinstall --no-deps dist/sparsespline_ffn-*.whl
	$(PYTHON) -c "import sparsespline_ffn; print(sparsespline_ffn.__version__)"

bench:
	$(PYTHON) benchmarks/run_all.py

bench-smoke:
	$(PYTHON) benchmarks/run_all.py --smoke

check: lint typecheck test bench-smoke smoke-dist

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-runs:
	rm -rf benchmark_runs/*
