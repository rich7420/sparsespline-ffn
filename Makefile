.PHONY: install install-dev lint test build check clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check --no-cache src tests examples benchmarks

test:
	$(PYTHON) -m pytest

build:
	$(PYTHON) -m build

check: lint test build

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
