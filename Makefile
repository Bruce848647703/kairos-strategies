PY ?= python3

.PHONY: help bootstrap test lint format clean research
help:
	@echo "targets: bootstrap test lint format clean research"

bootstrap:
	$(PY) -m venv .venv
	. .venv/bin/activate && $(PY) -m pip install -U pip && pip install -e ".[dev,plot]"

test:
	$(PY) -m pytest -q

lint:
	@if command -v ruff >/dev/null 2>&1; then ruff check .; else echo "pip install ruff 后再 lint"; fi

format:
	@if command -v ruff >/dev/null 2>&1; then ruff format .; else echo "pip install ruff 后再 format"; fi

research:
	$(PY) examples/run_all.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist
