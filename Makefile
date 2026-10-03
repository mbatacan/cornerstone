.DEFAULT_GOAL := test

.PHONY: test
test:
	uv run pytest -m "not slow"

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info .ruff_cache/ .pytest_cache/ htmlcov/ .coverage
