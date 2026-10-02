.DEFAULT_GOAL := help
PYTHON        ?= python
VENV          ?= .venv
PIP           := $(VENV)/bin/pip
PY            := $(VENV)/bin/python

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install         Create .venv and install all dev dependencies (pip)"
	@echo "  install-uv      Install via uv (optional, requires uv on PATH)"
	@echo "  test            Run unit tests (excludes @slow)"
	@echo "  test-all        Run all tests including slow integration tests"
	@echo "  lint            Run pylint on src/"
	@echo "  typecheck       Run mypy on src/"
	@echo "  format          Auto-format with black"
	@echo "  format-check    Check formatting without modifying files"
	@echo "  build           Build a distributable wheel"
	@echo "  pin             Regenerate requirements*.txt from pyproject.toml"
	@echo "  docker-build    Build the local dev Docker image"
	@echo "  docker-up       Start JupyterLab + MLflow via docker compose"
	@echo "  docker-down     Stop docker compose services"
	@echo "  bundle-validate Run 'databricks bundle validate -t dev'"
	@echo "  bundle-deploy   Run 'databricks bundle deploy' (set TARGET=staging|prod)"
	@echo "  clean           Remove build artifacts and __pycache__"
	@echo ""

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
.PHONY: install
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "✔ venv ready at $(VENV). Activate with: source $(VENV)/bin/activate"

.PHONY: install-uv
install-uv:
	uv sync

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
.PHONY: test
test:
	$(PY) -m pytest -m "not slow" tests/

.PHONY: test-all
test-all:
	$(PY) -m pytest tests/ --cov=src --cov-report=term-missing

# ---------------------------------------------------------------------------
# Linting & formatting
# ---------------------------------------------------------------------------
.PHONY: lint
lint:
	$(PY) -m pylint src/

.PHONY: typecheck
typecheck:
	$(PY) -m mypy src/

.PHONY: format
format:
	$(PY) -m black src/ tests/

.PHONY: format-check
format-check:
	$(PY) -m black --check src/ tests/

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
.PHONY: build
build:
	$(PY) -m build

.PHONY: pin
pin:
	@if command -v uv > /dev/null 2>&1; then \
		uv pip compile pyproject.toml -o requirements.txt; \
		uv pip compile pyproject.toml --extra dev -o requirements-dev.txt; \
	else \
		$(PY) -m piptools compile pyproject.toml -o requirements.txt; \
		$(PY) -m piptools compile pyproject.toml --extra dev -o requirements-dev.txt; \
	fi
	@echo "✔ requirements.txt and requirements-dev.txt updated"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
.PHONY: docker-build
docker-build:
	docker build -t ds-template:local .

.PHONY: docker-up
docker-up:
	docker compose up --build

.PHONY: docker-down
docker-down:
	docker compose down

# ---------------------------------------------------------------------------
# Databricks Asset Bundles
# ---------------------------------------------------------------------------
TARGET ?= dev

.PHONY: bundle-validate
bundle-validate:
	databricks bundle validate -t $(TARGET)

.PHONY: bundle-deploy
bundle-deploy:
	@if [ "$(TARGET)" = "prod" ]; then \
		read -p "Deploy to PRODUCTION? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1; \
	fi
	databricks bundle deploy -t $(TARGET)

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .mypy_cache/ .pytest_cache/ htmlcov/ .coverage
