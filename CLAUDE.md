# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# --- Setup (uv) ---
uv sync

# --- Checks ---
make test                       # unit tests only (excludes @slow)
uv run pytest                   # everything, incl. @slow tests
uv run ruff check               # lint
uv run ruff format              # format
uv run ty check src/            # type-check
uv build                        # build wheel → dist/
make clean                      # remove caches and build artifacts

# --- Run pipeline directly ---
CORNERSTONE_ENV=local uv run python -m src.models.train
CORNERSTONE_ENV=local uv run python -m src.models.predict

# --- CLI ---
uv run ds-template train --env local
uv run ds-template predict --env local
uv run ds-template promote --version 1     # point the champion alias at a version
uv run ds-template bundle validate --target dev
uv run ds-template bundle deploy --target staging

# --- API (needs the serve extra: uv sync --all-extras) ---
uv run uvicorn src.serving.app:app --reload
docker build -t ds-template .
# Container reading a local registry: MLflow stores absolute artifact paths in
# the sqlite DB, so mount both at the SAME absolute path as on the host.
#   docker run --rm -p 8000:8000 \
#     -e CORNERSTONE_MLFLOW__TRACKING_URI="sqlite:///$PWD/mlflow.db" \
#     -v "$PWD/mlflow.db:$PWD/mlflow.db" -v "$PWD/mlruns:$PWD/mlruns" ds-template

# --- Local tools ---
uv run jupyter lab
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# --- Docs ---
uv run mkdocs serve
```

Pre-commit: `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`, `ruff`, `ruff-format`.
Install with `uv run pre-commit install`.

## Architecture

DS/MLE project template. Structured as a pip-installable package under
`src/`. The deploy team receives a wheel built with `uv build`; they
`pip install` it rather than re-translating code.

### Module map

```
src/
  config/settings.py       → Layered config: default.yaml → {env}.yaml → env vars
  data/load_data.py        → Data loading (pandas; add Spark variant here)
  features/build_features.py → Feature engineering
  models/
    train.py               → Training entrypoint: config → seed → MLflow run → fit → log
    predict.py             → load_champion(), score(), and the batch predict() entrypoint
    registry.py            → MLflow registry helpers (aliases: set_model_alias, get_version_by_alias)
    schemas.py             → PredictionRecord pydantic model (the prediction schema contract)
  tracking/mlflow_utils.py → start_run() context manager with auto-tags; log_model_with_signature
  monitoring/
    drift.py               → PSI, KS drift tests, feature_drift_report
  logging/logger.py        → JSON structured logger factory (get_logger)
  utils/
    helpers.py             → print_banner and misc utilities
    seed.py                → set_all_seeds(seed) for random/numpy/torch
  serving/app.py           → FastAPI app: /health, /ready, /predict (shares load_champion + score)
  cli.py                   → typer CLI: train | predict | promote | bundle validate/deploy
```

### Pipeline flow

```
src/configs/{env}.yaml + env vars
        ↓
src/config/settings.py (get_settings)
        ↓
src/data/load_data.py → src/features/build_features.py
        ↓
src/models/train.py (MLflow run via tracking/mlflow_utils.py)
        ↓
src/models/predict.py → PredictionRecord validation → parquet/Delta predictions table
```

### PredictionRecord schema

Every scored row is validated against `src/models/schemas.py:PredictionRecord`
before it is written. This is the contract the deploy team reads from the
predictions table. See the model docstring for the fields.

### Config system

`src/config/settings.py:get_settings()` merges:
1. `src/configs/default.yaml`
2. `src/configs/{CORNERSTONE_ENV}.yaml`
3. Environment variables (prefix `CORNERSTONE_`, double-underscore for nesting)

Use `get_settings.cache_clear()` in tests to reset between cases.

### Databricks Asset Bundle

`databricks.yml` defines `dev`, `staging`, and `prod` targets with
train and score jobs. Set `DATABRICKS_HOST` and `DATABRICKS_TOKEN` env vars
before deploying. DAB validation runs in GitHub Actions (`bundle-validate` job,
skipped when the secrets are not set).

### Docs

System design (data flow, environments, promotion/rollback): `docs/arch/system.md`.

MkDocs Material (`mkdocs.yml`). Architecture stubs live in `docs/arch/`.
`docs/handoff.md` is the deploy-team-facing reference.

## Key conventions

- **Tooling**: `uv` (lockfile `uv.lock`), `ruff` for lint + format, `ty` for types,
  `pytest` for tests.
- **Tests**: unit tests in `tests/unit/`, integration in `tests/integration/`.
  Slow tests are marked `@pytest.mark.slow` and excluded from `make test`.
- **Python**: 3.11 (matches Databricks Runtime 15.4 LTS). Do not use 3.12+ syntax.
  No `from __future__` imports.
- **Environments**: `CORNERSTONE_ENV=local` is the laptop (sqlite MLflow, local parquet);
  `dev`/`staging`/`prod` are Databricks. Config only; no code branches on the platform.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`).
- **Registry**: models are promoted with aliases (`champion`), not stages. On
  Databricks the model name is a Unity Catalog `catalog.schema.model` name.
- Every training run must go through `tracking/mlflow_utils.start_run()` — never
  call `mlflow.start_run()` directly.
- Every prediction row must validate against `PredictionRecord` before being written.
- Model artifacts >10 MB: MLflow registry or Azure Blob Storage, not `models/`.
