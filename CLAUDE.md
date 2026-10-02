# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# --- Setup (pip + venv — primary path used by Boeing teammates) ---
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# --- Setup (uv — personal dev alternative) ---
uv sync

# --- Common targets ---
make test            # unit tests only (fast, excludes @slow)
make test-all        # all tests including integration (runs training)
make lint            # pylint src/
make typecheck       # mypy src/
make format          # black src/ tests/
make build           # build wheel → dist/

# --- Run pipeline directly ---
CORNERSTONE_ENV=dev python -m src.models.train
CORNERSTONE_ENV=dev python -m src.models.predict

# --- CLI (after pip install -e .) ---
ds-template train --env dev
ds-template predict --env dev
ds-template bundle validate --target dev
ds-template bundle deploy --target staging

# --- Docker local dev (JupyterLab on :8888, MLflow on :5000) ---
make docker-up

# --- Databricks Asset Bundle ---
make bundle-validate              # validate dev target
make bundle-deploy TARGET=staging # deploy to staging

# --- Regenerate pinned requirements ---
make pin

# --- Docs ---
python -m mkdocs serve
```

Pre-commit: `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`, `black`.
Install with `pre-commit install` (after activating venv).

## Architecture

Boeing alerting & prognostics DS/MLE project template. Structured as a
pip-installable package under `src/`. The deploy team receives a wheel
built from `make build`; they `pip install` it rather than re-translating code.

### Module map

```
src/
  config/settings.py       → Layered config: default.yaml → {env}.yaml → env vars
  data/load_data.py        → Data loading (pandas; add Spark variant here)
  features/build_features.py → Feature engineering
  models/
    train.py               → Training entrypoint: config → seed → MLflow run → fit → log
    predict.py             → Scoring entrypoint: load model → score → emit_alert
    registry.py            → MLflow registry helpers (get_latest_version, transition_stage)
  alerts/
    metadata.py            → AlertMetadata pydantic model (the alert schema contract)
    emit.py                → emit_alert(): validates & persists alerts to Delta/parquet
    thresholds.py          → Threshold calibration (precision-targeted, threshold sweep)
  tracking/mlflow_utils.py → start_run() context manager with auto-tags; log_model_with_signature
  monitoring/
    metrics.py             → precision@k, false_alarm_rate, lead time, summarise_alert_performance
    drift.py               → PSI, KS drift tests, feature_drift_report
  prognostics/metrics.py   → alpha_lambda_accuracy, rul_mae, rul_rmse, rul_score, prognostic_horizon
  logging/logger.py        → JSON structured logger factory (get_logger)
  utils/
    helpers.py             → print_banner and misc utilities
    seed.py                → set_all_seeds(seed) for random/numpy/torch
  cli.py                   → typer CLI: train | predict | bundle validate/deploy
```

### Pipeline flow

```
configs/{env}.yaml + env vars
        ↓
src/config/settings.py (get_settings)
        ↓
src/data/load_data.py → src/features/build_features.py
        ↓
src/models/train.py (MLflow run via tracking/mlflow_utils.py)
        ↓
src/models/predict.py → src/alerts/emit.py → Delta/parquet alerts table
```

### AlertMetadata schema

Every emitted alert carries a fixed set of fields defined in
`src/alerts/metadata.py:AlertMetadata`. This is the contract the deploy team
reads from the Delta table. Required fields include: `model_name`,
`model_version`, `registered_model_uri`, `mlflow_run_id`, `git_sha`,
`source_system`, `source_asset_id`, `input_window_start/end`, `score`,
`threshold`. See the model docstring for the full list.

### Config system

`src/config/settings.py:get_settings()` merges:
1. `configs/default.yaml`
2. `configs/{CORNERSTONE_ENV}.yaml`
3. Environment variables (prefix `CORNERSTONE_`, double-underscore for nesting)

Use `get_settings.cache_clear()` in tests to reset between cases.

### Databricks Asset Bundle

`deployment/databricks.yml` defines `dev`, `staging`, and `prod` targets with
train and score jobs. Set `DATABRICKS_HOST` and `DATABRICKS_TOKEN` env vars
before deploying. DAB validation runs in GitLab CI (`bundle-validate` stage).

### Docs

MkDocs Material (`mkdocs.yml`). Architecture stubs live in `docs/arch/`.
`docs/handoff.md` is the deploy-team-facing reference.

## Key conventions

- **Linter**: `pylint` (not ruff check). Run with `make lint`.
- **Formatter**: `black` — pre-commit enforces it; `make format` to fix.
- **Type checker**: `mypy`.
- **Tests**: `pytest`; unit tests in `tests/unit/`, integration in `tests/integration/`.
  Slow tests are marked `@pytest.mark.slow` and excluded from `make test`.
- **Python**: 3.11 (matches Databricks Runtime 15.4 LTS). Do not use 3.12+ syntax.
- **Install path**: pip + venv is primary; uv is optional. No Makefile target
  requires uv on PATH.
- Every training run must go through `tracking/mlflow_utils.start_run()` — never
  call `mlflow.start_run()` directly.
- Every alert row must validate against `AlertMetadata` before being written.
- Model artifacts >10 MB: MLflow registry or Azure Blob Storage, not `models/`.
