# Handoff Guide for the Deploy Team

This document describes how to install, configure, and run the `ds-template`
package, and defines the alert output schema your downstream systems can rely on.

---

## Installing the package

The DS team ships a wheel artifact from CI.  Install it in any Python 3.11 environment:

```bash
pip install ds_template-<version>-py3-none-any.whl
```

No source code access required.  All public functions are importable from `src.*`.

---

## Configuration

All configuration is via **environment variables** prefixed `CORNERSTONE_`.
No config files need to be deployed alongside the wheel.

| Variable | Default | Description |
|---|---|---|
| `CORNERSTONE_ENV` | `dev` | `dev` / `staging` / `prod` |
| `CORNERSTONE_MLFLOW__TRACKING_URI` | `mlruns` | MLflow tracking server URI |
| `CORNERSTONE_MLFLOW__REGISTERED_MODEL_NAME` | `ds-template-model` | Registry model name |
| `CORNERSTONE_ALERTS__TABLE_PATH` | `output/alerts` | Delta table path (or local parquet dir) |
| `CORNERSTONE_ALERTS__DEFAULT_THRESHOLD` | `0.5` | Alert decision threshold |
| `CORNERSTONE_ALERTS__DEFAULT_SEVERITY` | `warn` | Default severity (`info`/`warn`/`critical`) |
| `CORNERSTONE_DATABRICKS_HOST` | — | Workspace URL (staging/prod only) |
| `CORNERSTONE_DATABRICKS_TOKEN` | — | PAT or service principal token |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Running the pipeline

### Training

```bash
CORNERSTONE_ENV=staging ds-template train
# or
CORNERSTONE_ENV=staging python -m src.models.train
```

### Scoring / alert emission

```bash
CORNERSTONE_ENV=staging ds-template predict \
    --source-system "aircraft-health-telemetry" \
    --asset-id "N12345"
```

### Databricks job (recommended for prod)

Use the Databricks Asset Bundle jobs defined in `deployment/databricks.yml`.
The `cornerstone_score` job runs scoring on a schedule and writes alerts to
the configured Delta table path.

---

## Alert output schema

Every alert row written to `CORNERSTONE_ALERTS__TABLE_PATH` conforms to this schema.
**This schema is stable across releases**; new optional fields may be added but
existing fields will not be renamed or removed without a major version bump.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `alert_id` | string (UUID4) | No | Unique alert identifier |
| `emitted_at` | ISO-8601 string (UTC) | No | When the alert was generated |
| `env` | string | No | `dev` / `staging` / `prod` |
| `model_name` | string | No | MLflow registered model name |
| `model_version` | string | No | Registered model version |
| `registered_model_uri` | string | No | `models:/<name>/<version>` |
| `mlflow_run_id` | string | No | MLflow run that produced the model |
| `git_sha` | string | No | Git SHA of the code that emitted this alert |
| `source_system` | string | No | Upstream telemetry system ID |
| `source_asset_id` | string | No | Asset identifier (tail number, serial, etc.) |
| `input_window_start` | ISO-8601 string (UTC) | No | Start of the scored data window |
| `input_window_end` | ISO-8601 string (UTC) | No | End of the scored data window |
| `feature_snapshot_uri` | string | Yes | Path to the feature frame used for scoring |
| `score` | float | No | Raw model score |
| `threshold` | float | No | Decision threshold crossed |
| `severity` | string | No | `info` / `warn` / `critical` |
| `prognostic_horizon_hours` | float | Yes | Predicted hours until failure (RUL models) |
| `explanation_uri` | string | Yes | SHAP / attribution artifact path |
| `upstream_pipeline_run_id` | string | Yes | Databricks job run ID |

### Example query (Databricks SQL)

```sql
SELECT
    source_asset_id,
    model_name,
    model_version,
    git_sha,
    score,
    threshold,
    severity,
    emitted_at,
    input_window_start,
    input_window_end
FROM delta.`dbfs:/mnt/prod/alerts`
WHERE env = 'prod'
  AND emitted_at >= current_timestamp() - INTERVAL 7 DAYS
ORDER BY emitted_at DESC;
```

---

## Contact

For questions about the model, features, or retraining schedule, contact the
DS/MLE team.  For infrastructure, cluster access, or Delta table permissions,
contact the DevOps/DataEng team.
