# Deployment Handoff

How to install, configure and run the `ds-template` package, and the prediction
output schema downstream systems can rely on.

---

## Installing the package

CI builds a wheel. Install it in any Python 3.11 environment:

```bash
pip install ds_template-<version>-py3-none-any.whl
```

---

## Configuration

Config is layered: `src/configs/default.yaml` → `src/configs/{CORNERSTONE_ENV}.yaml` →
environment variables (highest priority). Nested keys use a double underscore.

| Variable | Default | Description |
|---|---|---|
| `CORNERSTONE_ENV` | `local` | `local` (laptop) / `dev` / `staging` / `prod` (Databricks) |
| `CORNERSTONE_MLFLOW__TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking URI |
| `CORNERSTONE_MLFLOW__REGISTERED_MODEL_NAME` | `ds-template-model` | Registry model name |
| `CORNERSTONE_PREDICTIONS__OUTPUT_PATH` | `output/predictions` | Directory (local) or Delta path for predictions |
| `CORNERSTONE_TRAINING__N_ESTIMATORS` | `100` | Example training hyperparameter |
| `CORNERSTONE_DATABRICKS_HOST` | — | Workspace URL (staging/prod only) |
| `CORNERSTONE_DATABRICKS_TOKEN` | — | PAT or service principal token |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Running the pipeline

```bash
CORNERSTONE_ENV=staging ds-template train
CORNERSTONE_ENV=staging ds-template predict
```

For prod, use the Databricks Asset Bundle jobs in `databricks.yml`
(`cornerstone_train`, `cornerstone_score`).

---

## Prediction output schema

Each scored row is validated against `src.models.schemas.PredictionRecord`
before it is written.

| Column | Type | Description |
|---|---|---|
| `row_id` | int | Index of the input row |
| `prediction` | int | Predicted class |
| `score` | float | Model probability for the predicted class |
| `model_name` | string | MLflow registered model name |
| `model_version` | string | Registered model version |
| `run_id` | string | MLflow run that produced the model |
| `git_sha` | string | Git SHA of the code that scored the data |
| `scored_at` | timestamp (UTC) | When the row was scored |
