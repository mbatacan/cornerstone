"""emit_alert — write validated alerts to a local parquet (dev) or Delta table (prod).

In dev/staging, alerts are written as parquet files under ``settings.alerts.table_path``.
In a real Databricks environment, replace the pandas parquet write with a
Delta append using the Spark session::

    spark.createDataFrame([row]).write.format("delta").mode("append").save(path)

The function returns a tuple of (rows_written, output_path) so callers can log
the result to MLflow or a downstream system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.alerts.metadata import AlertMetadata
from src.config.settings import get_settings
from src.logging.logger import get_logger

logger = get_logger(__name__)


def emit_alert(
    scores: pd.DataFrame,
    metadata_defaults: AlertMetadata,
    score_col: str = "score",
    threshold: Optional[float] = None,
) -> tuple[int, str]:
    """Validate and persist alerts to the configured sink.

    Filters ``scores`` to rows where ``score_col`` exceeds ``threshold``,
    enriches each row with ``metadata_defaults``, validates against
    :class:`~src.alerts.metadata.AlertMetadata`, and appends to the alerts table.

    Args:
        scores: DataFrame with at least a ``score_col`` column.  May also
            carry per-row overrides for any AlertMetadata field (e.g.
            ``source_asset_id``, ``input_window_start``).
        metadata_defaults: AlertMetadata instance whose fields are used as
            defaults for any column not present in ``scores``.
        score_col: Column in ``scores`` containing the model output score.
        threshold: Decision threshold.  Defaults to
            ``settings.alerts.default_threshold`` if not provided.

    Returns:
        Tuple of (number of alert rows written, output path / table name).

    Raises:
        ValueError: If ``score_col`` is not found in ``scores``.
        pydantic.ValidationError: If any alert row fails schema validation.
    """
    cfg = get_settings()
    effective_threshold = (
        threshold if threshold is not None else cfg.alerts.default_threshold
    )

    if score_col not in scores.columns:
        raise ValueError(f"Column '{score_col}' not found in scores DataFrame.")

    alert_rows = scores[scores[score_col] >= effective_threshold].copy()

    if alert_rows.empty:
        logger.info("emit_alert: no rows exceeded threshold %.3f", effective_threshold)
        return 0, cfg.alerts.table_path

    defaults = metadata_defaults.to_dict()
    records = []
    for _, row in alert_rows.iterrows():
        row_dict = {
            **defaults,
            **{k: v for k, v in row.items() if k in AlertMetadata.model_fields},
        }
        row_dict["score"] = float(row[score_col])
        row_dict["threshold"] = effective_threshold
        validated = AlertMetadata(**row_dict)
        records.append(validated.to_dict())

    output_df = pd.DataFrame(records)

    output_path = Path(cfg.alerts.table_path)
    output_path.mkdir(parents=True, exist_ok=True)
    partition_file = (
        output_path
        / f"alerts_{metadata_defaults.emitted_at.strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    output_df.to_parquet(partition_file, index=False)

    logger.info(
        "emit_alert: wrote %d alert(s) to %s",
        len(records),
        partition_file,
    )
    return len(records), str(partition_file)
