"""ds-template — Boeing alerting & prognostics DS/MLE project template.

Public API surface for the deploy team::

    from src.alerts.metadata import AlertMetadata
    from src.alerts.emit import emit_alert
    from src.config.settings import get_settings
    from src.tracking.mlflow_utils import start_run, log_model_with_signature
    from src.monitoring.metrics import summarise_alert_performance
    from src.monitoring.drift import feature_drift_report
    from src.prognostics.metrics import alpha_lambda_accuracy, rul_mae, rul_rmse
"""
