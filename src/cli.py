"""Command-line interface for ds-template.

Entry point registered in pyproject.toml as ``ds-template``.

Usage::

    ds-template train [--env local]
    ds-template predict [--env local]
    ds-template promote --version 3 [--alias champion]
    ds-template bundle validate [--target dev]
    ds-template bundle deploy --target staging
"""

import os
import subprocess
import sys

import mlflow
import typer

app = typer.Typer(
    name="ds-template",
    help="Personal DS/ML project template CLI.",
    add_completion=False,
)
bundle_app = typer.Typer(help="Databricks Asset Bundle commands.")
app.add_typer(bundle_app, name="bundle")


@app.command()
def train(
    env: str = typer.Option("local", "--env", "-e", help="CORNERSTONE_ENV value."),
) -> None:
    """Run the training pipeline."""
    os.environ["CORNERSTONE_ENV"] = env
    # Import after setting env so settings picks up the right env
    from src.config.settings import get_settings  # noqa: PLC0415

    get_settings.cache_clear()
    from src.models.train import train as _train  # noqa: PLC0415

    model_uri = _train()
    typer.echo(f"Training complete. Model URI: {model_uri}")


@app.command()
def predict(
    env: str = typer.Option("local", "--env", "-e", help="CORNERSTONE_ENV value."),
) -> None:
    """Run the scoring pipeline and write predictions."""
    os.environ["CORNERSTONE_ENV"] = env
    from src.config.settings import get_settings  # noqa: PLC0415

    get_settings.cache_clear()
    from src.models.predict import predict as _predict  # noqa: PLC0415

    n, path = _predict()
    typer.echo(f"Predict complete: {n} row(s) written to {path}")


@app.command()
def promote(
    version: str = typer.Option(
        ..., "--version", "-v", help="Model version to promote."
    ),
    alias: str = typer.Option(
        None, "--alias", "-a", help="Alias; default from config."
    ),
    env: str = typer.Option("local", "--env", "-e", help="CORNERSTONE_ENV value."),
) -> None:
    """Point the model alias at a version (promotion, or rollback to an old one)."""
    os.environ["CORNERSTONE_ENV"] = env
    from src.config.settings import get_settings  # noqa: PLC0415

    get_settings.cache_clear()
    from src.models.registry import set_model_alias  # noqa: PLC0415

    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    if cfg.mlflow.registry_uri:
        mlflow.set_registry_uri(cfg.mlflow.registry_uri)
    alias = alias or cfg.mlflow.model_alias
    set_model_alias(cfg.mlflow.registered_model_name, alias, version)
    typer.echo(f"{cfg.mlflow.registered_model_name}@{alias} -> v{version}")


@bundle_app.command("validate")
def bundle_validate(
    target: str = typer.Option("dev", "--target", "-t"),
) -> None:
    """Run ``databricks bundle validate`` for the given target."""
    result = subprocess.run(
        ["databricks", "bundle", "validate", "-t", target],
        check=False,
    )
    sys.exit(result.returncode)


@bundle_app.command("deploy")
def bundle_deploy(
    target: str = typer.Option(..., "--target", "-t", help="dev | staging | prod"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Deploy the Databricks Asset Bundle to the given target."""
    if target == "prod" and not yes:
        typer.confirm("You are about to deploy to PRODUCTION. Continue?", abort=True)
    result = subprocess.run(
        ["databricks", "bundle", "deploy", "-t", target],
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    app()
