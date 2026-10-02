"""Command-line interface for ds-template.

Entry point registered in pyproject.toml as ``ds-template``.

Usage::

    ds-template train [--env dev]
    ds-template predict [--env dev]
    ds-template bundle validate [--target dev]
    ds-template bundle deploy --target staging
"""

from __future__ import annotations

import os
import subprocess
import sys

import typer

app = typer.Typer(
    name="ds-template",
    help="Boeing DS/MLE alerting & prognostics template CLI.",
    add_completion=False,
)
bundle_app = typer.Typer(help="Databricks Asset Bundle commands.")
app.add_typer(bundle_app, name="bundle")


@app.command()
def train(
    env: str = typer.Option("dev", "--env", "-e", help="CORNERSTONE_ENV value."),
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
    env: str = typer.Option("dev", "--env", "-e", help="CORNERSTONE_ENV value."),
    source_system: str = typer.Option("demo", "--source-system"),
    source_asset_id: str = typer.Option("asset-001", "--asset-id"),
) -> None:
    """Run the scoring pipeline and emit alerts."""
    os.environ["CORNERSTONE_ENV"] = env
    from src.config.settings import get_settings  # noqa: PLC0415

    get_settings.cache_clear()
    from src.models.predict import predict as _predict  # noqa: PLC0415

    n, path = _predict(
        source_system=source_system,
        source_asset_id=source_asset_id,
    )
    typer.echo(f"Predict complete: {n} alert(s) written to {path}")


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
