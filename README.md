# Data Science Template Repo

🚧 Under construction 🚧

## Quick Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
make test
```

### Contributing

Before committing anything to the repository, set up our pre-commit hooks:

```bash
uv run pre-commit install
```

### VSCode Extensions

If developing in VSCode (highly recommended), add the following extensions for linting, type checking, and code formatting:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python): IntelliSense, debugging, Jupyter notebooks
- [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff): linting and formatting on save.

## Example: End-to-End Workflow

Train a model using the provided modules:

```bash
uv run ds-template train --env dev
```
