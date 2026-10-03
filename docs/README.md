# Data Science Template Repo

## Quick Start

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install dependencies (creates `.venv` on Python 3.11):

```bash
uv sync
```

3. Run the checks:

```bash
make test                # unit tests
uv run ruff check        # lint
uv run ruff format       # format
uv run ty check src/     # type-check
```

### Contributing

Before committing anything to the repository, set up the pre-commit hooks:

```bash
uv run pre-commit install
```

### VSCode Extensions

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python): IntelliSense, debugging, Jupyter notebooks
- [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff): linting and formatting on save

### Adding dependencies

```bash
uv add <package>             # runtime
uv add --dev <package>       # dev only
```

Commit the updated `pyproject.toml` and `uv.lock`.

## DVC

### `data` Folder

Our large files are located in the `data` directory. If you would like to push a new large file to remote using DVC, make sure that file is in the `data` directory.

### Adding Files to `data`

Once you've setup DVC as instructed in the [read.me](../../README.md#2-dvc) and your files are now in the `data` directory (subdirectory `interim` or `raw`), run the following command:
```
dvc add data
```

This will add all files currently in the `data` folder, including your latest files.

Then push to remote:
```
dvc push
```

The `data.dvc` file should have been edited once you have run the commands above. Make sure to commit this file to GitHub. Others will need the latest version of this file to be able to pull your newly added file.

### More on DVC

See the [DVC docs](https://dvc.org/doc) to learn more about DVC.
