# MITA Data Science Template Repo

[Under construction]

## Quick Start

To start, install the required and recommended libraries.

1. Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
2. Install dependencies:

```bash
poetry install
```

### Contributing

Before committing anything to the repository, set up our pre-commit hooks:

```bash
pre-commit install
```

### VSCode Extensions

If developing in VSCode (highly recommended), add the following extensions for linting, type checking, and code formatting:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python): IntelliSense (Pylance), Linting, Debugging (multi-threaded, remote), Jupyter Notebooks, code 
- [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff): A Visual Studio Code extension with support for the Ruff linter.


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

This repo depends on a Python 3.9.12 so it may work with the [Carrot Framework](https://pages.github.ibm.com/cao/DS_Standards/latest).

To install the correct python version, we recommend using [pyenv](https://github.com/pyenv/pyenv).

1. You may install pyenv via [homebrew](https://brew.sh/):

```
brew install pyenv
```

2. Next, install the correct python version:

```
pyenv install 3.9.12
```

The response should look like this:
```
python-build: use openssl@1.1 from homebrew
python-build: use readline from homebrew
Installing Python-3.9.12...
python-build: use readline from homebrew
python-build: use zlib from xcode sdk
Installed Python-3.9.12 to /Users/erikarussi/.pyenv/versions/3.9.12
```

Note the folder in which this version was installed: `/Users/erikarussi/.pyenv/versions/3.9.12`

## Poetry

3. If you already have [Poetry installed](https://python-poetry.org/docs/#installation), you can now use the correct python version you installed in step 2 above. Note that it's the same folder in which Python 3.9.12 was installed with the added `bin/python` at the end.

```
poetry env use /Users/erikarussi/.pyenv/versions/3.9.12/bin/python
```

4. Verify that you're using the correct python version by running this command:

```
poetry env info
```

You should see a response like this:
```
Virtualenv
Python:         3.9.12
Implementation: CPython
Path:           /Users/erikarussi/Library/Caches/pypoetry/virtualenvs/external-ltr-T_-utIVA-py3.9
Executable:     /Users/erikarussi/Library/Caches/pypoetry/virtualenvs/external-ltr-T_-utIVA-py3.9/bin/python
Valid:          True
```

5. Finally, you can install all the dependencies of the project by running the following command in the root directory of this project:

```
poetry install
```

Note that when you run `poetry install`, the `src` directory is reset as the root directory of the project so python files can access all modules in the `src` directory.

### Adding New Dependencies

You may add a new dependency with poetry by running:
```
poetry add <name of package>
```
The new package will be added as a dependency in the `pyproject.toml` file. 


See [poetry docs](https://python-poetry.org/docs/cli/#add) if you would like to install a specific version of a package.

### Updating Dependencies

In order to get the latest versions of the dependencies, you should use the update command:
```
poetry update
```

## Use in VSCode or Jupyter Notebook

### Environments and Kernels

The Team usually works in VSCode. If you're using VSCode to run code, please ensure you're using the correct kernel or environment with the prerequisites already installed.

The kernel for this project starts with `external-ltr`. It is automatically installed when you run the `poetry install` command. You may have to restart VSCode to see this kernel as an option.

In VSCode, you can select the correct kernel on the top right of the interactive window in which you are running code:
![VSCode](../images/vscode.png)

In Jupyter Notebooks, the kernel is under the `Kernel` dropdown:

![Jupyter Notebook](../images/jupyter.png)

If you're still not seeing a kernel, you can create one for this project with ipython by running:
```
poetry run ipython kernel install --user --name=external-ltr
```

### Jupyter Debugging

When running any .py files or notebooks in interactive mode on VSCode, you may run into Jupyter specific issues. You may have to install an older version of the Jupyter extension to circumvent these issues. See [here](https://stackoverflow.com/questions/75350840/jupyter-notebook-error-jupyter-command-jupyter-notebook-not-found) for how to do it:

### Black Formatter

We also use the [`black` formatter](https://black.readthedocs.io/en/stable/) in VSCode to automatically format our code when we save in VSCode. [Here](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0) is how you can do the same.


## Adding new files to `/src`

When you add a new file to `/src`, you need to re-run `poetry install` to be able to import from it.

## Poetry Debugging

If you're having issues with Poetry, here are some workarounds that have worked for us:
- Run `poetry update` -- maybe your dependencies are not up to date
- If you're running poetry in VSCode's terminal, try running it in your machine's terminal, or vice versa
- Try deleting the `poetry.lock` file that gets generated when you run `poetry install` and rerun `poetry install` after deleting
- Delete any `.venv` folder created and rerun `poetry install` after deleting
- Delete the repo on your local machine, reclone the repo, and rerun `poetry install`
- If you're using `pyenv`, these steps may work:
```
#1. 
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile

#2. 
source ~/.bash_profile 

#3. 
pyenv shell 3.9.10 #(or whichever python version you want to use)

#4. to check which version you're using
poetry env info #should return the same version you've specified in step 3

#5. poetry install
```
- if you're using Conda:
```
#1.  Go to this directory and clean up prior poetry environment:
/Users/username/Library/Caches/pypoetry/virtualenvs

#2. activate the py10 (which is 3.10.4)
conda activate py10

#3.
poetry env use /Users/username/opt/anaconda3/envs/py10/bin/python

#4.
poetry install

#5. in VSCode
venv path = /Users/dggeorge/Library/Caches/pypoetry/virtualenvs
```