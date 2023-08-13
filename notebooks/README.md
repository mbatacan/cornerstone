# notebooks
Directory for storing initial exploratory analysis and modeling purpose Jupyter Notebooks.

## Templates
Templates of Jupyter Notebooks which utilize the carrot framework are included, and helps bring to speed exploration and modeling. Note that the templates utilize nbextensions, so if they are not set up, please refer to the carrot installation instructions on Github.


## Merging Jupyter Notebooks using nbmerge
Refer to [**nbmerge Documentation**](../../../../../documentation/tools/nb_merge.md) for detailed instructions.


## Formatting Jupyter Notebooks using black-nb
Jupyter Notebook code may be formatted using black-nb with the following command:
`black-nb --config <path-to-pyproject.toml> <target-notebook>`

Note that the above command only follows the configurations for black, but not for other configurations such as isort.
