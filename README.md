# Conda environment with environment.yml

[![Binder](http://mybinder.org/badge.svg)](http://beta.mybinder.org/v2/gh/binder-examples/conda_environment/v1.0?filepath=index.ipynb)

A Binder-compatible repo with an `environment.yml` file.

Access this Binder at the following URL:

http://beta.mybinder.org/v2/gh/binder-examples/conda_environment/v1.0?filepath=index.ipynb

## Notes
The `environment.yml` file should list all Python libraries on which your notebooks
depend, specified as though they were created using the following `conda` commands:

```
source activate example-environment
conda env export > environment.yml
```

Note that the only libraries available to you will be the ones specified in
the `environment.yml`, so be sure to include everything that you need!
