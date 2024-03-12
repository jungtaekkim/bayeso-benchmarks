<p align="center">
<img src="https://raw.githubusercontent.com/jungtaekkim/bayeso/main/docs/_static/assets/logo_bayeso_capitalized.svg" width="400" />
</p>

# BayesO Benchmarks: Benchmark Functions for Bayesian Optimization
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7577330.svg)](https://doi.org/10.5281/zenodo.7577330)
[![Build Status](https://github.com/jungtaekkim/bayeso-benchmarks/actions/workflows/pytest.yml/badge.svg)](https://github.com/jungtaekkim/bayeso-benchmarks/actions/workflows/pytest.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the implementation of benchmark functions for Bayesian optimization.
The details of benchmark functions can be found in [these notes](https://jungtaek.github.io/notes/benchmarks_bo.pdf).

* [https://bayeso.org](https://bayeso.org)

## Installation
We recommend installing it with `virtualenv`.
You can choose one of three installation options.

* Using PyPI repository (for user installation)

To install the released version in PyPI repository, command it.

```shell
pip install bayeso-benchmarks
```

* Using source code (for developer installation)

To install `bayeso-benchmarks` from source code, command the following in the `bayeso-benchmarks` root.

```shell
pip install .
```

* Using source code (for editable development mode)

To use editable development mode, command the following in the `bayeso-benchmarks` root.

```shell
pip install -e .
```

If you want to install the packages required for development, you can simply add `[dev]`.
For example, `pip install .[dev]` or `pip install -e .[dev]`.

* Uninstallation

If you would like to uninstall `bayeso-benchmarks`, command it.

```shell
pip uninstall bayeso-benchmarks
```

## Simple Example
A simple example on Branin function is shown below.
```python
from bayeso_benchmarks import Branin

obj_fun = Branin()
bounds = obj_fun.get_bounds()

X = obj_fun.sample_uniform(100)

Y = obj_fun.output(X)
Y_noise = obj_fun.output_gaussian_noise(X)
```

## Citation
```
@misc{KimJ2023software,
    author={Kim, Jungtaek},
    title={{BayesO Benchmarks}: Benchmark Functions for {Bayesian} Optimization},
    doi={10.5281/zenodo.7577330},
    url={https://github.com/jungtaekkim/bayeso-benchmarks},
    howpublished={\url{https://doi.org/10.5281/zenodo.7577330}},
    year={2023}
}
```

## License
[MIT License](LICENSE)
