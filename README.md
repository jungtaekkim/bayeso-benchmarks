# BayesO Benchmarks
[![Build Status](https://travis-ci.org/jungtaekkim/bayeso-benchmarks.svg?branch=main)](https://travis-ci.org/jungtaekkim/bayeso-benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Benchmarks for Bayesian optimization.
The details of benchmark functions can be found in [these notes](http://jungtaek.github.io/notes/benchmarks_bo.pdf).

## Installation
We recommend installing it with `virtualenv`.
You can choose one of three installation options.

* Using PyPI repository (for user installation)

To install the released version in PyPI repository, command it.

```shell
$ pip install bayeso-benchmarks
```

* Using source code (for developer installation)

To install `bayeso-benchmarks` from source code, command

```shell
$ pip install .
```
in the `bayeso-benchmarks` root.

* Using source code (for editable development mode)

To use editable development mode, command

```shell
$ pip install -r requirements.txt
$ python setup.py develop
```
in the `bayeso-benchmarks` root.

* Uninstallation

If you would like to uninstall `bayeso-benchmarks`, command it.

```shell
$ pip uninstall bayeso-benchmarks
```

## Required Packages
Mandatory pacakges are inlcuded in `requirements.txt`.
The following `requirements` files include the package list, the purpose of which is described as follows.

* `requirements-dev.txt`: It is for developing the `bayeso-benchmarks` package.

## Author
* [Jungtaek Kim](http://jungtaek.github.io) (POSTECH)

## Contact
* Jungtaek Kim: [jtkim@postech.ac.kr](mailto:jtkim@postech.ac.kr)

## License
[MIT License](LICENSE)
