#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 3, 2023
#

import numpy as np
import scipy.optimize as scio
import pytest

from bayeso_benchmarks import Ackley
from bayeso_benchmarks import Branin
from bayeso_benchmarks import GramacyAndLee2012
from bayeso_benchmarks import HolderTable
from bayeso_benchmarks import Kim1
from bayeso_benchmarks import Kim2
from bayeso_benchmarks import Kim3
from bayeso_benchmarks import Rastrigin
from bayeso_benchmarks import Michalewicz
from bayeso_benchmarks import Shubert


TEST_EPSILON = 1e-7


def _test_global_minimum(obj_fun):
    fun_target = lambda bx: np.squeeze(obj_fun.output(bx), axis=1)

    grids = obj_fun.sample_grids(100)

    list_bx = []
    list_by = []

    for initial in grids:
        results = scio.minimize(fun_target, initial, method='L-BFGS-B', bounds=obj_fun.get_bounds())

        list_bx.append(results.x)
        list_by.append(results.fun)

    ind_minimum = np.argmin(np.squeeze(list_by))
    bx_best = list_bx[ind_minimum]
    y_best = list_by[ind_minimum]

    print(bx_best)
    print(obj_fun.global_minimum)
    print(y_best)

    X = np.array(list_bx)
    by = np.squeeze(list_by)
    indices = np.argsort(by)
    X = X[indices]
    by = by[indices]

    for bx_candidate, y_candidate in zip(X, by):
        if np.abs(obj_fun.global_minimum - y_candidate) < 1e0:
            print(bx_candidate, y_candidate)

    assert np.abs(obj_fun.global_minimum - y_best) < TEST_EPSILON

    for global_minimizer in obj_fun.get_global_minimizers():
        assert np.abs(obj_fun.global_minimum - fun_target(global_minimizer)[0]) < TEST_EPSILON

def test_global_minimum_branin():
    class_fun = Branin
    obj_fun = class_fun()

    _test_global_minimum(obj_fun)

def test_global_minimum_gramacyandlee2012():
    class_fun = GramacyAndLee2012
    obj_fun = class_fun()

    _test_global_minimum(obj_fun)

def test_global_minimum_holdertable():
    class_fun = HolderTable
    obj_fun = class_fun()

    _test_global_minimum(obj_fun)

def test_global_minimum_kim1():
    class_fun = Kim1
    obj_fun = class_fun()

    _test_global_minimum(obj_fun)

def test_global_minimum_kim2():
    class_fun = Kim2
    obj_fun = class_fun()

    _test_global_minimum(obj_fun)

def test_global_minimum_kim3():
    class_fun = Kim3
    obj_fun = class_fun()

    _test_global_minimum(obj_fun)

def test_global_minimum_shubert():
    class_fun = Shubert
    obj_fun = class_fun()

    _test_global_minimum(obj_fun)
