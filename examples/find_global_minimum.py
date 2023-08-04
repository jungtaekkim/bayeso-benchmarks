#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: August 4, 2023
#

import numpy as np
import scipy.optimize as scio

from bayeso_benchmarks import Ackley
from bayeso_benchmarks import Cosines
from bayeso_benchmarks import Griewank
from bayeso_benchmarks import Levy
from bayeso_benchmarks import Rastrigin
from bayeso_benchmarks import Rosenbrock
from bayeso_benchmarks import Sphere
from bayeso_benchmarks import Zakharov
from bayeso_benchmarks import GramacyAndLee2012
from bayeso_benchmarks import Beale
from bayeso_benchmarks import Bohachevsky
from bayeso_benchmarks import Branin
from bayeso_benchmarks import Bukin6
from bayeso_benchmarks import DeJong5
from bayeso_benchmarks import DropWave
from bayeso_benchmarks import Easom
from bayeso_benchmarks import Eggholder
from bayeso_benchmarks import GoldsteinPrice
from bayeso_benchmarks import HolderTable
from bayeso_benchmarks import Kim1
from bayeso_benchmarks import Kim2
from bayeso_benchmarks import Kim3
from bayeso_benchmarks import Michalewicz
from bayeso_benchmarks import Shubert
from bayeso_benchmarks import SixHumpCamel
from bayeso_benchmarks import ThreeHumpCamel
from bayeso_benchmarks import Colville
from bayeso_benchmarks import Hartmann3D
from bayeso_benchmarks import Hartmann6D


TEST_EPSILON = 1e-7

all_benchmarks = [
    Ackley,
    Cosines,
    Griewank,
    Levy,
    Rastrigin,
    Rosenbrock,
    Sphere,
    Zakharov,
    GramacyAndLee2012,
    Beale,
    Bohachevsky,
    Branin,
    Bukin6,
    DeJong5,
    DropWave,
    Easom,
    Eggholder,
    GoldsteinPrice,
    HolderTable,
    Kim1,
    Kim2,
    Kim3,
    Michalewicz,
    Shubert,
    SixHumpCamel,
    ThreeHumpCamel,
    Colville,
    Hartmann3D,
    Hartmann6D,
]


def find_global_minimum(obj_fun):
    print(obj_fun.name, flush=True)

    fun_target = lambda bx: np.squeeze(obj_fun.output(bx), axis=1)

    grids = obj_fun.sample_grids(5)
    grids = np.concatenate((grids, obj_fun.get_global_minimizers()), axis=0)

    list_bx = []
    list_by = []

    for initial in grids:
        results = scio.minimize(fun_target, initial, method='L-BFGS-B', bounds=obj_fun.get_bounds())

        list_bx.append(results.x)
        list_by.append(results.fun)

    ind_minimum = np.argmin(np.squeeze(list_by))
    bx_best = list_bx[ind_minimum]
    y_best = list_by[ind_minimum]

    print(bx_best, flush=True)
    print(obj_fun.global_minimum, flush=True)
    print(y_best, flush=True)
    print('', flush=True)

    X = np.array(list_bx)
    by = np.squeeze(list_by)
    indices = np.argsort(by)
    X = X[indices]
    by = by[indices]

    print('candidates of global optima', flush=True)
    for bx_candidate, y_candidate in zip(X, by):
        if (obj_fun.global_minimum - y_candidate) > 0:
            print(bx_candidate, y_candidate, flush=True)
    print('', flush=True)

    assert (obj_fun.global_minimum - y_best) <= TEST_EPSILON
    assert (obj_fun.global_minimum - y_best) <= 0

    for global_minimizer in obj_fun.get_global_minimizers():
        assert (obj_fun.global_minimum - fun_target(global_minimizer)[0]) <= 0


if __name__ == '__main__':
    for class_fun in all_benchmarks:
        try:
            obj_fun = class_fun()
        except:
            obj_fun = class_fun(2)

        find_global_minimum(obj_fun)
