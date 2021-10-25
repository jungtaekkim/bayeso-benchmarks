#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = 0.0

    for ind in range(0, dim_bx - 1):
        y += 100 * (bx[ind+1] - bx[ind]**2)**2 + (bx[ind] - 1.0)**2
    return y


class Rosenbrock(Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))
        assert dim_problem > 1

        dim_bx = np.inf
        bounds = np.array([
            [-2.048, 2.048],
        ])
        global_minimizers = np.array([
            [1.0],
        ])
        global_minimum = 0.0
        dim_problem = dim_problem

        function = lambda bx: fun_target(bx, dim_problem)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)
        except:
            super(Rosenbrock, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)
