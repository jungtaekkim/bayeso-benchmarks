#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 20, 2022
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = 10.0 * dim_bx

    for ind in range(0, dim_bx):
        y += bx[ind]**2 - 10.0 * np.cos(2.0 * np.pi * bx[ind])

    return y


class Rastrigin(Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))

        dim_bx = np.inf
        bounds = np.array([
            [-5.12, 5.12],
        ])
        global_minimizers = np.array([
            [0.0],
        ])
        global_minimum = 0.0
        dim_problem = dim_problem

        function = lambda bx: fun_target(bx, dim_problem)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)
