#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 4, 2023
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    first_term = np.sum(bx**2)

    inner_term = 0.0
    for ind in range(1, dim_bx + 1):
        inner_term += 0.5 * ind * bx[ind - 1]

    second_term = inner_term**2
    third_term = inner_term**4

    y = first_term + second_term + third_term
    return y


class Zakharov(Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))

        dim_bx = np.inf
        bounds = np.array([
            [-5.0, 10.0],
        ])
        global_minimizers = np.array([
            [0.0],
        ])
        global_minimum = 0.0
        dim_problem = dim_problem

        function = lambda bx: fun_target(bx, dim_problem)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)
