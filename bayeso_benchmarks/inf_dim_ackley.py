#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx,
    a=20.0,
    b=0.2,
    c=2.0*np.pi
):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert isinstance(c, float)

    y = -a * np.exp(-b * np.linalg.norm(bx, ord=2, axis=0) * np.sqrt(1.0 / dim_bx)) - np.exp(1.0 / dim_bx * np.sum(np.cos(c * bx), axis=0)) + a + np.exp(1.0)
    return y


class Ackley(Function):
    def __init__(self, dim_problem):
        assert isinstance(dim_problem, int)

        dim_bx = np.inf
        bounds = np.array([
            [-32.768, 32.768],
        ])
        global_minimizers = np.array([
            [0.0],
        ])
        global_minimum = 0.0
        dim_problem = dim_problem

        function = lambda bx: fun_target(bx, dim_problem)

        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem)
