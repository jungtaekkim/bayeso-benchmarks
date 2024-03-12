#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 6, 2023
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = 100.0 * np.sqrt(np.abs(bx[1] - 0.01 * bx[0]**2)) + 0.01 * np.abs(bx[0] + 10.0)
    return y


class Bukin6(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 2
        bounds = np.array([
            [-15, -5],
            [-3, 3],
        ])
        assert bounds.shape[0] == dim_bx
        assert bounds.shape[1] == 2

        global_minimizers = np.array([
            [-10.0, 1.0],
        ])
        global_minimum = 0.0
        function = lambda bx: fun_target(bx, dim_bx)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
