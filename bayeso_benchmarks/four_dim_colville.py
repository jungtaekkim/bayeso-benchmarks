#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 30, 2022
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = 100.0 * (bx[0]**2 - bx[1])**2
    y += (bx[0] - 1.0)**2
    y += (bx[2] - 1.0)**2
    y += 90.0 * (bx[2]**2 - bx[3])**2
    y += 10.1 * ((bx[1] - 1.0)**2 + (bx[3] - 1.0)**2)
    y += 19.8 * (bx[1] - 1.0) * (bx[3] - 1.0)

    return y


class Colville(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 4
        bounds = np.array([
            [-10.0, 10.0],
            [-10.0, 10.0],
            [-10.0, 10.0],
            [-10.0, 10.0],
        ])
        assert bounds.shape[0] == dim_bx
        assert bounds.shape[1] == 2

        global_minimizers = np.array([
            [1.0, 1.0, 1.0, 1.0],
        ])
        global_minimum = 0.0
        function = lambda bx: fun_target(bx, dim_bx)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
