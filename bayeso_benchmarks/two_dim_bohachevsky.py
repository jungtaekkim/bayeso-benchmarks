#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = bx[0]**2 + 2.0 * bx[1]**2 - 0.3 * np.cos(3.0 * np.pi * bx[0]) - 0.4 * np.cos(4.0 * np.pi * bx[1]) + 0.7
    return y


class Bohachevsky(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 2
        bounds = np.array([
            [-100.0, 100.0],
            [-100.0, 100.0],
        ])
        global_minimizers = np.array([
            [0.0, 0.0],
        ])
        global_minimum = 0.0
        function = lambda bx: fun_target(bx, dim_bx)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
        except:
            super(Bohachevsky, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
