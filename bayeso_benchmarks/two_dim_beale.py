#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = (1.5 - bx[0] + bx[0] * bx[1])**2 + (2.25 - bx[0] + bx[0] * bx[1]**2)**2 + (2.625 - bx[0] + bx[0] * bx[1]**3)**2
    return y


class Beale(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 2
        bounds = np.array([
            [-4.5, 4.5],
            [-4.5, 4.5],
        ])
        global_minimizers = np.array([
            [3.0, 0.5],
        ])
        global_minimum = 0.0
        function = lambda bx: fun_target(bx, dim_bx)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
        except:
            super(Beale, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
