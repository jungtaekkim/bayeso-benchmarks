#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = (4.0 - 2.1 * bx[0]**2 + bx[0]**4 / 3.0) * bx[0]**2 + bx[0] * bx[1] + (-4.0 + 4.0 * bx[1]**2) * bx[1]**2
    return y


class SixHumpCamel(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 2
        bounds = np.array([
            [-3.0, 3.0],
            [-2.0, 2.0],
        ])
        global_minimizers = np.array([
            [0.0898, -0.7126],
            [-0.0898, 0.7126],
        ])
        global_minimum = -1.0316
        function = lambda bx: fun_target(bx, dim_bx)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
        except:
            super(SixHumpCamel, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
