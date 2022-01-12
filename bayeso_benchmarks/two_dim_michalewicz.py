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

    for ind in range(0, dim_bx):
        y += np.sin(bx[ind]) * np.sin(((ind + 1.0) * bx[ind]**2) / np.pi)**(2.0 * 10.0)
    y *= -1.0

    return y


class Michalewicz(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 2
        bounds = np.array([
            [0.0, np.pi],
            [0.0, np.pi],
        ])
        global_minimizers = np.array([
            [2.20290552, 1.57079632],
        ])
        global_minimum = -1.8013034
        function = lambda bx: fun_target(bx, dim_bx)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
        except:
            super(Michalewicz, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
