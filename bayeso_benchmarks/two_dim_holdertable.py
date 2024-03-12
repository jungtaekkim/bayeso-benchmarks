#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: August 4, 2023
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = -1.0 * np.abs(np.sin(bx[0]) * np.cos(bx[1]) * np.exp(np.abs(1.0 - np.sqrt(bx[0]**2 + bx[1]**2) / np.pi)))
    return y


class HolderTable(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 2
        bounds = np.array([
            [-10.0, 10.0],
            [-10.0, 10.0],
        ])
        assert bounds.shape[0] == dim_bx
        assert bounds.shape[1] == 2

        global_minimizers = np.array([
            [8.05502347, 9.66459002],
            [8.05502347, -9.66459002],
            [-8.05502347, 9.66459002],
            [-8.05502347, -9.66459002],
        ])
        global_minimum = -19.2085025679
        function = lambda bx: fun_target(bx, dim_bx)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
