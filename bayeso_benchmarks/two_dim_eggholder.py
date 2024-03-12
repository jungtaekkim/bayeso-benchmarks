#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: August 4, 2023
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = -1.0 * (bx[1] + 47.0) * np.sin(np.sqrt(np.abs(bx[1] + bx[0] / 2.0 + 47.0))) - bx[0] * np.sin(np.sqrt(np.abs(bx[0] - (bx[1] + 47.0))))
    return y


class Eggholder(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 2
        bounds = np.array([
            [-512.0, 512.0],
            [-512.0, 512.0],
        ])
        assert bounds.shape[0] == dim_bx
        assert bounds.shape[1] == 2

        global_minimizers = np.array([
            [512.0, 404.2318066],
        ])
        global_minimum = -959.6406627209
        function = lambda bx: fun_target(bx, dim_bx)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
