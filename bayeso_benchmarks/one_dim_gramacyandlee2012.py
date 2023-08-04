#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: August 4, 2023
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = np.sin(10.0 * np.pi * bx[0]) / (2 * bx[0]) + (bx[0] - 1.0)**4
    return y


class GramacyAndLee2012(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 1
        bounds = np.array([
            [0.5, 2.5],
        ])
        assert bounds.shape[0] == dim_bx
        assert bounds.shape[1] == 2

        global_minimizers = np.array([
            [0.54856344],
        ])
        global_minimum = -0.8690111350
        function = lambda bx: fun_target(bx, dim_bx)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
