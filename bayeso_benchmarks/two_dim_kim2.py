#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 27, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = np.sin(bx[0] / 1.0) + np.cos(bx[1] / 1.0) \
        + np.sin(bx[0] / 2.0) + np.cos(bx[1] / 2.0) \
        + np.sin(bx[0] / 4.0) + np.cos(bx[1] / 4.0) \
        + np.sin(bx[0] / 8.0) + np.cos(bx[1] / 8.0) \
        + np.sin(bx[0] / 16.0) + np.cos(bx[1] / 16.0) \
        + 0.0032 * (bx[0] - 20.0)**2 + 0.0016 * (bx[1] - 20.0)**2
    return y


class Kim2(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 2
        bounds = np.array([
            [-128.0, 128.0],
            [-128.0, 128.0],
        ])
        global_minimizers = np.array([
            [-2.1013466, 34.14526252],
        ])
        global_minimum = -3.454387473489018
        function = lambda bx: fun_target(bx, dim_bx)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
        except:
            super(Kim2, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
