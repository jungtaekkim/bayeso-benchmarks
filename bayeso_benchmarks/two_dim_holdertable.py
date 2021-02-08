#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = -1.0 * np.abs(np.sin(bx[0]) * np.cos(bx[1]) * np.exp(np.abs(1.0 - np.sqrt(bx[0]**2 + bx[1]**2) / np.pi)))
    return y


class HolderTable(Function):
    def __init__(self):
        dim_bx = 2
        bounds = np.array([
            [-10.0, 10.0],
            [-10.0, 10.0],
        ])
        global_minimizers = np.array([
            [8.05502, 9.66459],
            [8.05502, -9.66459],
            [-8.05502, 9.66459],
            [-8.05502, -9.66459],
        ])
        global_minimum = -19.2085
        function = lambda bx: fun_target(bx, dim_bx)

        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function)
