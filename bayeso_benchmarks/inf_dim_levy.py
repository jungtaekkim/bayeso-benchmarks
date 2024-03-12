#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 29, 2022
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx,
):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    bw = []
    for x in bx:
        w = 1.0 + (x - 1.0) / 4.0
        bw.append(w)
    bw = np.array(bw)

    y = np.sin(np.pi * bw[0])**2

    for w in bw[:-1]:
        y += (w - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w + 1.0)**2)

    y += (bw[-1] - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * bw[-1])**2)
    return y


class Levy(Function):
    def __init__(self, dim_problem, seed=None):
        assert isinstance(dim_problem, int)
        assert isinstance(seed, (type(None), int))

        dim_bx = np.inf
        bounds = np.array([
            [-10.0, 10.0],
        ])
        global_minimizers = np.array([
            [1.0],
        ])
        global_minimum = 0.0
        dim_problem = dim_problem

        function = lambda bx: fun_target(bx, dim_problem)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem, seed=seed)
