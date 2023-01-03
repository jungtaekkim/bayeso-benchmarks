#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 3, 2023
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    first_term = 0.0
    for ind in range(1, 6):
        first_term += ind * np.cos((ind + 1.0) * bx[0] + ind)

    second_term = 0.0
    for ind in range(1, 6):
        second_term += ind * np.cos((ind + 1.0) * bx[1] + ind)

    y = first_term * second_term
    return y


class Shubert(Function):
    def __init__(self,
        bounds=np.array([
            [-10, 10],
            [-10, 10],
        ]),
        seed=None
    ):
        assert isinstance(bounds, np.ndarray)
        assert isinstance(seed, (type(None), int))
        assert len(bounds.shape) == 2
        assert bounds.shape[1] == 2

        dim_bx = 2
        assert bounds.shape[0] == dim_bx

        global_minimizers = np.array([
            [-7.08350641, -7.70831374],
        ])
        global_minimum = -186.73090883
        function = lambda bx: fun_target(bx, dim_bx)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
        except:
            super(Shubert, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
