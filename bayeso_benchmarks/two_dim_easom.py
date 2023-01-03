#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 3, 2023
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = -1.0 * np.cos(bx[0]) * np.cos(bx[1]) * np.exp(-1.0 * (bx[0] - np.pi)**2 - (bx[1] - np.pi)**2)
    return y


class Easom(Function):
    def __init__(self,
        bounds=np.array([
            [-100, 100],
            [-100, 100],
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
            [np.pi, np.pi],
        ])
        global_minimum = -1.0
        function = lambda bx: fun_target(bx, dim_bx)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
        except:
            super(Easom, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
