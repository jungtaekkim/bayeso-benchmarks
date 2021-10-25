#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = -1.0 * (bx[1] + 47.0) * np.sin(np.sqrt(np.abs(bx[1] + bx[0] / 2.0 + 47.0))) - bx[0] * np.sin(np.sqrt(np.abs(bx[0] - (bx[1] + 47.0))))
    return y


class Eggholder(Function):
    def __init__(self,
        bounds=np.array([
            [-512.0, 512.0],
            [-512.0, 512.0],
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
            [512.0, 404.2319],
        ])
        global_minimum = -959.6406627
        function = lambda bx: fun_target(bx, dim_bx)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
        except:
            super(Eggholder, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
