#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx, slope):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    assert isinstance(slope, float)

    y = slope * bx[0]
    return y


class Linear(Function):
    def __init__(self,
        bounds=np.array([
            [-10, 10],
        ]),
        slope=1.0,
        seed=None
    ):
        assert isinstance(slope, float)
        assert isinstance(bounds, np.ndarray)
        assert isinstance(seed, (type(None), int))
        assert len(bounds.shape) == 2
        assert bounds.shape[0] == 1
        assert bounds.shape[1] == 2
        assert bounds[0, 0] < bounds[0, 1]

        dim_bx = bounds.shape[0]

        if slope > 0.0:
            global_minimizers = np.array([
                [bounds[0, 0]],
            ])
            global_minimum = slope * bounds[0, 0]
        else:
            global_minimizers = np.array([
                [bounds[0, 1]],
            ])
            global_minimum = slope * bounds[0, 1]
        function = lambda bx: fun_target(bx, dim_bx, slope)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
        except:
            super(Linear, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
