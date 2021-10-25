#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx, constant):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    assert isinstance(constant, float)

    y = constant
    return y


class Constant(Function):
    def __init__(self,
        bounds = np.array([
            [-10.0, 10.0],
        ]),
        constant=0.0,
        seed=None
    ):
        assert isinstance(constant, float)
        assert isinstance(bounds, np.ndarray)
        assert isinstance(seed, (type(None), int))
        assert len(bounds.shape) == 2
        assert bounds.shape[0] == 1
        assert bounds.shape[1] == 2
        assert bounds[0, 0] < bounds[0, 1]

        dim_bx = bounds.shape[0]
        min_bx = bounds[0, 0]
        max_bx = bounds[0, 1]
        global_minimizers = np.array([
            [min_bx],
            [max_bx],
        ])
        global_minimum = constant
        function = lambda bx: fun_target(bx, dim_bx, constant)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
        except:
            super(Constant, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
