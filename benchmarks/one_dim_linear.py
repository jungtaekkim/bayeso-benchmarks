import numpy as np

from benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx, slope):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    assert isinstance(slope, float)

    y = slope * bx[0]
    return y


class Linear(Function):
    def __init__(self,  
        slope=1.0
    ):
        assert isinstance(slope, float)

        dim_bx = 1
        bounds = np.array([
            [-10, 10],
        ])
        global_minimizers = np.array([
            [slope * -10],
        ])
        global_minimum = slope * -10.0
        function = lambda bx: fun_target(bx, dim_bx, slope)

        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function)

