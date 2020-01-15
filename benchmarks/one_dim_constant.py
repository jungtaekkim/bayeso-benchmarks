import numpy as np

from benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx, constant):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    assert isinstance(constant, float)

    y = constant
    return y


class Constant(Function):
    def __init__(self,  
        constant=0.0
    ):
        assert isinstance(constant, float)

        dim_bx = 1
        min_bx = -10.0
        max_bx = 10.0
        bounds = np.array([
            [min_bx, max_bx],
        ])
        global_minimizers = np.array([
            [min_bx],
            [max_bx],
        ])
        global_minimum = constant
        function = lambda bx: fun_target(bx, dim_bx, constant)

        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function)

