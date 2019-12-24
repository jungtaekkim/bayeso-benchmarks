import numpy as np

from benchmarks.benchmark_base import Function


def fun_target(bx, num_dim):
    assert len(bx.shape) == 1
    assert bx.shape[0] == num_dim

    y = (4.0 - 2.1 * bx[0]**2 + bx[0]**4 / 3.0) * bx[0]**2 + bx[0] * bx[1] + (-4.0 + 4.0 * bx[1]**2) * bx[1]**2
    return y


class SixHumpCamel(Function):
    def __init__(self):
        num_dim = 2
        bounds = np.array([
            [-3.0, 3.0],
            [-2.0, 2.0],
        ])
        global_minimizers = np.array([
            [0.0898, -0.7126],
            [-0.0898, 0.7126],
        ])
        global_minimum = -1.0316
        function = lambda bx: fun_target(bx, num_dim)

        Function.__init__(self, num_dim, bounds, global_minimizers, global_minimum, function)

