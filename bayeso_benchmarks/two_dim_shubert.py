#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: August 4, 2023
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
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 2
        bounds = np.array([
            [-10, 10],
            [-10, 10],
        ])
        assert bounds.shape[0] == dim_bx
        assert bounds.shape[1] == 2

        global_minimizers = np.array([
            [-7.08350641, -7.70831374],
            [5.48286421, -7.70831373],
            [-0.8003211, -7.70831374],
            [4.85805688, -7.08350641],
            [-7.70831374, -0.8003211],
            [-7.70831374, 5.4828642],
            [4.85805688, 5.48286421],
        ])
        global_minimum = -186.7309088311

        function = lambda bx: fun_target(bx, dim_bx)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
