#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: August 4, 2023
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    A = np.array([
        [-32.0, -16.0, 0.0, 16.0, 32.0, -32.0, -16.0, 0.0, 16.0, 32.0, -32.0, -16.0, 0.0, 16.0, 32.0, -32.0, -16.0, 0.0, 16.0, 32.0, -32.0, -16.0, 0.0, 16.0, 32.0],
        [-32.0, -32.0, -32.0, -32.0, -32.0, -16.0, -16.0, -16.0, -16.0, -16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 16.0, 16.0, 16.0, 16.0, 32.0, 32.0, 32.0, 32.0, 32.0],
    ])
    y = 0.002

    for ind in range(0, 25):
        cur_y = 1.0 / (ind + 1.0 + (bx[0] - A[0, ind])**6 + (bx[1] - A[1, ind])**6)
        y += cur_y
    y = y**(-1)

    return y


class DeJong5(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 2
        bounds = np.array([
            [-65.536, 65.536],
            [-65.536, 65.536],
        ])
        assert bounds.shape[0] == dim_bx
        assert bounds.shape[1] == 2

        global_minimizers = np.array([
            [-31.97707837, -31.97795471],
            [-31.99140499, -31.99140499],
            [-32.01411043, -32.01411352],
            [-32.01747329, -32.01236504],
            [-32.0293114, -32.01718511],
            [-31.9618885, -32.00659555],
            [-32.0400369, -31.9824982],
            [-31.98255954, -32.04163256],
        ])
        global_minimum = 0.9980038378
        function = lambda bx: fun_target(bx, dim_bx)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
