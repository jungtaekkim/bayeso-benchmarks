#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: August 4, 2023
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ])
    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828],
    ])

    outer = 0.0
    for i_ in range(0, 4):
        inner = 0.0
        for j_ in range(0, 3):
            inner += A[i_, j_] * (bx[j_] - P[i_, j_])**2
        outer += alpha[i_] * np.exp(-1.0 * inner)

    y = -1.0 * outer
    return y


class Hartmann3D(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 3
        bounds = np.array([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        assert bounds.shape[0] == dim_bx
        assert bounds.shape[1] == 2

        global_minimizers = np.array([
            [0.11458889, 0.55564889, 0.85254698],
        ])
        global_minimum = -3.8627797874
        function = lambda bx: fun_target(bx, dim_bx)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
