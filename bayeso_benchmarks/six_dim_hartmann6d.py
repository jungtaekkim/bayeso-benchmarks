#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])

    outer = 0.0
    for i_ in range(0, 4):
        inner = 0.0
        for j_ in range(0, 6):
            inner += A[i_, j_] * (bx[j_] - P[i_, j_])**2
        outer += alpha[i_] * np.exp(-1.0 * inner)

    y = -1.0 * outer
    return y


class Hartmann6D(Function):
    def __init__(self,
        bounds=np.array([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]),
        seed=None
    ):
        assert isinstance(bounds, np.ndarray)
        assert isinstance(seed, (type(None), int))
        assert len(bounds.shape) == 2
        assert bounds.shape[1] == 2

        dim_bx = 6
        assert bounds.shape[0] == dim_bx

        global_minimizers = np.array([
            [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
        ])
        global_minimum = -3.322368
        function = lambda bx: fun_target(bx, dim_bx)

        try:
            super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
        except:
            super(Hartmann6D, self).__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
