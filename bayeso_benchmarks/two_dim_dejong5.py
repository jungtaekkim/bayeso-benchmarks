#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
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
    def __init__(self):
        dim_bx = 2
        bounds = np.array([
            [-65.536, 65.536],
            [-65.536, 65.536],
        ])
        global_minimizers = np.array([
            [-32.10428207, -32.13705826],
            [-32.07150588, -32.13705826],
            [-32.03872968, -32.13705826],
            [-32.00595349, -32.13705826],
            [-31.97317729, -32.13705826],
            [-31.9404011, -32.13705826],
            [-31.90762491, -32.13705826],
            [-32.13705826, -32.10428207],
            [-32.10428207, -32.10428207],
            [-32.07150588, -32.10428207],
            [-32.03872968, -32.10428207],
            [-32.00595349, -32.10428207],
            [-31.97317729, -32.10428207],
            [-31.9404011, -32.10428207],
            [-31.90762491, -32.10428207],
            [-31.87484871, -32.10428207],
        ])
        global_minimum = 0.9980038753198021
        function = lambda bx: fun_target(bx, dim_bx)

        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function)
