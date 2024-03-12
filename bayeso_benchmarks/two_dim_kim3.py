#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: August 4, 2023
#

import numpy as np

from bayeso_benchmarks import utils
from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = np.sin(bx[0] / 1.0) + np.cos(bx[1] / 1.0) \
        + np.sin(bx[0] / 2.0) + np.cos(bx[1] / 2.0) \
        + np.sin(bx[0] / 4.0) + np.cos(bx[1] / 4.0) \
        + np.sin(bx[0] / 8.0) + np.cos(bx[1] / 8.0) \
        + np.sin(bx[0] / 16.0) + np.cos(bx[1] / 16.0) \
        + 0.0016 * (bx[0] - 40.0)**2 + 0.0008 * (bx[1] - 40.0)**2 \
        - 256000.0 * utils.pdf_two_dim_normal(
            bx, np.array([-120.0, -120.0]),
            np.array([[1000.0, 0.0], [0.0, 1000.0]])
        ) \
        - 256000.0 * utils.pdf_two_dim_normal(
            bx, np.array([-120.0, +120.0]),
            np.array([[1000.0, 0.0], [0.0, 1000.0]])
        )

    return y


class Kim3(Function):
    def __init__(self, seed=None):
        assert isinstance(seed, (type(None), int))

        dim_bx = 2
        bounds = np.array([
            [-256.0, 256.0],
            [-256.0, 256.0],
        ])
        assert bounds.shape[0] == dim_bx
        assert bounds.shape[1] == 2

        global_minimizers = np.array([
            [48.12477173, 34.19859065],
        ])
        global_minimum = -4.9439679194
        function = lambda bx: fun_target(bx, dim_bx)

        super().__init__(dim_bx, bounds, global_minimizers, global_minimum, function, seed=seed)
