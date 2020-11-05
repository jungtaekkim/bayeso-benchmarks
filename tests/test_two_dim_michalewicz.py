import numpy as np
import pytest

from benchmarks.two_dim_michalewicz import *

class_fun = Michalewicz

TEST_EPSILON = 1e-3


def test_init():
    obj_fun = class_fun()

def test_validate_properties():
    obj_fun = class_fun()
    obj_fun.validate_properties()

def test_output():
    obj_fun = class_fun()
    bounds = obj_fun.get_bounds()

    grids = obj_fun.get_grids(3)
    truths_grids = np.array([
        [-0.00000000e+00],
        [-9.76562500e-04],
        [-0.00000000e+00],
        [-1.00000000e+00],
        [-1.00097656e+00],
        [-1.00000000e+00],
        [-0.00000000e+00],
        [-9.76562500e-04],
        [-0.00000000e+00],
    ])
    
    print(grids)
    print(obj_fun.output(grids))
    print(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
