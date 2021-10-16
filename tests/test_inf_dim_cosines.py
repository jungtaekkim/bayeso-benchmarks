#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np
import pytest

from bayeso_benchmarks.inf_dim_cosines import *
# last updated: February 8, 2021

class_fun = Cosines

TEST_EPSILON = 1e-5


def test_init():
    obj_fun = class_fun(5)

    with pytest.raises(TypeError) as error:
        class_fun()
    with pytest.raises(AssertionError) as error:
        class_fun('abc')
    with pytest.raises(AssertionError) as error:
        class_fun(2.1)
    with pytest.raises(AssertionError) as error:
        class_fun(2, seed='abc')
    with pytest.raises(AssertionError) as error:
        class_fun(2, seed=2.1)

def test_validate_properties():
    obj_fun = class_fun(1)
    obj_fun.validate_properties()

def test_output():
    obj_fun = class_fun(3)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [-2.7],
        [-2.8],
        [-2.7],
        [-2.8],
        [-2.9],
        [-2.8],
        [-2.7],
        [-2.8],
        [-2.7],
        [-2.8],
        [-2.9],
        [-2.8],
        [-2.9],
        [-3.0],
        [-2.9],
        [-2.8],
        [-2.9],
        [-2.8],
        [-2.7],
        [-2.8],
        [-2.7],
        [-2.8],
        [-2.9],
        [-2.8],
        [-2.7],
        [-2.8],
        [-2.7]
    ])
    
    print(grids)
    print(obj_fun.output(grids))
    print(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
