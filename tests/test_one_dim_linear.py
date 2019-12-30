import numpy as np
import pytest

from benchmarks.one_dim_linear import *

class_fun = Linear

TEST_EPSILON = 1e-5


def test_init():
    obj_fun = class_fun()
    obj_fun = class_fun(2.0)

    with pytest.raises(AssertionError) as error:
        class_fun(2)
    with pytest.raises(AssertionError) as error:
        class_fun('abc')

def test_validate_properties():
    obj_fun = class_fun()
    obj_fun.validate_properties()

def test_output():
    obj_fun = class_fun()
    bounds = obj_fun.bounds

    grids = obj_fun.get_grids(3)
    truths_grids = np.array([
        [-10.0],
        [0.0],
        [10.0],
    ])
    
    print(grids)
    print(obj_fun.output(grids))
    assert np.all(obj_fun.output(grids) == truths_grids)
   
