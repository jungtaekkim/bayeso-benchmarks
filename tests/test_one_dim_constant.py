#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np
import pytest

from bayeso_benchmarks.one_dim_constant import *

class_fun = Constant

TEST_EPSILON = 1e-5


def test_init():
    bounds = np.array([[0.0, 10.0]])
    obj_fun = class_fun()
    obj_fun = class_fun(bounds=bounds, constant=2.0)

    with pytest.raises(AssertionError) as error:
        class_fun(bounds=2)
    with pytest.raises(AssertionError) as error:
        class_fun(bounds=np.array([0.0, 10.0]))
    with pytest.raises(AssertionError) as error:
        class_fun(bounds=[0.0, 10.0])
    with pytest.raises(AssertionError) as error:
        class_fun(bounds=np.array([[10.0, 0.0]]))
    with pytest.raises(AssertionError) as error:
        class_fun(constant=2)
    with pytest.raises(AssertionError) as error:
        class_fun(constant='abc')
    with pytest.raises(AssertionError) as error:
        class_fun(seed='abc')
    with pytest.raises(AssertionError) as error:
        class_fun(seed=2.1)

def test_validate_properties():
    obj_fun = class_fun()
    obj_fun.validate_properties()

def test_output():
    obj_fun = class_fun()
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [0.0],
        [0.0],
        [0.0],
    ])
    
    print(grids)
    print(obj_fun.output(grids))
    print(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
