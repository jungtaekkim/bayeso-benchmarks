#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np
import pytest

from bayeso_benchmarks.two_dim_dejong5 import *

class_fun = DeJong5

TEST_EPSILON = 1e-3


def test_init():
    obj_fun = class_fun()

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
        [499.99985236],
        [499.99917292],
        [499.99985236],
        [499.99917292],
        [12.67050581],
        [499.99917292],
        [499.99985236],
        [499.99917292],
        [499.99985236],
    ])
    
    print(grids)
    print(obj_fun.output(grids))
    print(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
