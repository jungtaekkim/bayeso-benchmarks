#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 13, 2023
#

import numpy as np
import pytest

from bayeso_benchmarks.two_dim_eggholder import *

class_fun = Eggholder
str_name = 'eggholder'

TEST_EPSILON = 1e-5


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
        [737.27824186],
        [192.69874664],
        [522.47207216],
        [-554.93052378],
        [-25.46033719],
        [-165.56113678],
        [1049.1316235],
        [557.15651626],
        [-126.16793738],
    ])
    
    print(grids)
    print(obj_fun.output(grids))
    print(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)

def test_name():
    obj_fun = class_fun()
    assert obj_fun.name == str_name

    assert obj_fun.__class__.__name__.lower() == str_name
    assert obj_fun.__class__.__qualname__.lower() == str_name
