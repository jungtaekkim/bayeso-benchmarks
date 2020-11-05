#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: November 5, 2020
#

import numpy as np
import pytest

from benchmarks.two_dim_threehumpcamel import *

class_fun = ThreeHumpCamel

TEST_EPSILON = 1e-5


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
        [2047.91666667],
        [25.0],
        [1997.91666667],
        [1997.91666667],
        [0.0],
        [1997.91666667],
        [1997.91666667],
        [25.0],
        [2047.91666667],
    ])
    
    print(grids)
    print(obj_fun.output(grids))
    print(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
