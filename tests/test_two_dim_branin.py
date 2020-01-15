import numpy as np
import pytest

from benchmarks.two_dim_branin import *

class_fun = Branin

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
        [308.12909601],
        [10.30790849],
        [10.96088904],
        [106.56869776],
        [24.12996441],
        [22.16653996],
        [17.50829952],
        [150.45202034],
        [145.87219088],
    ])
    
    print(grids)
    print(obj_fun.output(grids))
    print(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)

