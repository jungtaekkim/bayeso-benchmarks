#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 4, 2022
#

import numpy as np
import pytest

from bayeso_benchmarks.one_dim_step import *

class_fun = Step
str_name = 'step'

TEST_EPSILON = 1e-5


def test_init():
    obj_fun = class_fun()
    obj_fun = class_fun(steps=[-5., -3., 0., 1.], step_values=[4., 3., -1.])

    with pytest.raises(AssertionError) as error:
        class_fun(steps='abc')
    with pytest.raises(AssertionError) as error:
        class_fun(step_values='abc')
    with pytest.raises(AssertionError) as error:
        class_fun(steps=[1., 2., 3., 4.], step_values=[1., 2., 3., 4.])
    with pytest.raises(AssertionError) as error:
        class_fun(steps=[1, 2, 3, 4], step_values=[1., 2., 3.])
    with pytest.raises(AssertionError) as error:
        class_fun(steps=[1., 2., 3., 4.], step_values=[1, 2, 3])
    with pytest.raises(AssertionError) as error:
        class_fun(steps=[1., 2., 5., 3.], step_values=[1., 2., 3.])
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

    grids = obj_fun.sample_grids(5)
    truths_grids = np.array([
        [-2.0],
        [0.0],
        [1.0],
        [-1.0],
        [-1.0],
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
