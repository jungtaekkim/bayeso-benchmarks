#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 4, 2022
#

import numpy as np
import pytest

from bayeso_benchmarks.inf_dim_rosenbrock import *

class_fun = Rosenbrock
str_name = 'rosenbrock'

TEST_EPSILON = 1e-5


def test_init():
    obj_fun = class_fun(2)

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
    obj_fun = class_fun(5)
    obj_fun.validate_properties()

def test_output():
    obj_fun = class_fun(3)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [7.81185245e+03],
        [5.67443514e+03],
        [4.37587862e+03],
        [4.32635663e+03],
        [2.18893931e+03],
        [8.90382790e+02],
        [7.80366045e+03],
        [5.66624314e+03],
        [4.36768662e+03],
        [2.18893931e+03],
        [1.76950891e+03],
        [2.18893931e+03],
        [4.21430400e+02],
        [2.00000000e+00],
        [4.21430400e+02],
        [2.18074731e+03],
        [1.76131691e+03],
        [2.18074731e+03],
        [4.36768662e+03],
        [2.23026930e+03],
        [9.31712780e+02],
        [4.31816463e+03],
        [2.18074731e+03],
        [8.82190790e+02],
        [4.35949462e+03],
        [2.22207730e+03],
        [9.23520780e+02],
    ])
    
    print(grids)
    print(obj_fun.output(grids))
    print(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)

def test_name():
    obj_fun = class_fun(2)
    assert obj_fun.name == str_name + '_2'

    obj_fun = class_fun(4)
    assert obj_fun.name == str_name + '_4'

    obj_fun = class_fun(16)
    assert obj_fun.name == str_name + '_16'

    assert obj_fun.__class__.__name__.lower() == str_name
    assert obj_fun.__class__.__qualname__.lower() == str_name
