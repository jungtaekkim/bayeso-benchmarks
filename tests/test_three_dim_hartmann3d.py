#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 4, 2022
#

import numpy as np
import pytest

from bayeso_benchmarks.three_dim_hartmann3d import *

class_fun = Hartmann3D
str_name = 'hartmann3d'

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
        [-6.79741166e-02],
        [-1.36461045e-01],
        [-9.13324430e-02],
        [-9.71082067e-02],
        [-1.85406663e-01],
        [-9.02038776e-02],
        [-3.09547170e-02],
        [-7.29043824e-02],
        [-8.47693855e-02],
        [-1.80480228e-02],
        [-8.39060933e-01],
        [-1.99426284e+00],
        [-2.57290548e-02],
        [-6.28022015e-01],
        [-1.95703928e+00],
        [-8.19356834e-03],
        [-2.25915245e-01],
        [-1.82665019e+00],
        [-2.73536768e-04],
        [-2.26230774e+00],
        [-3.34829168e-01],
        [-2.04016877e-04],
        [-1.48565839e+00],
        [-3.25957854e-01],
        [-3.77271851e-05],
        [-2.24631259e-01],
        [-3.00476074e-01],
    ])

    print(grids)
    print(obj_fun.output(grids))
    print(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)

def test_call():
    obj_fun = class_fun()
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [-6.79741166e-02],
        [-1.36461045e-01],
        [-9.13324430e-02],
        [-9.71082067e-02],
        [-1.85406663e-01],
        [-9.02038776e-02],
        [-3.09547170e-02],
        [-7.29043824e-02],
        [-8.47693855e-02],
        [-1.80480228e-02],
        [-8.39060933e-01],
        [-1.99426284e+00],
        [-2.57290548e-02],
        [-6.28022015e-01],
        [-1.95703928e+00],
        [-8.19356834e-03],
        [-2.25915245e-01],
        [-1.82665019e+00],
        [-2.73536768e-04],
        [-2.26230774e+00],
        [-3.34829168e-01],
        [-2.04016877e-04],
        [-1.48565839e+00],
        [-3.25957854e-01],
        [-3.77271851e-05],
        [-2.24631259e-01],
        [-3.00476074e-01],
    ])

    print(grids)
    print(obj_fun(grids))
    print(np.abs(obj_fun(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun(grids) - truths_grids) < TEST_EPSILON)

def test_name():
    obj_fun = class_fun()
    assert obj_fun.name == str_name

    assert obj_fun.__class__.__name__.lower() == str_name
    assert obj_fun.__class__.__qualname__.lower() == str_name
