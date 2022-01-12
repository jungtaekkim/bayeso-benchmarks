#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 27, 2021
#

import numpy as np
import pytest

from bayeso_benchmarks.two_dim_kim2 import *

class_fun = Kim2

TEST_EPSILON = 1e-5
SCALE_NOISE = 2.0
SEED = 42


def test_init():
    obj_fun = class_fun()

    with pytest.raises(AssertionError) as error:
        class_fun(seed=1.0)
    with pytest.raises(AssertionError) as error:
        class_fun(seed='abc')

def test_validate_properties():
    obj_fun = class_fun()
    obj_fun.validate_properties()

def test_output():
    obj_fun = class_fun()
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [101.67527989],
        [35.75642525],
        [74.69517061],
        [72.83885464],
        [6.92],
        [45.85874536],
        [85.29127989],
        [19.37242525],
        [58.31117061],
    ])

    for elem in obj_fun.output(grids):
        print('{},'.format(elem))

    print(grids)
    print(obj_fun.output(grids))
    print(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)

def test_output_constant_noise():
    obj_fun = class_fun()
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [103.67527989],
        [37.75642525],
        [76.69517061],
        [74.83885464],
        [8.92],
        [47.85874536],
        [87.29127989],
        [21.37242525],
        [60.31117061],
    ])

    for elem in obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE):
        print('{},'.format(elem))

    print(grids)
    print(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE))
    print(np.abs(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE) - truths_grids) < TEST_EPSILON + SCALE_NOISE)
    assert np.all(np.abs(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE) - truths_grids) < TEST_EPSILON + SCALE_NOISE)

def test_output_gaussian_noise():
    obj_fun = class_fun(seed=SEED)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [102.6687082],
        [35.47989665],
        [75.99054769],
        [75.88491435],
        [6.45169325],
        [45.39047145],
        [88.44970553],
        [20.90729471],
        [57.37222184],
    ])
    outputs = obj_fun.output_gaussian_noise(grids, scale_noise=SCALE_NOISE)

    for elem in outputs:
        print('{},'.format(elem))

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)
