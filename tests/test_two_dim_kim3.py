#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 4, 2022
#

import numpy as np
import pytest

from bayeso_benchmarks.two_dim_kim3 import *

class_fun = Kim3
str_name = 'kim3'

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
        [208.90875507],
        [72.1885316],
        [145.18351361],
        [145.56021506],
        [8.83995458],
        [81.83497908],
        [176.14075507],
        [39.4205316],
        [112.41551361],
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
        [210.90875507],
        [74.1885316],
        [147.18351361],
        [147.56021506],
        [10.83995458],
        [83.83497908],
        [178.14075507],
        [41.4205316],
        [114.41551361],
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
        [209.90218338],
        [71.912003],
        [146.47889068],
        [148.60627477],
        [8.37164783],
        [81.36670517],
        [179.2991807],
        [40.95540106],
        [111.47656484],
    ])
    outputs = obj_fun.output_gaussian_noise(grids, scale_noise=SCALE_NOISE)

    for elem in outputs:
        print('{},'.format(elem))

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)

def test_name():
    obj_fun = class_fun()
    assert obj_fun.name == str_name

    assert obj_fun.__class__.__name__.lower() == str_name
    assert obj_fun.__class__.__qualname__.lower() == str_name
