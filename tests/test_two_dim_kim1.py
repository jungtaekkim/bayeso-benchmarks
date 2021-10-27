#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 27, 2021
#

import numpy as np
import pytest

from bayeso_benchmarks.two_dim_kim1 import *

class_fun = Kim1

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
        [9.91424384],
        [2.97034052],
        [4.2184372],
        [8.54390332],
        [1.6],
        [2.84809668],
        [7.35424384],
        [0.41034052],
        [1.6584372],
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
        [11.91424384],
        [4.97034052],
        [6.2184372],
        [10.54390332],
        [3.6],
        [4.84809668],
        [9.35424384],
        [2.41034052],
        [3.6584372],
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
        [10.90767214],
        [2.69381192],
        [5.51381428],
        [11.58996303],
        [1.13169325],
        [2.37982277],
        [10.51266947],
        [1.94520998],
        [0.71948843],
    ])
    outputs = obj_fun.output_gaussian_noise(grids, scale_noise=SCALE_NOISE)

    for elem in outputs:
        print('{},'.format(elem))

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)
