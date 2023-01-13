#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 3, 2023
#

import numpy as np
import pytest

from bayeso_benchmarks.two_dim_easom import *

class_fun = Easom
str_name = 'easom'

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
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [-2.67528799e-09],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
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
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [-2.67528799e-09],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
    ])

    print(grids)
    print(obj_fun(grids))
    print(np.abs(obj_fun(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun(grids) - truths_grids) < TEST_EPSILON)

def test_output_constant_noise():
    obj_fun = class_fun()
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [2.0],
        [2.0],
        [2.0],
        [2.0],
        [2.0],
        [2.0],
        [2.0],
        [2.0],
        [2.0],
    ])

    print(grids)
    print(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE))
    print(np.abs(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE) - truths_grids) < TEST_EPSILON + SCALE_NOISE)
    assert np.all(np.abs(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE) - truths_grids) < TEST_EPSILON + SCALE_NOISE)

def test_output_gaussian_noise():
    obj_fun = class_fun(seed=SEED)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [0.99342831],
        [-0.2765286],
        [1.29537708],
        [3.04605971],
        [-0.46830675],
        [-0.46827391],
        [3.15842563],
        [1.53486946],
        [-0.93894877],
    ])
    outputs = obj_fun.output_gaussian_noise(grids, scale_noise=SCALE_NOISE)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)

def test_output_sparse_gaussian_noise():
    obj_fun = class_fun(seed=SEED)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [9.93428306e-01],
        [-2.76528602e-01],
        [0.0],
        [0.0],
        [-2.67528799e-09],
        [-4.68273914e-01],
        [0.0],
        [1.53486946],
        [-9.38948772e-01],
    ])
    outputs = obj_fun.output_sparse_gaussian_noise(grids, scale_noise=SCALE_NOISE, sparsity=0.3)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)

def test_output_student_t_noise():
    obj_fun = class_fun(seed=SEED)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [1.1492346],
        [-2.16338047],
        [2.63811703],
        [-1.58355153],
        [1.4728255],
        [3.78670737],
        [-3.92829244],
        [-1.11199715],
        [-1.14321992],
    ])
    outputs = obj_fun.output_student_t_noise(grids, scale_noise=SCALE_NOISE, dof=4.0)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)

def test_output_sparse_student_t_noise():
    obj_fun = class_fun(seed=SEED)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [1.14923460],
        [-2.16338047],
        [0.0],
        [0.0],
        [-2.67528799e-09],
        [0.0],
        [-3.92829244],
        [0.0],
        [0.0],
    ])
    outputs = obj_fun.output_sparse_student_t_noise(grids, scale_noise=SCALE_NOISE, dof=4.0, sparsity=0.3)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)

def test_name():
    obj_fun = class_fun()
    assert obj_fun.name == str_name

    assert obj_fun.__class__.__name__.lower() == str_name
    assert obj_fun.__class__.__qualname__.lower() == str_name
