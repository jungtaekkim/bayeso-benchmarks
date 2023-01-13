#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 3, 2023
#

import numpy as np
import pytest

from bayeso_benchmarks.two_dim_shubert import *

class_fun = Shubert
str_name = 'shubert'

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
        [0.06674108],
        [1.15175294],
        [0.86375707],
        [1.15175294],
        [19.87583625],
        [14.90588261],
        [0.86375707],
        [14.90588261],
        [11.17866608],
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
        [0.06674108],
        [1.15175294],
        [0.86375707],
        [1.15175294],
        [19.87583625],
        [14.90588261],
        [0.86375707],
        [14.90588261],
        [11.17866608],
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
        [2.06674108],
        [3.15175294],
        [2.86375707],
        [3.15175294],
        [21.87583625],
        [16.90588261],
        [2.86375707],
        [16.90588261],
        [13.17866608],
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
        [1.06016939],
        [0.87522434],
        [2.15913415],
        [4.19781266],
        [19.4075295],
        [14.4376087],
        [4.02218271],
        [16.44075207],
        [10.2397173],
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
        [1.06016939],
        [0.87522434],
        [0.86375707],
        [1.15175294],
        [19.87583625],
        [14.4376087],
        [0.86375707],
        [16.44075207],
        [10.2397173],
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
        [1.21597568],
        [-1.01162753],
        [3.50187411],
        [-0.43179858],
        [21.34866175],
        [18.69258998],
        [-3.06453537],
        [13.79388546],
        [10.03544615],
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
        [1.21597568],
        [-1.01162753],
        [0.86375707],
        [1.15175294],
        [19.87583625],
        [14.90588261],
        [-3.06453537],
        [14.90588261],
        [11.17866608],
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
