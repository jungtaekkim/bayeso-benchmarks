#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 4, 2022
#

import numpy as np
import pytest

from bayeso_benchmarks.two_dim_branin import *

class_fun = Branin
str_name = 'branin'

TEST_EPSILON = 1e-5
SCALE_NOISE = 2.0
SEED = 42


def test_init():
    obj_fun = class_fun()

    with pytest.raises(AssertionError) as error:
        class_fun(a='abc')
    with pytest.raises(AssertionError) as error:
        class_fun(b='abc')
    with pytest.raises(AssertionError) as error:
        class_fun(c='abc')
    with pytest.raises(AssertionError) as error:
        class_fun(r='abc')
    with pytest.raises(AssertionError) as error:
        class_fun(s='abc')
    with pytest.raises(AssertionError) as error:
        class_fun(t='abc')
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

def test_call():
    obj_fun = class_fun()
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
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
    print(obj_fun(grids))
    print(np.abs(obj_fun(grids) - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(obj_fun(grids) - truths_grids) < TEST_EPSILON)

def test_output_constant_noise():
    obj_fun = class_fun()
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
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
    print(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE))
    print(np.abs(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE) - truths_grids) < TEST_EPSILON + SCALE_NOISE)
    assert np.all(np.abs(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE) - truths_grids) < TEST_EPSILON + SCALE_NOISE)

def test_output_gaussian_noise():
    obj_fun = class_fun(seed=SEED)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [309.12252432],
        [10.03137988],
        [12.25626611],
        [109.61475748],
        [23.66165766],
        [21.69826604],
        [20.66672515],
        [151.9868898],
        [144.93324211],
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
        [309.12252432],
        [10.03137988],
        [10.96088904],
        [106.56869776],
        [24.12996441],
        [21.69826604],
        [17.50829952],
        [151.9868898],
        [144.93324211],
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
        [309.27833061],
        [8.14452801],
        [13.59900607],
        [104.98514624],
        [25.60278992],
        [25.95324732],
        [13.58000707],
        [149.34002319],
        [144.72897096],
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
        [309.27833061],
        [8.14452801],
        [10.96088904],
        [106.56869776],
        [24.12996441],
        [22.16653996],
        [13.58000707],
        [150.45202034],
        [145.87219088],
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
