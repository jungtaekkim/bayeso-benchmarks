#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 6, 2023
#

import numpy as np
import pytest

from bayeso_benchmarks.inf_dim_griewank import *

class_fun = Griewank
str_name = 'griewank'

TEST_EPSILON = 1e-5
SCALE_NOISE = 2.0
SEED = 42


def test_init():
    obj_fun = class_fun(16)

    with pytest.raises(AssertionError) as error:
        class_fun(1.0)
    with pytest.raises(AssertionError) as error:
        class_fun('abc')
    with pytest.raises(AssertionError) as error:
        class_fun(4, seed=1.0)
    with pytest.raises(AssertionError) as error:
        class_fun(4, seed='abc')

def test_validate_properties():
    obj_fun = class_fun(8)
    obj_fun.validate_properties()

def test_output():
    obj_fun = class_fun(4)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [361.01465247],
        [270.33689088],
        [361.01465247],
        [271.02183025],
        [180.01205465],
        [271.02183025],
        [361.01465247],
        [270.33689088],
        [361.01465247],
        [270.98533321],
        [181.66375729],
        [270.98533321],
        [180.97814841],
        [91.98891104],
        [180.97814841],
        [270.98533321],
        [181.66375729],
        [270.98533321],
        [361.01465247],
        [270.33689088],
        [361.01465247],
        [271.02183025],
        [180.01205465],
        [271.02183025],
        [361.01465247],
        [270.33689088],
        [361.01465247],
        [270.98518323],
        [181.67054476],
        [270.98518323],
        [180.97792496],
        [91.99902348],
        [180.97792496],
        [270.98518323],
        [181.67054476],
        [270.98518323],
        [181.01483126],
        [90.3287998],
        [181.01483126],
        [91.02209662],
        [0.],
        [91.02209662],
        [181.01483126],
        [90.3287998],
        [181.01483126],
        [270.98518323],
        [181.67054476],
        [270.98518323],
        [180.97792496],
        [91.99902348],
        [180.97792496],
        [270.98518323],
        [181.67054476],
        [270.98518323],
        [361.01465247],
        [270.33689088],
        [361.01465247],
        [271.02183025],
        [180.01205465],
        [271.02183025],
        [361.01465247],
        [270.33689088],
        [361.01465247],
        [270.98533321],
        [181.66375729],
        [270.98533321],
        [180.97814841],
        [91.98891104],
        [180.97814841],
        [270.98533321],
        [181.66375729],
        [270.98533321],
        [361.01465247],
        [270.33689088],
        [361.01465247],
        [271.02183025],
        [180.01205465],
        [271.02183025],
        [361.01465247],
        [270.33689088],
        [361.01465247],
    ])
    outputs = obj_fun.output(grids)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)

def test_call():
    obj_fun = class_fun(2)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [180.01205465],
        [91.98891104],
        [180.01205465],
        [91.99902348],
        [0.],
        [91.99902348],
        [180.01205465],
        [91.98891104],
        [180.01205465],
    ])
    outputs = obj_fun(grids)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)

def test_output_constant_noise():
    obj_fun = class_fun(2)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [182.01205465],
        [93.98891104],
        [182.01205465],
        [93.99902348],
        [2.],
        [93.99902348],
        [182.01205465],
        [93.98891104],
        [182.01205465],
    ])
    outputs = obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON + SCALE_NOISE)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON + SCALE_NOISE)

def test_output_gaussian_noise():
    obj_fun = class_fun(4, seed=SEED)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [362.00808078],
        [270.06036228],
        [362.31002955],
        [274.06788997],
        [179.5437479],
        [270.55355634],
        [364.1730781],
        [271.87176034],
        [360.0757037],
        [272.07045329],
        [180.73692191],
        [270.0538737],
        [181.46207295],
        [88.16235055],
        [177.52831274],
        [269.86075815],
        [179.63809505],
        [271.61382787],
        [359.19860432],
        [267.51228348],
        [363.94595001],
        [270.57027765],
        [180.14711106],
        [268.17233388],
        [359.92588702],
        [270.55873606],
        [358.71266531],
        [271.73657926],
        [180.46926738],
        [270.40179573],
        [179.77451173],
        [95.70357985],
        [180.95093051],
        [268.86976137],
        [183.31563459],
        [268.54349593],
        [181.43255845],
        [86.40945955],
        [178.35845916],
        [91.41581909],
        [1.47693316],
        [91.36483318],
        [180.78353469],
        [89.72659241],
        [178.05778727],
        [269.54549481],
        [180.74926722],
        [273.09942768],
        [181.66516154],
        [88.47294317],
        [181.6260929],
        [270.21501867],
        [180.31670076],
        [272.20853581],
        [363.07665151],
        [272.19945112],
        [359.33621742],
        [270.4034055],
        [180.67458151],
        [272.97292051],
        [360.05630399],
        [269.96557293],
        [358.80198252],
        [268.59291996],
        [183.28880894],
        [273.69781326],
        [180.83412817],
        [93.99597684],
        [181.70142046],
        [269.6950937],
        [182.3865485],
        [274.06140634],
        [360.94300039],
        [273.46617819],
        [355.77516226],
        [272.66563526],
        [180.18614879],
        [270.42381555],
        [361.19817402],
        [266.36175305],
        [360.57530869],
    ])
    outputs = obj_fun.output_gaussian_noise(grids, scale_noise=SCALE_NOISE)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)

def test_output_sparse_gaussian_noise():
    obj_fun = class_fun(4, seed=SEED)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [361.01465247],
        [270.33689088],
        [362.31002955],
        [274.06788997],
        [179.5437479],
        [270.55355634],
        [361.01465247],
        [270.33689088],
        [361.01465247],
        [270.98533321],
        [181.66375729],
        [270.0538737],
        [180.97814841],
        [91.98891104],
        [180.97814841],
        [270.98533321],
        [181.66375729],
        [271.61382787],
        [359.19860432],
        [270.33689088],
        [361.01465247],
        [271.02183025],
        [180.14711106],
        [271.02183025],
        [361.01465247],
        [270.55873606],
        [358.71266531],
        [270.98518323],
        [181.67054476],
        [270.98518323],
        [180.97792496],
        [91.99902348],
        [180.97792496],
        [270.98518323],
        [181.67054476],
        [268.54349593],
        [181.01483126],
        [90.3287998],
        [178.35845916],
        [91.41581909],
        [0.],
        [91.02209662],
        [180.78353469],
        [89.72659241],
        [181.01483126],
        [269.54549481],
        [180.74926722],
        [270.98518323],
        [180.97792496],
        [88.47294317],
        [180.97792496],
        [270.98518323],
        [180.31670076],
        [270.98518323],
        [361.01465247],
        [270.33689088],
        [361.01465247],
        [271.02183025],
        [180.67458151],
        [271.02183025],
        [361.01465247],
        [269.96557293],
        [358.80198252],
        [270.98533321],
        [181.66375729],
        [273.69781326],
        [180.97814841],
        [93.99597684],
        [180.97814841],
        [269.6950937],
        [181.66375729],
        [270.98533321],
        [361.01465247],
        [273.46617819],
        [361.01465247],
        [272.66563526],
        [180.01205465],
        [271.02183025],
        [361.19817402],
        [270.33689088],
        [361.01465247],
    ])
    outputs = obj_fun.output_sparse_gaussian_noise(grids, scale_noise=SCALE_NOISE, sparsity=0.3)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)

def test_output_student_t_noise():
    obj_fun = class_fun(1, seed=SEED)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [93.14825808],
        [-2.16338047],
        [94.63714051],
    ])
    outputs = obj_fun.output_student_t_noise(grids, scale_noise=SCALE_NOISE, dof=4.0)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)

def test_output_sparse_student_t_noise():
    obj_fun = class_fun(2, seed=SEED)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [181.16128925],
        [89.82553057],
        [180.01205465],
        [91.99902348],
        [0.],
        [91.99902348],
        [176.08376221],
        [91.98891104],
        [180.01205465],
    ])
    outputs = obj_fun.output_sparse_student_t_noise(grids, scale_noise=SCALE_NOISE, dof=4.0, sparsity=0.3)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)
    assert np.all(np.abs(outputs - truths_grids) < TEST_EPSILON)

def test_name():
    obj_fun = class_fun(2)
    assert obj_fun.name == str_name + '_2'

    obj_fun = class_fun(4)
    assert obj_fun.name == str_name + '_4'

    obj_fun = class_fun(16)
    assert obj_fun.name == str_name + '_16'

    assert obj_fun.__class__.__name__.lower() == str_name
    assert obj_fun.__class__.__qualname__.lower() == str_name
