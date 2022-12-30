#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 30, 2022
#

import numpy as np
import pytest

from bayeso_benchmarks.four_dim_colville import *

class_fun = Colville
str_name = 'colville'

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
        [2304082.],
        [2111692.],
        [1939322.],
        [1223962.],
        [1211572.],
        [1219202.],
        [2304042.],
        [2111652.],
        [1939282.],
        [1103962.],
        [911572.],
        [739202.],
        [23842.],
        [11452.],
        [19082.],
        [1103922.],
        [911532.],
        [739162.],
        [2304042.],
        [2111652.],
        [1939282.],
        [1223922.],
        [1211532.],
        [1219162.],
        [2304002.],
        [2111612.],
        [1939242.],
        [2090692.],
        [1900282.],
        [1729892.],
        [1010572.],
        [1000162.],
        [1009772.],
        [2090652.],
        [1900242.],
        [1729852.],
        [1090572.],
        [900162.],
        [729772.],
        [10452.],
        [42.],
        [9652.],
        [1090532.],
        [900122.],
        [729732.],
        [2090652.],
        [1900242.],
        [1729852.],
        [1010532.],
        [1000122.],
        [1009732.],
        [2090612.],
        [1900202.],
        [1729812.],
        [1899322.],
        [1710892.],
        [1542482.],
        [819202.],
        [810772.],
        [822362.],
        [1899282.],
        [1710852.],
        [1542442.],
        [1099202.],
        [910772.],
        [742362.],
        [19082.],
        [10652.],
        [22242.],
        [1099162.],
        [910732.],
        [742322.],
        [1899282.],
        [1710852.],
        [1542442.],
        [819162.],
        [810732.],
        [822322.],
        [1899242.],
        [1710812.],
        [1542402.],
    ])
    outputs = obj_fun(grids)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)

    assert np.all(np.abs(obj_fun.output(grids) - truths_grids) < TEST_EPSILON)

def test_call():
    obj_fun = class_fun()
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [2304082.],
        [2111692.],
        [1939322.],
        [1223962.],
        [1211572.],
        [1219202.],
        [2304042.],
        [2111652.],
        [1939282.],
        [1103962.],
        [911572.],
        [739202.],
        [23842.],
        [11452.],
        [19082.],
        [1103922.],
        [911532.],
        [739162.],
        [2304042.],
        [2111652.],
        [1939282.],
        [1223922.],
        [1211532.],
        [1219162.],
        [2304002.],
        [2111612.],
        [1939242.],
        [2090692.],
        [1900282.],
        [1729892.],
        [1010572.],
        [1000162.],
        [1009772.],
        [2090652.],
        [1900242.],
        [1729852.],
        [1090572.],
        [900162.],
        [729772.],
        [10452.],
        [42.],
        [9652.],
        [1090532.],
        [900122.],
        [729732.],
        [2090652.],
        [1900242.],
        [1729852.],
        [1010532.],
        [1000122.],
        [1009732.],
        [2090612.],
        [1900202.],
        [1729812.],
        [1899322.],
        [1710892.],
        [1542482.],
        [819202.],
        [810772.],
        [822362.],
        [1899282.],
        [1710852.],
        [1542442.],
        [1099202.],
        [910772.],
        [742362.],
        [19082.],
        [10652.],
        [22242.],
        [1099162.],
        [910732.],
        [742322.],
        [1899282.],
        [1710852.],
        [1542442.],
        [819162.],
        [810732.],
        [822322.],
        [1899242.],
        [1710812.],
        [1542402.],
    ])
    outputs = obj_fun(grids)

    print(grids)
    print(outputs)
    print(np.abs(outputs - truths_grids) < TEST_EPSILON)

    assert np.all(np.abs(obj_fun(grids) - truths_grids) < TEST_EPSILON)

def test_name():
    obj_fun = class_fun()
    assert obj_fun.name == str_name

    assert obj_fun.__class__.__name__.lower() == str_name
    assert obj_fun.__class__.__qualname__.lower() == str_name
