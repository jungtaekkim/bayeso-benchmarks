#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 4, 2023
#

import numpy as np
import pytest

from bayeso_benchmarks.inf_dim_zakharov import *

class_fun = Zakharov
str_name = 'zakharov'

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
        [391350.],
        [10181.25],
        [825.],
        [36014.94140625],
        [66.50390625],
        [70149.31640625],
        [220.3125],
        [24726.5625],
        [572920.3125],
        [204441.50390625],
        [1627.44140625],
        [6094.62890625],
        [10162.5],
        [693.75],
        [160537.5],
        [160.25390625],
        [70130.56640625],
        [954882.12890625],
        [94270.3125],
        [201.5625],
        [24820.3125],
        [1721.19140625],
        [6075.87890625],
        [318961.81640625],
        [900.],
        [160631.25],
        [1502175.],
        [94176.5625],
        [107.8125],
        [24726.5625],
        [1627.44140625],
        [5982.12890625],
        [318868.06640625],
        [806.25],
        [160537.5],
        [1502081.25],
        [35996.19140625],
        [47.75390625],
        [70130.56640625],
        [89.0625],
        [24595.3125],
        [572789.0625],
        [6075.87890625],
        [318849.31640625],
        [2256404.00390625],
        [10256.25],
        [787.5],
        [160631.25],
        [141.50390625],
        [70111.81640625],
        [954863.37890625],
        [24801.5625],
        [572882.8125],
        [3264651.5625],
        [10275.],
        [806.25],
        [160650.],
        [160.25390625],
        [70130.56640625],
        [954882.12890625],
        [24820.3125],
        [572901.5625],
        [3264670.3125],
        [1721.19140625],
        [6075.87890625],
        [318961.81640625],
        [787.5],
        [160518.75],
        [1502062.5],
        [70224.31640625],
        [954863.37890625],
        [4578033.69140625],
        [295.3125],
        [24801.5625],
        [572995.3125],
        [6169.62890625],
        [318943.06640625],
        [2256497.75390625],
        [160725.],
        [1502156.25],
        [6252900.],
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
        [3270.3125],
        [243.06640625],
        [125.],
        [31.25],
        [224.31640625],
        [3326.5625],
        [3345.3125],
        [16250.87890625],
        [51050.],
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
        [3.27231250e+03],
        [2.45066406e+02],
        [1.27000000e+02],
        [3.32500000e+01],
        [2.26316406e+02],
        [3.32856250e+03],
        [3.34731250e+03],
        [1.62528789e+04],
        [5.10520000e+04],
    ])

    print(grids)
    print(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE))
    print(np.abs(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE) - truths_grids) < TEST_EPSILON + SCALE_NOISE)
    assert np.all(np.abs(obj_fun.output_constant_noise(grids, scale_noise=SCALE_NOISE) - truths_grids) < TEST_EPSILON + SCALE_NOISE)

def test_output_gaussian_noise():
    obj_fun = class_fun(4, seed=SEED)
    bounds = obj_fun.get_bounds()

    grids = obj_fun.sample_grids(3)
    truths_grids = np.array([
        [391350.99342831],
        [10180.9734714],
        [826.29537708],
        [36017.98746596],
        [66.0355995],
        [70148.84813234],
        [223.47092563],
        [24728.09736946],
        [572919.37355123],
        [204442.58902634],
        [1626.51457086],
        [6093.69744674],
        [10162.98392454],
        [689.92343951],
        [160534.05016433],
        [159.12933119],
        [70128.54074401],
        [954882.75740092],
        [94268.49645185],
        [198.7378926],
        [24823.24379754],
        [1720.73985365],
        [6076.01396266],
        [318958.96690988],
        [898.91123455],
        [160631.47184518],
        [1502172.69801285],
        [94177.31389604],
        [106.61122262],
        [24725.9791125],
        [1626.23799303],
        [5985.83346262],
        [318868.0394118],
        [804.13457814],
        [160539.14508982],
        [1502078.8083127],
        [35996.60913344],
        [43.834566],
        [70127.91003415],
        [89.45622247],
        [24596.78943316],
        [572789.40523656],
        [6075.64760969],
        [318848.71419886],
        [2256401.04686227],
        [10254.81031158],
        [786.57872246],
        [160633.36424445],
        [142.19114283],
        [70108.29032594],
        [954864.02707419],
        [24800.79233544],
        [572881.458656],
        [3264652.78585258],
        [10277.06199904],
        [808.11256024],
        [160648.32156495],
        [159.6354815],
        [70131.22893311],
        [954884.0799965],
        [24819.35415152],
        [572901.19118205],
        [3264668.09983005],
        [1718.798993],
        [6077.50395789],
        [318964.52888631],
        [787.35597976],
        [160520.7570658],
        [1502063.22327205],
        [70223.02616674],
        [954864.10169746],
        [4578036.76747938],
        [295.24084792],
        [24804.69178731],
        [572990.07300979],
        [6171.27271126],
        [318943.24050039],
        [2256497.15589155],
        [160725.18352155],
        [1502152.27486217],
        [6252899.56065622],
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
        [391350.],
        [10181.25],
        [826.29537708],
        [36017.98746596],
        [66.0355995],
        [70148.84813234],
        [220.3125],
        [24726.5625],
        [572920.3125],
        [204441.50390625],
        [1627.44140625],
        [6093.69744674],
        [10162.5],
        [693.75],
        [160537.5],
        [160.25390625],
        [70130.56640625],
        [954882.75740092],
        [94268.49645185],
        [201.5625],
        [24820.3125],
        [1721.19140625],
        [6076.01396266],
        [318961.81640625],
        [900.],
        [160631.47184518],
        [1502172.69801285],
        [94176.5625],
        [107.8125],
        [24726.5625],
        [1627.44140625],
        [5982.12890625],
        [318868.06640625],
        [806.25],
        [160537.5],
        [1502078.8083127],
        [35996.19140625],
        [47.75390625],
        [70127.91003415],
        [89.45622247],
        [24595.3125],
        [572789.0625],
        [6075.64760969],
        [318848.71419886],
        [2256404.00390625],
        [10254.81031158],
        [786.57872246],
        [160631.25],
        [141.50390625],
        [70108.29032594],
        [954863.37890625],
        [24801.5625],
        [572881.458656],
        [3264651.5625],
        [10275.],
        [806.25],
        [160650.],
        [160.25390625],
        [70131.22893311],
        [954882.12890625],
        [24820.3125],
        [572901.19118205],
        [3264668.09983005],
        [1721.19140625],
        [6075.87890625],
        [318964.52888631],
        [787.5],
        [160520.7570658],
        [1502062.5],
        [70223.02616674],
        [954863.37890625],
        [4578033.69140625],
        [295.3125],
        [24804.69178731],
        [572995.3125],
        [6171.27271126],
        [318943.06640625],
        [2256497.75390625],
        [160725.18352155],
        [1502156.25],
        [6252900.],
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
        [71.4617346],
        [8.09052578],
        [752.63811703],
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
        [3271.4617346],
        [240.90302578],
        [125.],
        [31.25],
        [224.31640625],
        [3326.5625],
        [3341.38420756],
        [16250.87890625],
        [51050.],
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
