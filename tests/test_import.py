#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 4, 2023
#


def test_import_benchmarks():
    import bayeso_benchmarks

def test_all_benchmarks():
    import bayeso_benchmarks

    list_str_names = [
        'ackley',
        'cosines',
        'levy',
        'rastrigin',
        'rosenbrock',
        'sphere',
        'zakharov',
        'constant',
        'gramacyandlee2012',
        'linear',
        'step',
        'colville',
        'hartmann3d',
        'hartmann6d',
        'beale',
        'bohachevsky',
        'branin',
        'bukin6',
        'dejong5',
        'dropwave',
        'easom',
        'eggholder',
        'goldsteinprice',
        'holdertable',
        'kim1',
        'kim2',
        'kim3',
        'michalewicz',
        'shubert',
        'sixhumpcamel',
        'threehumpcamel',
    ]

    names = []
    for class_benchmark in bayeso_benchmarks.all_benchmarks:
        print(class_benchmark.__name__.lower())
        names.append(class_benchmark.__name__.lower())

    for str_name in list_str_names:
        assert str_name in names

    qualnames = []
    for class_benchmark in bayeso_benchmarks.all_benchmarks:
        print(class_benchmark.__qualname__.lower())
        qualnames.append(class_benchmark.__qualname__.lower())

    for str_name in list_str_names:
        assert str_name in qualnames

def test_num_benchmarks():
    import bayeso_benchmarks

    assert bayeso_benchmarks.num_benchmarks == 32

def test_import_benchmark_base():
    import bayeso_benchmarks.benchmark_base
    from bayeso_benchmarks.benchmark_base import Function

def test_import_one_dim_constant():
    import bayeso_benchmarks.one_dim_constant
    from bayeso_benchmarks.one_dim_constant import Constant
    from bayeso_benchmarks import Constant

def test_import_one_dim_gramacyandlee2012():
    import bayeso_benchmarks.one_dim_gramacyandlee2012
    from bayeso_benchmarks.one_dim_gramacyandlee2012 import GramacyAndLee2012
    from bayeso_benchmarks import GramacyAndLee2012

def test_import_one_dim_linear():
    import bayeso_benchmarks.one_dim_linear
    from bayeso_benchmarks.one_dim_linear import Linear
    from bayeso_benchmarks import Linear

def test_import_one_dim_step():
    import bayeso_benchmarks.one_dim_step
    from bayeso_benchmarks.one_dim_step import Step
    from bayeso_benchmarks import Step

def test_import_two_dim_beale():
    import bayeso_benchmarks.two_dim_beale
    from bayeso_benchmarks.two_dim_beale import Beale
    from bayeso_benchmarks import Beale

def test_import_two_dim_bohachevsky():
    import bayeso_benchmarks.two_dim_bohachevsky
    from bayeso_benchmarks.two_dim_bohachevsky import Bohachevsky
    from bayeso_benchmarks import Bohachevsky

def test_import_two_dim_branin():
    import bayeso_benchmarks.two_dim_branin
    from bayeso_benchmarks.two_dim_branin import Branin
    from bayeso_benchmarks import Branin

def test_import_two_dim_bukin6():
    import bayeso_benchmarks.two_dim_bukin6
    from bayeso_benchmarks.two_dim_bukin6 import Bukin6
    from bayeso_benchmarks import Bukin6

def test_import_two_dim_dropwave():
    import bayeso_benchmarks.two_dim_dropwave
    from bayeso_benchmarks.two_dim_dropwave import DropWave
    from bayeso_benchmarks import DropWave

def test_import_two_dim_easom():
    import bayeso_benchmarks.two_dim_easom
    from bayeso_benchmarks.two_dim_easom import Easom
    from bayeso_benchmarks import Easom

def test_import_two_dim_eggholder():
    import bayeso_benchmarks.two_dim_eggholder
    from bayeso_benchmarks.two_dim_eggholder import Eggholder
    from bayeso_benchmarks import Eggholder

def test_import_two_dim_goldsteinprice():
    import bayeso_benchmarks.two_dim_goldsteinprice
    from bayeso_benchmarks.two_dim_goldsteinprice import GoldsteinPrice
    from bayeso_benchmarks import GoldsteinPrice

def test_import_two_dim_holdertable():
    import bayeso_benchmarks.two_dim_holdertable
    from bayeso_benchmarks.two_dim_holdertable import HolderTable
    from bayeso_benchmarks import HolderTable

def test_import_two_dim_kim1():
    import bayeso_benchmarks.two_dim_kim1
    from bayeso_benchmarks.two_dim_kim1 import Kim1
    from bayeso_benchmarks import Kim1

def test_import_two_dim_kim2():
    import bayeso_benchmarks.two_dim_kim2
    from bayeso_benchmarks.two_dim_kim2 import Kim2
    from bayeso_benchmarks import Kim2

def test_import_two_dim_kim3():
    import bayeso_benchmarks.two_dim_kim3
    from bayeso_benchmarks.two_dim_kim3 import Kim3
    from bayeso_benchmarks import Kim3

def test_import_two_dim_michalewicz():
    import bayeso_benchmarks.two_dim_michalewicz
    from bayeso_benchmarks.two_dim_michalewicz import Michalewicz
    from bayeso_benchmarks import Michalewicz

def test_import_two_dim_shubert():
    import bayeso_benchmarks.two_dim_shubert
    from bayeso_benchmarks.two_dim_shubert import Shubert
    from bayeso_benchmarks import Shubert

def test_import_two_dim_sixhumpcamel():
    import bayeso_benchmarks.two_dim_sixhumpcamel
    from bayeso_benchmarks.two_dim_sixhumpcamel import SixHumpCamel
    from bayeso_benchmarks import SixHumpCamel

def test_import_two_dim_threehumpcamel():
    import bayeso_benchmarks.two_dim_threehumpcamel
    from bayeso_benchmarks.two_dim_threehumpcamel import ThreeHumpCamel
    from bayeso_benchmarks import ThreeHumpCamel

def test_import_four_dim_colville():
    import bayeso_benchmarks.four_dim_colville
    from bayeso_benchmarks.four_dim_colville import Colville
    from bayeso_benchmarks import Colville

def test_import_three_dim_hartmann3d():
    import bayeso_benchmarks.three_dim_hartmann3d
    from bayeso_benchmarks.three_dim_hartmann3d import Hartmann3D
    from bayeso_benchmarks import Hartmann3D

def test_import_six_dim_hartmann6d():
    import bayeso_benchmarks.six_dim_hartmann6d
    from bayeso_benchmarks.six_dim_hartmann6d import Hartmann6D
    from bayeso_benchmarks import Hartmann6D

def test_import_inf_dim_ackley():
    import bayeso_benchmarks.inf_dim_ackley
    from bayeso_benchmarks.inf_dim_ackley import Ackley
    from bayeso_benchmarks import Ackley

def test_import_inf_dim_cosines():
    import bayeso_benchmarks.inf_dim_cosines
    from bayeso_benchmarks.inf_dim_cosines import Cosines
    from bayeso_benchmarks import Cosines

def test_import_inf_dim_griewank():
    import bayeso_benchmarks.inf_dim_griewank
    from bayeso_benchmarks.inf_dim_griewank import Griewank
    from bayeso_benchmarks import Griewank

def test_import_inf_dim_levy():
    import bayeso_benchmarks.inf_dim_levy
    from bayeso_benchmarks.inf_dim_levy import Levy
    from bayeso_benchmarks import Levy

def test_import_inf_dim_rastrigin():
    import bayeso_benchmarks.inf_dim_rastrigin
    from bayeso_benchmarks.inf_dim_rastrigin import Rastrigin
    from bayeso_benchmarks import Rastrigin

def test_import_inf_dim_rosenbrock():
    import bayeso_benchmarks.inf_dim_rosenbrock
    from bayeso_benchmarks.inf_dim_rosenbrock import Rosenbrock
    from bayeso_benchmarks import Rosenbrock

def test_import_inf_dim_sphere():
    import bayeso_benchmarks.inf_dim_sphere
    from bayeso_benchmarks.inf_dim_sphere import Sphere
    from bayeso_benchmarks import Sphere

def test_import_inf_dim_zakharov():
    import bayeso_benchmarks.inf_dim_zakharov
    from bayeso_benchmarks.inf_dim_zakharov import Zakharov
    from bayeso_benchmarks import Zakharov
