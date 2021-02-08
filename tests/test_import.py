#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#


def test_import_benchmarks():
    import bayeso_benchmarks

def test_import_benchmark_base():
    import bayeso_benchmarks.benchmark_base
    from bayeso_benchmarks.benchmark_base import Function

def test_import_one_dim_constant():
    import bayeso_benchmarks.one_dim_constant
    from bayeso_benchmarks.one_dim_constant import Constant

def test_import_one_dim_gramacyandlee2012():
    import bayeso_benchmarks.one_dim_gramacyandlee2012
    from bayeso_benchmarks.one_dim_gramacyandlee2012 import GramacyAndLee2012

def test_import_one_dim_linear():
    import bayeso_benchmarks.one_dim_linear
    from bayeso_benchmarks.one_dim_linear import Linear

def test_import_one_dim_step():
    import bayeso_benchmarks.one_dim_step
    from bayeso_benchmarks.one_dim_step import Step

def test_import_two_dim_beale():
    import bayeso_benchmarks.two_dim_beale
    from bayeso_benchmarks.two_dim_beale import Beale

def test_import_two_dim_bohachevsky():
    import bayeso_benchmarks.two_dim_bohachevsky
    from bayeso_benchmarks.two_dim_bohachevsky import Bohachevsky

def test_import_two_dim_branin():
    import bayeso_benchmarks.two_dim_branin
    from bayeso_benchmarks.two_dim_branin import Branin

def test_import_two_dim_eggholder():
    import bayeso_benchmarks.two_dim_eggholder
    from bayeso_benchmarks.two_dim_eggholder import Eggholder

def test_import_two_dim_goldsteinprice():
    import bayeso_benchmarks.two_dim_goldsteinprice
    from bayeso_benchmarks.two_dim_goldsteinprice import GoldsteinPrice

def test_import_two_dim_holdertable():
    import bayeso_benchmarks.two_dim_holdertable
    from bayeso_benchmarks.two_dim_holdertable import HolderTable

def test_import_two_dim_michalewicz():
    import bayeso_benchmarks.two_dim_michalewicz
    from bayeso_benchmarks.two_dim_michalewicz import Michalewicz

def test_import_two_dim_sixhumpcamel():
    import bayeso_benchmarks.two_dim_sixhumpcamel
    from bayeso_benchmarks.two_dim_sixhumpcamel import SixHumpCamel

def test_import_two_dim_threehumpcamel():
    import bayeso_benchmarks.two_dim_threehumpcamel
    from bayeso_benchmarks.two_dim_threehumpcamel import ThreeHumpCamel

def test_import_three_dim_hartmann3d():
    import bayeso_benchmarks.three_dim_hartmann3d
    from bayeso_benchmarks.three_dim_hartmann3d import Hartmann3D

def test_import_six_dim_hartmann6d():
    import bayeso_benchmarks.six_dim_hartmann6d
    from bayeso_benchmarks.six_dim_hartmann6d import Hartmann6D

def test_import_inf_dim_ackley():
    import bayeso_benchmarks.inf_dim_ackley
    from bayeso_benchmarks.inf_dim_ackley import Ackley

def test_import_inf_dim_rosenbrock():
    import bayeso_benchmarks.inf_dim_rosenbrock
    from bayeso_benchmarks.inf_dim_rosenbrock import Rosenbrock

def test_import_inf_dim_sphere():
    import bayeso_benchmarks.inf_dim_sphere
    from bayeso_benchmarks.inf_dim_sphere import Sphere
