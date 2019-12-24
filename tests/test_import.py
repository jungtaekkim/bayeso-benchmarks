def test_import_benchmarks():
    import benchmarks

def test_import_benchmark_base():
    import benchmarks.benchmark_base
    from benchmarks.benchmark_base import Function

def test_import_one_dim_linear():
    import benchmarks.one_dim_linear
    from benchmarks.one_dim_linear import Linear

def test_import_two_dim_branin():
    import benchmarks.two_dim_branin
    from benchmarks.two_dim_branin import Branin

def test_import_two_dim_eggholder():
    import benchmarks.two_dim_eggholder
    from benchmarks.two_dim_eggholder import Eggholder

def test_import_two_dim_holdertable():
    import benchmarks.two_dim_holdertable
    from benchmarks.two_dim_holdertable import HolderTable

def test_import_two_dim_sixhumpcamel():
    import benchmarks.two_dim_sixhumpcamel
    from benchmarks.two_dim_sixhumpcamel import SixHumpCamel

def test_import_six_dim_hartmann6d():
    import benchmarks.six_dim_hartmann6d
    from benchmarks.six_dim_hartmann6d import Hartmann6D

