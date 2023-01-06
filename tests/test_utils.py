#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 6, 2023
#

import numpy as np
import pytest

from bayeso_benchmarks import utils

TEST_EPSILON = 1e-5


def test_get_benchmark():
    with pytest.raises(TypeError) as error:
        benchmark = utils.get_benchmark()
    with pytest.raises(ValueError) as error:
        benchmark = utils.get_benchmark('abc', seed=None)

    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('ackley')
    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('ackley', seed='abc')

    benchmark = utils.get_benchmark('ackley', dim=4, seed=42)
    print(benchmark.output(np.array([0.0, 0.0, 0.0, 0.0])))

    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('cosines')

    benchmark = utils.get_benchmark('cosines', dim=4, seed=None)

    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('griewank')

    benchmark = utils.get_benchmark('griewank', dim=4, seed=None)

    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('levy')

    benchmark = utils.get_benchmark('levy', dim=2, seed=None)

    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('rastrigin')

    benchmark = utils.get_benchmark('rastrigin', dim=8, seed=None)

    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('rosenbrock')

    benchmark = utils.get_benchmark('rosenbrock', dim=8, seed=None)

    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('sphere')

    benchmark = utils.get_benchmark('sphere', dim=16, seed=None)

    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('zakharov')

    benchmark = utils.get_benchmark('zakharov', dim=16, seed=None)

    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('constant')
    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('constant', constant=None)
    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('constant', bounds=None)
    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('constant', bounds=np.array([0.0, 10.0]), constant=10.0, seed=None)

    benchmark = utils.get_benchmark('constant', bounds=np.array([[0.0, 10.0]]), constant=10.0, seed=None)

    benchmark = utils.get_benchmark('gramacyandlee2012')

    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('linear')

    benchmark = utils.get_benchmark('linear', bounds=np.array([[0.0, 10.0]]), slope=-1.2, seed=None)

    with pytest.raises(AssertionError) as error:
        benchmark = utils.get_benchmark('step')

    benchmark = utils.get_benchmark('step', steps=[0.0, 3.0, 7.0, 10.0], step_values=[-2.1, 4.0, 10.0], seed=None)

    benchmark = utils.get_benchmark('beale')
    benchmark = utils.get_benchmark('bohachevsky')

    benchmark = utils.get_benchmark('branin')
    print(benchmark.output(np.array([1.0, 1.0])))

    benchmark = utils.get_benchmark('bukin6')
    benchmark = utils.get_benchmark('dejong5')
    benchmark = utils.get_benchmark('dropwave')
    benchmark = utils.get_benchmark('easom')
    benchmark = utils.get_benchmark('eggholder')
    benchmark = utils.get_benchmark('goldsteinprice')
    benchmark = utils.get_benchmark('holdertable')
    benchmark = utils.get_benchmark('kim1')
    benchmark = utils.get_benchmark('kim2')
    benchmark = utils.get_benchmark('kim3')
    benchmark = utils.get_benchmark('michalewicz')
    benchmark = utils.get_benchmark('shubert')
    benchmark = utils.get_benchmark('sixhumpcamel')
    benchmark = utils.get_benchmark('threehumpcamel')

    benchmark = utils.get_benchmark('colville')
    benchmark = utils.get_benchmark('hartmann3d')
    benchmark = utils.get_benchmark('hartmann6d')

def test_pdf_two_dim_normal():
    bx = np.array([0.0, 1.0])
    mu = np.array([1.0, 1.0])
    Cov = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    with pytest.raises(AssertionError) as error:
        value = utils.pdf_two_dim_normal(np.array([1.0, 1.0, 1.0]), mu, Cov)
    with pytest.raises(AssertionError) as error:
        value = utils.pdf_two_dim_normal(np.array([2.0]), mu, Cov)
    with pytest.raises(AssertionError) as error:
        value = utils.pdf_two_dim_normal(bx, np.array([1.0, 1.0, 1.0]), Cov)
    with pytest.raises(AssertionError) as error:
        value = utils.pdf_two_dim_normal(bx, np.array([3.0]), Cov)

    value = utils.pdf_two_dim_normal(bx, mu, Cov)
    print(value)

    assert np.abs(0.06584073599896273 - value) < TEST_EPSILON
