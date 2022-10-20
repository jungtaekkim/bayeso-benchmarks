#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 20, 2022
#

import numpy as np

import bayeso_benchmarks as bb


def get_benchmark(str_fun, seed=None, **kwargs):
    if str_fun == 'ackley':
        assert 'dim' in kwargs
        dim = kwargs['dim']

        benchmark = bb.Ackley(dim, seed=seed)
    elif str_fun == 'cosines':
        assert 'dim' in kwargs
        dim = kwargs['dim']

        benchmark = bb.Cosines(dim, seed=seed)
    elif str_fun == 'rosenbrock':
        assert 'dim' in kwargs
        dim = kwargs['dim']

        benchmark = bb.Rosenbrock(dim, seed=seed)
    elif str_fun == 'sphere':
        assert 'dim' in kwargs
        dim = kwargs['dim']

        benchmark = bb.Sphere(dim, seed=seed)
    elif str_fun == 'constant':
        assert 'bounds' in kwargs
        assert 'constant' in kwargs
        bounds = kwargs['bounds']
        constant = kwargs['constant']

        benchmark = bb.Constant(bounds=bounds, constant=constant, seed=seed)
    elif str_fun == 'gramacyandlee2012':
        benchmark = bb.GramacyAndLee2012(seed=seed)
    elif str_fun == 'linear':
        assert 'bounds' in kwargs
        assert 'slope' in kwargs
        bounds = kwargs['bounds']
        slope = kwargs['slope']

        benchmark = bb.Linear(bounds=bounds, slope=slope, seed=seed)
    elif str_fun == 'step':
        assert 'steps' in kwargs
        assert 'step_values' in kwargs
        steps = kwargs['steps']
        step_values = kwargs['step_values']

        benchmark = bb.Step(steps=steps, step_values=step_values, seed=seed)
    elif str_fun == 'beale':
        benchmark = bb.Beale(seed=seed)
    elif str_fun == 'bohachevsky':
        benchmark = bb.Bohachevsky(seed=seed)
    elif str_fun == 'branin':
        benchmark = bb.Branin(seed=seed)
    elif str_fun == 'dejong5':
        benchmark = bb.DeJong5(seed=seed)
    elif str_fun == 'dropwave':
        benchmark = bb.DropWave(seed=seed)
    elif str_fun == 'eggholder':
        benchmark = bb.Eggholder(seed=seed)
    elif str_fun == 'goldsteinprice':
        benchmark = bb.GoldsteinPrice(seed=seed)
    elif str_fun == 'holdertable':
        benchmark = bb.HolderTable(seed=seed)
    elif str_fun == 'kim1':
        benchmark = bb.Kim1(seed=seed)
    elif str_fun == 'kim2':
        benchmark = bb.Kim2(seed=seed)
    elif str_fun == 'kim3':
        benchmark = bb.Kim3(seed=seed)
    elif str_fun == 'michalewicz':
        benchmark = bb.Michalewicz(seed=seed)
    elif str_fun == 'sixhumpcamel':
        benchmark = bb.SixHumpCamel(seed=seed)
    elif str_fun == 'threehumpcamel':
        benchmark = bb.ThreeHumpCamel(seed=seed)
    elif str_fun == 'hartmann3d':
        benchmark = bb.Hartmann3D(seed=seed)
    elif str_fun == 'hartmann6d':
        benchmark = bb.Hartmann6D(seed=seed)
    else:
        raise ValueError('missing str_fun.')

    return benchmark

def pdf_two_dim_normal(bx, mu, Cov):
    assert bx.shape[0] == mu.shape[0] == Cov.shape[0] == Cov.shape[1] == 2
    
    dim = bx.shape[0]

    term_first = ((2.0 * np.pi)**(-0.5 * dim)) * (np.linalg.det(Cov)**(-0.5))
    term_second = np.exp(-0.5 * np.dot(np.dot((bx - mu), np.linalg.inv(Cov)), bx - mu))
    
    return term_first * term_second
