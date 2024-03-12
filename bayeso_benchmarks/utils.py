#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: January 4, 2023
#

import numpy as np

import bayeso_benchmarks as bb


def get_benchmark(str_fun, seed=None, **kwargs):
    count = 0

    for class_benchmark in bb.all_benchmarks:
        if str_fun == class_benchmark.__name__.lower():
            target_class = class_benchmark
            count += 1

    if count == 0:
        raise ValueError('missing str_fun.')
    elif count > 1:
        raise ValueError('duplicate class name.')

    if str_fun in [
        'ackley',
        'cosines',
        'griewank',
        'levy',
        'rastrigin',
        'rosenbrock',
        'sphere',
        'zakharov',
    ]:
        assert 'dim' in kwargs
        dim = kwargs['dim']

        benchmark = target_class(dim, seed=seed)
    elif str_fun == 'constant':
        assert 'bounds' in kwargs
        assert 'constant' in kwargs
        bounds = kwargs['bounds']
        constant = kwargs['constant']

        benchmark = target_class(bounds=bounds, constant=constant, seed=seed)
    elif str_fun == 'linear':
        assert 'bounds' in kwargs
        assert 'slope' in kwargs
        bounds = kwargs['bounds']
        slope = kwargs['slope']

        benchmark = target_class(bounds=bounds, slope=slope, seed=seed)
    elif str_fun == 'step':
        assert 'steps' in kwargs
        assert 'step_values' in kwargs
        steps = kwargs['steps']
        step_values = kwargs['step_values']

        benchmark = target_class(steps=steps, step_values=step_values, seed=seed)
    else:
        benchmark = target_class(seed=seed)

    return benchmark

def pdf_two_dim_normal(bx, mu, Cov):
    assert bx.shape[0] == mu.shape[0] == Cov.shape[0] == Cov.shape[1] == 2
    
    dim = bx.shape[0]

    term_first = ((2.0 * np.pi)**(-0.5 * dim)) * (np.linalg.det(Cov)**(-0.5))
    term_second = np.exp(-0.5 * np.dot(np.dot((bx - mu), np.linalg.inv(Cov)), bx - mu))
    
    return term_first * term_second
