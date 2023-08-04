#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: August 4, 2023
#

import numpy as np
import os
import matplotlib.pyplot as plt


def plot_1d(obj_fun,
    str_fun,
    str_x_axis=r'$x$',
    str_y_axis=r'$f(x)$',
    str_figures='../figures',
    show_figure=False,
    bounds=None, # for zooming in on part of a figure
):
    print(str_fun)

    if bounds is None:
        bounds = obj_fun.get_bounds()
    print(bounds)
    assert bounds.shape[0] == 1

    X = np.linspace(bounds[0, 0], bounds[0, 1], 1000)
    Y = obj_fun.output(X[..., np.newaxis]).flatten()

    assert len(X.shape) == 1
    assert len(Y.shape) == 1
    assert X.shape[0] == Y.shape[0]

    plt.rc('text', usetex=True)

    _ = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    ax.plot(X, Y,
        linewidth=4,
        marker='None')

    ax.set_xlabel(str_x_axis, fontsize=36)
    ax.set_ylabel(str_y_axis, fontsize=36)
    ax.tick_params(labelsize=24)

    ax.set_xlim([np.min(X), np.max(X)])
    ax.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(str_figures, str_fun + '.pdf'),
        format='pdf',
        transparent=True,
        bbox_inches='tight')

    if show_figure:
        plt.show()

    plt.close('all')

def plot_2d(obj_fun,
    str_fun,
    str_x1_axis=r'$x_1$',
    str_x2_axis=r'$x_2$',
    str_y_axis=r'$f(\mathbf{x})$',
    str_figures='../figures',
    show_figure=False,
    bounds=None, # for zooming in on part of a figure
):
    print(str_fun)

    if bounds is None:
        bounds = obj_fun.get_bounds()
    print(bounds)
    assert bounds.shape[0] == 2

    num_grids = 300

    X1 = np.linspace(bounds[0, 0], bounds[0, 1], num_grids)
    X2 = np.linspace(bounds[1, 0], bounds[1, 1], num_grids)

    if obj_fun.name == 'easom':
        num_grids_additional = 300

        X1 = np.concatenate([
                X1,
                np.linspace(np.pi - 3.0, np.pi + 3.0, num_grids_additional)
            ], axis=0)
        X2 = np.concatenate([
                X2,
                np.linspace(np.pi - 3.0, np.pi + 3.0, num_grids_additional)
            ], axis=0)

        X1 = np.sort(X1)
        X2 = np.sort(X2)

    X1, X2 = np.meshgrid(X1, X2)
    X = np.concatenate((X1[..., np.newaxis], X2[..., np.newaxis]), axis=2)
    X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))

    Y = obj_fun.output(X).flatten()

    assert len(X.shape) == 2
    assert len(Y.shape) == 1
    assert X.shape[0] == Y.shape[0]

    Y = np.reshape(Y, (X1.shape[0], X2.shape[0]))

    plt.rc('text', usetex=True)

    _ = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(X1, X2, Y,
        cmap='coolwarm',
        linewidth=0)

    ax.set_xlabel(str_x1_axis, fontsize=24, labelpad=10)
    ax.set_ylabel(str_x2_axis, fontsize=24, labelpad=10)
    ax.set_zlabel(str_y_axis, fontsize=24, labelpad=10)
    ax.tick_params(labelsize=16)

    ax.set_xlim([np.min(X1), np.max(X1)])
    ax.set_ylim([np.min(X2), np.max(X2)])
    ax.grid()

    cbar = plt.colorbar(surf,
        shrink=0.6,
        aspect=12,
        pad=0.15,
    )
    cbar.ax.tick_params(labelsize=16)

    if np.max(Y) > 1000:
        plt.ticklabel_format(axis='z', style='sci', scilimits=(0, 0), useMathText=True)
        ax.zaxis.get_offset_text().set_fontsize(14)

    plt.tight_layout()
    plt.savefig(os.path.join(str_figures, str_fun + '.pdf'),
        format='pdf',
        transparent=True,
        bbox_inches='tight')

    if show_figure:
        plt.show()

    plt.close('all')


if __name__ == '__main__':
    # one dim.
    from bayeso_benchmarks.one_dim_gramacyandlee2012 import GramacyAndLee2012 as target_class
    obj_fun = target_class()
    plot_1d(obj_fun, 'gramacyandlee2012_1d')

    from bayeso_benchmarks.inf_dim_ackley import Ackley as target_class
    obj_fun = target_class(1)
    plot_1d(obj_fun, 'ackley_1d')

    from bayeso_benchmarks.inf_dim_cosines import Cosines as target_class
    obj_fun = target_class(1)
    plot_1d(obj_fun, 'cosines_1d')

    from bayeso_benchmarks.inf_dim_griewank import Griewank as target_class
    obj_fun = target_class(1)
    plot_1d(obj_fun, 'griewank_1d')
    plot_1d(obj_fun, 'griewank_zoom_in_1d', bounds=np.array([[-50, 50]]))

    from bayeso_benchmarks.inf_dim_levy import Levy as target_class
    obj_fun = target_class(1)
    plot_1d(obj_fun, 'levy_1d')

    from bayeso_benchmarks.inf_dim_rastrigin import Rastrigin as target_class
    obj_fun = target_class(1)
    plot_1d(obj_fun, 'rastrigin_1d')

    from bayeso_benchmarks.inf_dim_sphere import Sphere as target_class
    obj_fun = target_class(1)
    plot_1d(obj_fun, 'sphere_1d')

    from bayeso_benchmarks.inf_dim_zakharov import Zakharov as target_class
    obj_fun = target_class(1)
    plot_1d(obj_fun, 'zakharov_1d')

    # two dim.
    from bayeso_benchmarks.two_dim_beale import Beale as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'beale_2d')

    from bayeso_benchmarks.two_dim_bohachevsky import Bohachevsky as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'bohachevsky_2d')

    from bayeso_benchmarks.two_dim_branin import Branin as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'branin_2d')

    from bayeso_benchmarks.two_dim_bukin6 import Bukin6 as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'bukin6_2d')

    from bayeso_benchmarks.two_dim_dejong5 import DeJong5 as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'dejong5_2d')

    from bayeso_benchmarks.two_dim_dropwave import DropWave as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'dropwave_2d')

    from bayeso_benchmarks.two_dim_easom import Easom as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'easom_2d')

    from bayeso_benchmarks.two_dim_eggholder import Eggholder as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'eggholder_2d')

    from bayeso_benchmarks.two_dim_goldsteinprice import GoldsteinPrice as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'goldsteinprice_2d')

    from bayeso_benchmarks.two_dim_holdertable import HolderTable as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'holdertable_2d')

    from bayeso_benchmarks.two_dim_kim1 import Kim1 as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'kim1_2d')

    from bayeso_benchmarks.two_dim_kim2 import Kim2 as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'kim2_2d')

    from bayeso_benchmarks.two_dim_kim3 import Kim3 as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'kim3_2d')

    from bayeso_benchmarks.two_dim_michalewicz import Michalewicz as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'michalewicz_2d')

    from bayeso_benchmarks.two_dim_shubert import Shubert as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'shubert_2d')

    from bayeso_benchmarks.two_dim_sixhumpcamel import SixHumpCamel as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'sixhumpcamel_2d')

    from bayeso_benchmarks.two_dim_threehumpcamel import ThreeHumpCamel as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'threehumpcamel_2d')

    from bayeso_benchmarks.inf_dim_ackley import Ackley as target_class
    obj_fun = target_class(2)
    plot_2d(obj_fun, 'ackley_2d')

    from bayeso_benchmarks.inf_dim_cosines import Cosines as target_class
    obj_fun = target_class(2)
    plot_2d(obj_fun, 'cosines_2d')

    from bayeso_benchmarks.inf_dim_griewank import Griewank as target_class
    obj_fun = target_class(2)
    plot_2d(obj_fun, 'griewank_2d')
    plot_2d(obj_fun, 'griewank_zoom_in_2d', bounds=np.array([[-50, 50], [-50, 50]]))

    from bayeso_benchmarks.inf_dim_levy import Levy as target_class
    obj_fun = target_class(2)
    plot_2d(obj_fun, 'levy_2d')

    from bayeso_benchmarks.inf_dim_rastrigin import Rastrigin as target_class
    obj_fun = target_class(2)
    plot_2d(obj_fun, 'rastrigin_2d')

    from bayeso_benchmarks.inf_dim_rosenbrock import Rosenbrock as target_class
    obj_fun = target_class(2)
    plot_2d(obj_fun, 'rosenbrock_2d')

    from bayeso_benchmarks.inf_dim_sphere import Sphere as target_class
    obj_fun = target_class(2)
    plot_2d(obj_fun, 'sphere_2d')

    from bayeso_benchmarks.inf_dim_zakharov import Zakharov as target_class
    obj_fun = target_class(2)
    plot_2d(obj_fun, 'zakharov_2d')
