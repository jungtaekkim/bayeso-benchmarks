#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: January 27, 2023
#


__version__ = '0.2.0'


from bayeso_benchmarks.inf_dim_ackley import Ackley
from bayeso_benchmarks.inf_dim_cosines import Cosines
from bayeso_benchmarks.inf_dim_griewank import Griewank
from bayeso_benchmarks.inf_dim_levy import Levy
from bayeso_benchmarks.inf_dim_rastrigin import Rastrigin
from bayeso_benchmarks.inf_dim_rosenbrock import Rosenbrock
from bayeso_benchmarks.inf_dim_sphere import Sphere
from bayeso_benchmarks.inf_dim_zakharov import Zakharov

from bayeso_benchmarks.one_dim_constant import Constant
from bayeso_benchmarks.one_dim_gramacyandlee2012 import GramacyAndLee2012
from bayeso_benchmarks.one_dim_linear import Linear
from bayeso_benchmarks.one_dim_step import Step

from bayeso_benchmarks.two_dim_beale import Beale
from bayeso_benchmarks.two_dim_bohachevsky import Bohachevsky
from bayeso_benchmarks.two_dim_branin import Branin
from bayeso_benchmarks.two_dim_bukin6 import Bukin6
from bayeso_benchmarks.two_dim_dejong5 import DeJong5
from bayeso_benchmarks.two_dim_dropwave import DropWave
from bayeso_benchmarks.two_dim_easom import Easom
from bayeso_benchmarks.two_dim_eggholder import Eggholder
from bayeso_benchmarks.two_dim_goldsteinprice import GoldsteinPrice
from bayeso_benchmarks.two_dim_holdertable import HolderTable
from bayeso_benchmarks.two_dim_kim1 import Kim1
from bayeso_benchmarks.two_dim_kim2 import Kim2
from bayeso_benchmarks.two_dim_kim3 import Kim3
from bayeso_benchmarks.two_dim_michalewicz import Michalewicz
from bayeso_benchmarks.two_dim_shubert import Shubert
from bayeso_benchmarks.two_dim_sixhumpcamel import SixHumpCamel
from bayeso_benchmarks.two_dim_threehumpcamel import ThreeHumpCamel

from bayeso_benchmarks.four_dim_colville import Colville
from bayeso_benchmarks.three_dim_hartmann3d import Hartmann3D
from bayeso_benchmarks.six_dim_hartmann6d import Hartmann6D


all_benchmarks = [
    Ackley,
    Cosines,
    Griewank,
    Levy,
    Rastrigin,
    Rosenbrock,
    Sphere,
    Zakharov,
    Constant,
    GramacyAndLee2012,
    Linear,
    Step,
    Beale,
    Bohachevsky,
    Branin,
    Bukin6,
    DeJong5,
    DropWave,
    Easom,
    Eggholder,
    GoldsteinPrice,
    HolderTable,
    Kim1,
    Kim2,
    Kim3,
    Michalewicz,
    Shubert,
    SixHumpCamel,
    ThreeHumpCamel,
    Colville,
    Hartmann3D,
    Hartmann6D,
]
num_benchmarks = len(all_benchmarks)
