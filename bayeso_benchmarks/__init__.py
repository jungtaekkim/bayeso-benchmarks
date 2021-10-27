#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 23, 2021
#


__version__ = '0.1.6'


from bayeso_benchmarks.inf_dim_ackley import Ackley
from bayeso_benchmarks.inf_dim_cosines import Cosines
from bayeso_benchmarks.inf_dim_rosenbrock import Rosenbrock
from bayeso_benchmarks.inf_dim_sphere import Sphere

from bayeso_benchmarks.one_dim_constant import Constant
from bayeso_benchmarks.one_dim_gramacyandlee2012 import GramacyAndLee2012
from bayeso_benchmarks.one_dim_linear import Linear
from bayeso_benchmarks.one_dim_step import Step

from bayeso_benchmarks.two_dim_beale import Beale
from bayeso_benchmarks.two_dim_bohachevsky import Bohachevsky
from bayeso_benchmarks.two_dim_branin import Branin
from bayeso_benchmarks.two_dim_dejong5 import DeJong5
from bayeso_benchmarks.two_dim_dropwave import DropWave
from bayeso_benchmarks.two_dim_eggholder import Eggholder
from bayeso_benchmarks.two_dim_goldsteinprice import GoldsteinPrice
from bayeso_benchmarks.two_dim_holdertable import HolderTable
from bayeso_benchmarks.two_dim_kim1 import Kim1
from bayeso_benchmarks.two_dim_kim2 import Kim2
from bayeso_benchmarks.two_dim_kim3 import Kim3
from bayeso_benchmarks.two_dim_michalewicz import Michalewicz
from bayeso_benchmarks.two_dim_sixhumpcamel import SixHumpCamel
from bayeso_benchmarks.two_dim_threehumpcamel import ThreeHumpCamel

from bayeso_benchmarks.three_dim_hartmann3d import Hartmann3D
from bayeso_benchmarks.six_dim_hartmann6d import Hartmann6D
