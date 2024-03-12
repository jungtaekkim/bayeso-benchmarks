#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 27, 2023
#


STR_VERSION = '0.2.0'


def test_version_bayeso():
    import bayeso_benchmarks
    assert bayeso_benchmarks.__version__ == STR_VERSION

def test_version_setup():
    import pkg_resources
    assert pkg_resources.require('bayeso-benchmarks')[0].version == STR_VERSION
