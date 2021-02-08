from setuptools import setup

path_requirements = 'requirements.txt'
list_packages = ['bayeso_benchmarks']

with open(path_requirements) as f:
    required = f.read().splitlines()

setup(
    name='bayeso-benchmarks',
    version='0.1.4',
    author='Jungtaek Kim',
    author_email='jtkim@postech.ac.kr',
    url='https://github.com/jungtaekkim/bayeso-benchmarks',
    license='MIT',
    description='Benchmarks for Bayesian optimization',
    packages=list_packages,
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, != 3.3.*, !=3.4.*, <4',
    install_requires=required,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
