from setuptools import setup
from pathlib import Path


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

path_requirements = 'requirements.txt'
list_packages = ['bayeso_benchmarks']

with open(path_requirements) as f:
    required = f.read().splitlines()

setup(
    name='bayeso-benchmarks',
    version='0.2.0',
    author='Jungtaek Kim',
    author_email='jungtaek.kim.mail@gmail.com',
    url='https://bayeso.org',
    license='MIT',
    description='Benchmark Functions for Bayesian optimization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=list_packages,
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, != 3.3.*, !=3.4.*, !=3.5.*, <4',
    install_requires=required,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
