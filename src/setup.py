from setuptools import setup
with open('../requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ieeg_soz_predictor',
    version='',
    packages=[''],
    package_dir={'': 'src'},
    install_requires=requirements,
    url='http://gitlab.liaa.dc.uba.ar/tpastore/ieeg_soz_predictor.git',
    license='',
    author='Tomás Ariel Pastore',
    author_email='tpastore@dc.uba.ar',
    description='Tool to analyse a large HFO database',
    long_description=open('README.md').read()
)
