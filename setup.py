from distutils.core import setup

setup(
    name='FlyPlotLib',
    version='0.0.1',
    author='Floris van Breugel',
    author_email='floris@caltech.edu',
    packages = ['fly_plot_lib'],
    license='BSD',
    description='pretty matplotlib plots, specifically for fruit fly trajectories',
    long_description=open('README.txt').read(),
)



