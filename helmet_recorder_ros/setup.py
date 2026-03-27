from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['helmet_recorder_ros', 'helmet_recorder_ros.imu'],
    package_dir={'': 'src'},
)
setup(**d)
