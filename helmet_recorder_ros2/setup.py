from setuptools import find_packages, setup
from glob import glob

package_name = 'helmet_recorder_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kai',
    maintainer_email='kai@example.com',
    description='ROS2 drivers and tools for helmet multi-camera recorder',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = helmet_recorder_ros2.camera_publisher:main',
            'imu_publisher = helmet_recorder_ros2.imu_publisher:main',
            'camera_imu_vis = helmet_recorder_ros2.camera_imu_vis:main',
            'fps_monitor = helmet_recorder_ros2.fps_monitor:main',
            'aruco_calib = helmet_recorder_ros2.aruco_calib:main',
            'save_cam_extrinsics = helmet_recorder_ros2.save_cam_extrinsics:main',
            'image_calib_save_extrinsics = helmet_recorder_ros2.image_calib_save_extrinsics:main',
            'multi_cam_vis = helmet_recorder_ros2.multi_cam_vis:main',
            'multi_cam_aruco_vis = helmet_recorder_ros2.multi_cam_aruco_vis:main',
        ],
    },
)
