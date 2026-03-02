from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value='',
        description='Path to config file. Empty means auto resolve to config/config.yaml from cwd.',
    )

    camera_publisher_node = Node(
        package='helmet_recorder_ros2',
        executable='camera_publisher',
        name='helmet_camera_publisher',
        output='screen',
        parameters=[{'config_path': LaunchConfiguration('config_path')}],
    )

    multi_cam_vis_node = Node(
        package='helmet_recorder_ros2',
        executable='multi_cam_vis',
        name='helmet_multi_camera_visualizer',
        output='screen',
        parameters=[{'config_path': LaunchConfiguration('config_path')}],
    )

    return LaunchDescription([
        config_path_arg,
        camera_publisher_node,
        multi_cam_vis_node,
    ])
