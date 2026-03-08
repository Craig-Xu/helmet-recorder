from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value='',
        description='Path to config file. Empty means auto resolve to config/config.yaml from cwd.',
    )

    high_res_arg = DeclareLaunchArgument(
        'high_res',
        default_value='true',
        description='Use high resolution tiles (640x480, default). Set to false for low res (160x120) to prioritize FPS.',
        choices=['true', 'false'],
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
        parameters=[
            {'config_path': LaunchConfiguration('config_path')},
            {'high_res': PythonExpression([
                "'", LaunchConfiguration('high_res'), "' == 'true'"
            ])}
        ],
    )

    return LaunchDescription([
        config_path_arg,
        high_res_arg,
        camera_publisher_node,
        multi_cam_vis_node,
    ])
