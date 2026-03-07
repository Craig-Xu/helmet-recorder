"""
Launch文件：同时启动相机发布、IMU发布和联合可视化
高同步性配置，最小化延迟
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value='',
        description='Path to config file. Empty means auto resolve to config/config.yaml from cwd.',
    )

    enable_viewer_arg = DeclareLaunchArgument(
        'enable_viewer',
        default_value='true',
        description='Whether to launch the camera+IMU viewer. Set to false for headless recording.',
        choices=['true', 'false'],
    )

    # 相机发布节点
    camera_publisher_node = Node(
        package='helmet_recorder_ros2',
        executable='camera_publisher',
        name='helmet_camera_publisher',
        output='screen',
        parameters=[{'config_path': LaunchConfiguration('config_path')}],
    )

    # IMU发布节点
    imu_publisher_node = Node(
        package='helmet_recorder_ros2',
        executable='imu_publisher',
        name='helmet_imu_publisher',
        output='screen',
        parameters=[{'config_path': LaunchConfiguration('config_path')}],
    )

    # 联合可视化节点（条件启动）
    camera_imu_vis_node = Node(
        package='helmet_recorder_ros2',
        executable='camera_imu_vis',
        name='helmet_camera_imu_visualizer',
        output='screen',
        parameters=[{'config_path': LaunchConfiguration('config_path')}],
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('enable_viewer'), "' == 'true'"
            ])
        ),
    )

    # 当 viewer 退出时，关闭所有节点（仅在启用 viewer 时生效）
    exit_event_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=camera_imu_vis_node,
            on_exit=[EmitEvent(event=Shutdown())],
        ),
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('enable_viewer'), "' == 'true'"
            ])
        ),
    )

    return LaunchDescription(
        [
            config_path_arg,
            enable_viewer_arg,
            camera_publisher_node,
            imu_publisher_node,
            camera_imu_vis_node,
            exit_event_handler,
        ]
    )
