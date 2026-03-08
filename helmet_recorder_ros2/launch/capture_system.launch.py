"""
统一Launch文件：灵活启动相机、IMU和可视化
可选择是否启用IMU，并根据配置自动选择合适的viewer
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # ========== 参数定义 ==========
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value='',
        description='Path to config file. Empty means auto resolve to config/config.yaml from cwd.',
    )

    enable_imu_arg = DeclareLaunchArgument(
        'enable_imu',
        default_value='true',
        description='Whether to enable IMU publisher. Set to false for camera-only mode.',
        choices=['true', 'false'],
    )

    enable_viewer_arg = DeclareLaunchArgument(
        'enable_viewer',
        default_value='true',
        description='Whether to launch viewer. Set to false for headless recording.',
        choices=['true', 'false'],
    )

    high_res_arg = DeclareLaunchArgument(
        'high_res',
        default_value='true',
        description='Use high resolution tiles (640x480, default). Set to false for low res (160x120) to prioritize FPS.',
        choices=['true', 'false'],
    )

    # ========== 传感器节点 ==========
    # 相机发布节点（一直启动）
    camera_publisher_node = Node(
        package='helmet_recorder_ros2',
        executable='camera_publisher',
        name='helmet_camera_publisher',
        output='screen',
        parameters=[{'config_path': LaunchConfiguration('config_path')}],
    )

    # IMU发布节点（条件启动）
    imu_publisher_node = Node(
        package='helmet_recorder_ros2',
        executable='imu_publisher',
        name='helmet_imu_publisher',
        output='screen',
        parameters=[{'config_path': LaunchConfiguration('config_path')}],
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('enable_imu'), "' == 'true'"
            ])
        ),
    )

    # ========== 可视化节点 ==========
    # 相机+IMU联合可视化节点（当 enable_viewer=true 且 enable_imu=true 时）
    camera_imu_vis_node = Node(
        package='helmet_recorder_ros2',
        executable='camera_imu_vis',
        name='helmet_camera_imu_visualizer',
        output='screen',
        parameters=[
            {'config_path': LaunchConfiguration('config_path')},
            {'high_res': PythonExpression([
                "'", LaunchConfiguration('high_res'), "' == 'true'"
            ])}
        ],
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('enable_viewer'), "' == 'true' and '",
                LaunchConfiguration('enable_imu'), "' == 'true'"
            ])
        ),
    )

    # 纯相机可视化节点（当 enable_viewer=true 且 enable_imu=false 时）
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
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('enable_viewer'), "' == 'true' and '",
                LaunchConfiguration('enable_imu'), "' == 'false'"
            ])
        ),
    )

    # ========== 退出事件处理 ==========
    # 当相机+IMU viewer退出时，关闭所有节点
    camera_imu_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=camera_imu_vis_node,
            on_exit=[EmitEvent(event=Shutdown())],
        ),
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('enable_viewer'), "' == 'true' and '",
                LaunchConfiguration('enable_imu'), "' == 'true'"
            ])
        ),
    )

    # 当纯相机viewer退出时，关闭所有节点
    multi_cam_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=multi_cam_vis_node,
            on_exit=[EmitEvent(event=Shutdown())],
        ),
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('enable_viewer'), "' == 'true' and '",
                LaunchConfiguration('enable_imu'), "' == 'false'"
            ])
        ),
    )

    return LaunchDescription([
        # 参数
        config_path_arg,
        enable_imu_arg,
        enable_viewer_arg,
        high_res_arg,
        # 传感器节点
        camera_publisher_node,
        imu_publisher_node,
        # 可视化节点
        camera_imu_vis_node,
        multi_cam_vis_node,
        # 退出处理
        camera_imu_exit_handler,
        multi_cam_exit_handler,
    ])
