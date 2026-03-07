#!/usr/bin/env python3
"""
IMU 数据发布节点 - 高性能低延迟版本
发布 sensor_msgs/Imu 消息，与相机时间戳严格同步
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3

from .imu.imu_manager import IMUManager
from .common import load_yaml, resolve_config_path

_QOS_REALTIME = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class IMUPublisherNode(Node):
    def __init__(self):
        super().__init__('helmet_imu_publisher')

        self.declare_parameter('config_path', '')
        config_path = resolve_config_path(str(self.get_parameter('config_path').value or ''))
        config = load_yaml(config_path)
        imu_cfg = config.get('imu', {})

        # IMU配置
        port = imu_cfg.get('port', '/dev/ttyACM0')
        baudrate = imu_cfg.get('bps', 460800)

        self.get_logger().info(f'IMU配置: port={port} baudrate={baudrate}')

        # 创建发布者
        self.imu_pub = self.create_publisher(Imu, '/helmet/imu/data', qos_profile=_QOS_REALTIME)

        # 创建IMU管理器
        self.imu_manager = IMUManager({
            'port': port,
            'baudrate': baudrate,
            'debug': False
        })

        # 性能优化：缓存方法引用减少查找开销
        self._publish = self.imu_pub.publish
        self._get_time = self.get_clock().now
        self._log_info = self.get_logger().info
        self._monotonic = time.monotonic

        # 统计
        self.msg_count = 0
        self.last_report_time = self._monotonic()

        # 设置回调函数
        self.imu_manager.set_data_callback(self._on_imu_data)

        # 启动IMU
        if self.imu_manager.start():
            self.get_logger().info('✅ IMU发布节点启动成功')
        else:
            self.get_logger().error('❌ IMU连接失败')

    def _on_imu_data(self, data: dict):
        """
        IMU数据回调 - 高性能版本
        关键优化：最小化Python开销，快速发布到ROS2
        """
        # 创建消息
        msg = Imu()
        msg.header.stamp = self._get_time().to_msg()
        msg.header.frame_id = 'imu_link'

        # 四元数 (q0=w, q1=x, q2=y, q3=z)
        msg.orientation = Quaternion(
            x=data['q1'],
            y=data['q2'],
            z=data['q3'],
            w=data['q0']
        )
        msg.orientation_covariance = [0.01, 0.0, 0.0,
                                       0.0, 0.01, 0.0,
                                       0.0, 0.0, 0.01]

        # 角速度 (°/s -> rad/s)
        msg.angular_velocity = Vector3(
            x=data['gyro_x'] * 0.017453292519943295,  # deg to rad
            y=data['gyro_y'] * 0.017453292519943295,
            z=data['gyro_z'] * 0.017453292519943295
        )
        msg.angular_velocity_covariance = [0.001, 0.0, 0.0,
                                            0.0, 0.001, 0.0,
                                            0.0, 0.0, 0.001]

        # 加速度 (g -> m/s²)
        msg.linear_acceleration = Vector3(
            x=data['acc_x'] * 9.80665,
            y=data['acc_y'] * 9.80665,
            z=data['acc_z'] * 9.80665
        )
        msg.linear_acceleration_covariance = [0.001, 0.0, 0.0,
                                               0.0, 0.001, 0.0,
                                               0.0, 0.0, 0.001]

        # 发布消息（零拷贝）
        self._publish(msg)

        # 统计
        self.msg_count += 1
        now = self._monotonic()
        if now - self.last_report_time >= 5.0:
            hz = self.msg_count / (now - self.last_report_time)
            self._log_info(f'IMU发布频率: {hz:.1f} Hz')
            self.msg_count = 0
            self.last_report_time = now

    def destroy_node(self):
        """清理资源"""
        self.get_logger().info('正在停止IMU发布节点...')
        self.imu_manager.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IMUPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
