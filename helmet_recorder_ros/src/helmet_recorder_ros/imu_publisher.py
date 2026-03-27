#!/usr/bin/env python3
"""
IMU 数据发布节点
发布 sensor_msgs/Imu 消息，与相机时间戳同步
"""

import time

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3

from helmet_recorder_ros.imu.imu_manager import IMUManager
from helmet_recorder_ros.common import load_yaml, resolve_config_path


def main():
    rospy.init_node('helmet_imu_publisher')

    config_path = resolve_config_path(rospy.get_param('~config_path', ''))
    config = load_yaml(config_path)
    imu_cfg = config.get('imu', {})

    port = imu_cfg.get('port', '/dev/ttyACM0')
    baudrate = imu_cfg.get('bps', 460800)

    rospy.loginfo(f'IMU配置: port={port} baudrate={baudrate}')

    imu_pub = rospy.Publisher('/helmet/imu/data', Imu, queue_size=1)

    imu_manager = IMUManager({
        'port': port,
        'baudrate': baudrate,
        'debug': False
    })

    # 性能优化：缓存方法引用减少查找开销
    _publish = imu_pub.publish
    _monotonic = time.monotonic
    msg_count = [0]
    last_report_time = [_monotonic()]

    def _on_imu_data(data: dict):
        """IMU数据回调 — 最小化 Python 开销，快速发布到 ROS"""
        msg = Imu()
        msg.header.stamp = rospy.Time.now()
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
            x=data['gyro_x'] * 0.017453292519943295,
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

        _publish(msg)

        msg_count[0] += 1
        now = _monotonic()
        if now - last_report_time[0] >= 5.0:
            hz = msg_count[0] / (now - last_report_time[0])
            rospy.loginfo(f'IMU发布频率: {hz:.1f} Hz')
            msg_count[0] = 0
            last_report_time[0] = now

    imu_manager.set_data_callback(_on_imu_data)

    if imu_manager.start():
        rospy.loginfo('IMU发布节点启动成功')
    else:
        rospy.logerr('IMU连接失败')

    def _shutdown():
        rospy.loginfo('正在停止IMU发布节点...')
        imu_manager.stop()

    rospy.on_shutdown(_shutdown)
    rospy.spin()


if __name__ == '__main__':
    main()
