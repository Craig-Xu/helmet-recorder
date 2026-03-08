#!/usr/bin/env python3
"""
相机+IMU联合可视化节点
实时显示8路相机画面和IMU姿态/加速度/角速度数据
"""

import threading
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, Imu

from .common import load_yaml, resolve_cam_to_video, resolve_config_path


class CameraIMUVisualizer(Node):
    def __init__(self):
        super().__init__('camera_imu_visualizer')

        self.declare_parameter('config_path', '')
        self.declare_parameter('high_res', False)  # 高分辨率模式
        
        config_path = resolve_config_path(str(self.get_parameter('config_path').value or ''))
        config = load_yaml(config_path)

        cam_cfg = config.get('camera', {})
        capture_ids = [int(x) for x in cam_cfg.get('ids', [0, 2, 4, 6, 8, 10, 12, 14])]
        cam_to_video = resolve_cam_to_video(capture_ids, cam_cfg.get('index_map', {}))
        self.camera_ids = sorted(cam_to_video.keys())
        self.topic_to_capture = {int(cam): int(vid) for cam, vid in cam_to_video.items()}

        # 显示配置：根据 high_res 参数选择分辨率
        high_res = self.get_parameter('high_res').value
        if high_res:
            self.tile_w = 640  # 高分辨率模式
            self.tile_h = 480
            self.get_logger().info('高分辨率模式: 640x480')
        else:
            self.tile_w = 160  # 低分辨率模式（优先帧率）
            self.tile_h = 120
            self.get_logger().info('低分辨率模式（优先帧率）: 160x120')
        self.max_cols = 4
        self.fullscreen = False  # 全屏状态

        # 数据存储
        self.frames = {cam_id: None for cam_id in self.camera_ids}
        self.imu_data = None
        self.imu_history = {
            'roll': deque(maxlen=100),
            'pitch': deque(maxlen=100),
            'yaw': deque(maxlen=100),
            'acc_norm': deque(maxlen=100),
            'gyro_norm': deque(maxlen=100),
        }
        self.lock = threading.Lock()
        self.empty_tile = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.uint8)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 订阅相机
        self.cam_subs = []
        for cam_id in self.camera_ids:
            topic = f'/helmet/cam{cam_id}/image_raw'
            sub = self.create_subscription(Image, topic, self.make_cam_callback(cam_id), qos_profile)
            self.cam_subs.append(sub)

        # 订阅IMU
        self.imu_sub = self.create_subscription(
            Imu, '/helmet/imu/data', self.imu_callback, qos_profile
        )

        # 性能优化：显示帧率限制和缓冲区预分配
        self.display_fps = 30  # 提高目标帧率
        self.display_interval = 1.0 / self.display_fps
        self.last_display_time = 0.0
        self.fps_counter = deque(maxlen=30)  # 用于计算实际帧率
        
        # 预分配IMU面板缓冲区（避免每帧创建）
        # 根据分辨率调整面板宽度
        if high_res:
            self.imu_panel_width = 400  # 高分辨率下使用更宽的面板
        else:
            self.imu_panel_width = 160  # 低分辨率下紧凑面板（与tile同宽）
        self.imu_panel_buffer = None  # 将在display_loop初始化

        self.stop_event = threading.Event()
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()

    def make_cam_callback(self, cam_id: int):
        def callback(msg: Image):
            try:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                if img.shape[:2] != (self.tile_h, self.tile_w):
                    # 使用最快的插值算法以提升帧率
                    img = cv2.resize(img, (self.tile_w, self.tile_h), interpolation=cv2.INTER_NEAREST)
                else:
                    # frombuffer返回的是view，需要copy以避免数据被覆盖
                    img = img.copy()
                with self.lock:
                    self.frames[cam_id] = img
            except Exception:
                pass

        return callback

    def imu_callback(self, msg: Imu):
        """IMU数据回调"""
        try:
            # 提取欧拉角（从四元数转换）
            q = msg.orientation
            roll, pitch, yaw = self.quat_to_euler(q.w, q.x, q.y, q.z)

            # 提取加速度和角速度（转回常用单位）
            acc_x = msg.linear_acceleration.x / 9.80665
            acc_y = msg.linear_acceleration.y / 9.80665
            acc_z = msg.linear_acceleration.z / 9.80665
            acc_norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

            gyro_x = msg.angular_velocity.x * 57.29577951308232  # rad/s to deg/s
            gyro_y = msg.angular_velocity.y * 57.29577951308232
            gyro_z = msg.angular_velocity.z * 57.29577951308232
            gyro_norm = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

            with self.lock:
                self.imu_data = {
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    'acc_x': acc_x,
                    'acc_y': acc_y,
                    'acc_z': acc_z,
                    'acc_norm': acc_norm,
                    'gyro_x': gyro_x,
                    'gyro_y': gyro_y,
                    'gyro_z': gyro_z,
                    'gyro_norm': gyro_norm,
                }
                self.imu_history['roll'].append(roll)
                self.imu_history['pitch'].append(pitch)
                self.imu_history['yaw'].append(yaw)
                self.imu_history['acc_norm'].append(acc_norm)
                self.imu_history['gyro_norm'].append(gyro_norm)

        except Exception as e:
            self.get_logger().warn(f'IMU数据处理错误: {e}')

    @staticmethod
    def quat_to_euler(w, x, y, z):
        """四元数转欧拉角（roll, pitch, yaw），单位：度"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll * 57.29577951308232, pitch * 57.29577951308232, yaw * 57.29577951308232

    def draw_label(self, frame: np.ndarray, cam_id: int) -> np.ndarray:
        """绘制相机标签（简化版）"""
        # 缩小字体以匹配更小的tile
        label = f'C{cam_id}'
        cv2.putText(frame, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return frame

    def draw_imu_panel(self, panel: np.ndarray) -> np.ndarray:
        """绘制IMU数据面板（复用缓冲区）"""
        # 清空面板（深灰色背景）
        panel[:] = (30, 30, 30)
        
        height = panel.shape[0]
        width = panel.shape[1]

        with self.lock:
            imu = self.imu_data

        if imu is None:
            cv2.putText(panel, 'IMU: No Data', (15, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            return panel

        # 根据面板宽度选择字体大小和布局
        if width >= 400:
            font_scale_title = 1.3
            font_scale_label = 0.8
            font_scale_value = 1.0
            thickness_title = 3
            thickness_label = 2  # 标签加粗
            thickness_normal = 2
            spacing = 55
            compact_mode = False
        else:
            # 低分辨率模式：紧凑布局，保证所有内容可见（面板高约240px）
            font_scale_title = 0.7
            font_scale_label = 0.45
            font_scale_value = 0.55
            thickness_title = 2
            thickness_label = 1
            thickness_normal = 1
            spacing = 17  # 极紧凑间距
            compact_mode = True  # 启用紧凑模式
        
        y_pos = 13 if compact_mode else 40
        
        # 标题
        cv2.putText(panel, 'IMU', (15, y_pos), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale_title, (255, 255, 255), thickness_title, cv2.LINE_AA)
        y_pos += spacing if compact_mode else spacing - 5
        
        # === 紧凑模式布局 ===
        if compact_mode:
            # 姿态角
            cv2.putText(panel, 'Orient(deg)', (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, (180, 180, 180), thickness_label, cv2.LINE_AA)
            y_pos += spacing
            cv2.putText(panel, f"R:{imu['roll']:6.1f}", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (100, 200, 255), thickness_normal, cv2.LINE_AA)
            y_pos += spacing
            cv2.putText(panel, f"P:{imu['pitch']:6.1f}", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (100, 255, 200), thickness_normal, cv2.LINE_AA)
            y_pos += spacing
            cv2.putText(panel, f"Y:{imu['yaw']:6.1f}", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (255, 200, 100), thickness_normal, cv2.LINE_AA)
            y_pos += spacing
            
            # 加速度
            cv2.putText(panel, 'Accel(g)', (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, (180, 180, 180), thickness_label, cv2.LINE_AA)
            y_pos += spacing
            cv2.putText(panel, f"X:{imu['acc_x']:5.2f}", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (0, 255, 128), thickness_normal, cv2.LINE_AA)
            y_pos += spacing
            cv2.putText(panel, f"Y:{imu['acc_y']:5.2f}", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (0, 255, 128), thickness_normal, cv2.LINE_AA)
            y_pos += spacing
            cv2.putText(panel, f"Z:{imu['acc_z']:5.2f}", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (0, 255, 128), thickness_normal, cv2.LINE_AA)
            y_pos += spacing
            
            # 角速度
            cv2.putText(panel, 'Gyro(deg/s)', (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, (180, 180, 180), thickness_label, cv2.LINE_AA)
            y_pos += spacing
            cv2.putText(panel, f"X:{imu['gyro_x']:6.1f}", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (255, 100, 255), thickness_normal, cv2.LINE_AA)
            y_pos += spacing
            cv2.putText(panel, f"Y:{imu['gyro_y']:6.1f}", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (255, 100, 255), thickness_normal, cv2.LINE_AA)
            y_pos += spacing
            cv2.putText(panel, f"Z:{imu['gyro_z']:6.1f}", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (255, 100, 255), thickness_normal, cv2.LINE_AA)
        else:
            # 分割线
            cv2.line(panel, (15, y_pos), (width - 15, y_pos), (100, 100, 100), 1)
            y_pos += spacing - 10
            
            # === 姿态角 ===
            cv2.putText(panel, 'Orientation (deg)', (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, (180, 180, 180), thickness_label, cv2.LINE_AA)
            y_pos += spacing - 15
            
            # Roll
            cv2.putText(panel, 'Roll:', (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, (100, 200, 255), thickness_label, cv2.LINE_AA)
            cv2.putText(panel, f"{imu['roll']:7.2f}", (width - 120, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (100, 200, 255), thickness_normal, cv2.LINE_AA)
            y_pos += spacing - 10
            
            # Pitch
            cv2.putText(panel, 'Pitch:', (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, (100, 255, 200), thickness_label, cv2.LINE_AA)
            cv2.putText(panel, f"{imu['pitch']:7.2f}", (width - 120, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (100, 255, 200), thickness_normal, cv2.LINE_AA)
            y_pos += spacing - 10
            
            # Yaw
            cv2.putText(panel, 'Yaw:', (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, (255, 200, 100), thickness_label, cv2.LINE_AA)
            cv2.putText(panel, f"{imu['yaw']:7.2f}", (width - 120, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (255, 200, 100), thickness_normal, cv2.LINE_AA)
            y_pos += spacing
            
            # 分割线
            cv2.line(panel, (15, y_pos - 10), (width - 15, y_pos - 10), (100, 100, 100), 1)
            y_pos += spacing - 20
            
            # === 加速度 ===
            cv2.putText(panel, 'Acceleration (g)', (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, (180, 180, 180), thickness_label, cv2.LINE_AA)
            y_pos += spacing - 15
            
            # X, Y, Z
            cv2.putText(panel, f"X:{imu['acc_x']:6.2f}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (0, 255, 128), thickness_normal, cv2.LINE_AA)
            y_pos += spacing - 10
            cv2.putText(panel, f"Y:{imu['acc_y']:6.2f}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (0, 255, 128), thickness_normal, cv2.LINE_AA)
            y_pos += spacing - 10
            cv2.putText(panel, f"Z:{imu['acc_z']:6.2f}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (0, 255, 128), thickness_normal, cv2.LINE_AA)
            y_pos += spacing
            
            # 分割线
            cv2.line(panel, (15, y_pos - 10), (width - 15, y_pos - 10), (100, 100, 100), 1)
            y_pos += spacing - 20
            
            # === 角速度 ===
            cv2.putText(panel, 'Gyroscope (deg/s)', (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, (180, 180, 180), thickness_label, cv2.LINE_AA)
            y_pos += spacing - 15
            
            # X, Y, Z
            cv2.putText(panel, f"X:{imu['gyro_x']:7.1f}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (255, 100, 255), thickness_normal, cv2.LINE_AA)
            y_pos += spacing - 10
            cv2.putText(panel, f"Y:{imu['gyro_y']:7.1f}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (255, 100, 255), thickness_normal, cv2.LINE_AA)
            y_pos += spacing - 10
            cv2.putText(panel, f"Z:{imu['gyro_z']:7.1f}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_value, (255, 100, 255), thickness_normal, cv2.LINE_AA)

        return panel

    def display_loop(self):
        """显示循环（优化后）"""
        import time
        
        window_name = 'Camera + IMU Viewer'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # 预分配网格行数
        num_rows = (len(self.camera_ids) + self.max_cols - 1) // self.max_cols
        
        # 预分配IMU面板缓冲区
        imu_panel_height = self.tile_h * num_rows
        self.imu_panel_buffer = np.zeros((imu_panel_height, self.imu_panel_width, 3), dtype=np.uint8)
        
        # 计算合适的窗口大小（内容大小 * 缩放因子）
        content_width = self.tile_w * self.max_cols + self.imu_panel_width
        content_height = self.tile_h * num_rows
        scale_factor = min(1920 / content_width, 1080 / content_height, 2.0)  # 允许更大缩放
        window_width = int(content_width * scale_factor)
        window_height = int(content_height * scale_factor)
        cv2.resizeWindow(window_name, window_width, window_height)
        
        # 窗口居中（非关键操作，失败不影响性能）
        try:
            cv2.moveWindow(window_name, (1920 - window_width) // 2, (1080 - window_height) // 2)
        except:
            pass

        while not self.stop_event.is_set():
            # 帧率限制
            current_time = time.time()
            if current_time - self.last_display_time < self.display_interval:
                time.sleep(0.001)  # 短暂睡眠避免空转
                continue
            
            # 计算实际帧率
            if self.last_display_time > 0:
                self.fps_counter.append(1.0 / (current_time - self.last_display_time))
            self.last_display_time = current_time
            
            # 快速获取帧快照
            with self.lock:
                snapshot = {cid: self.frames[cid] for cid in self.camera_ids}

            # 构建相机网格
            ids = list(self.camera_ids)
            rows = []
            for i in range(0, len(ids), self.max_cols):
                row = []
                for cam_id in ids[i : i + self.max_cols]:
                    frame = snapshot[cam_id]
                    if frame is None:
                        frame = self.empty_tile.copy()
                        cv2.putText(
                            frame,
                            f'Cam{cam_id}',
                            (10, self.tile_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            1,
                        )
                    else:
                        # 直接使用快照，不再额外copy
                        frame = self.draw_label(frame, cam_id)
                    row.append(frame)
                # 补齐行
                while len(row) < self.max_cols:
                    row.append(self.empty_tile)
                rows.append(np.hstack(row))

            camera_grid = np.vstack(rows) if rows else self.empty_tile

            # 绘制IMU面板（复用缓冲区）
            self.draw_imu_panel(self.imu_panel_buffer)

            # 拼接显示
            combined = np.hstack([camera_grid, self.imu_panel_buffer])
            
            # 更新窗口标题，显示实际帧率
            if len(self.fps_counter) > 0:
                actual_fps = sum(self.fps_counter) / len(self.fps_counter)
                cv2.setWindowTitle(window_name, f'{window_name} - {actual_fps:.1f} FPS')
            
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('User pressed q, shutting down...')
                rclpy.shutdown()
                break
            elif key == ord('f'):
                # 切换全屏
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CameraIMUVisualizer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_event.set()
        node.display_thread.join(timeout=2.0)
        executor.shutdown()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
