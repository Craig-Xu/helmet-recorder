#!/usr/bin/env python3
"""
ROS2 多相机图像可视化节点
订阅多个 /helmet/cam{id}/image_raw 话题并在一个窗口中网格化显示。

用法:
    python3 scripts/ros2_multi_cam_vis.py
"""

import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import yaml
from pathlib import Path

# ── 把项目根目录加入路径，以便读取 config.yaml ──
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

class MultiCameraVisualizer(Node):
    def __init__(self):
        super().__init__('multi_camera_visualizer')
        self.bridge = CvBridge()
        
        # 加载配置
        config_path = _HERE / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 定义要订阅的相机 ID
        self.camera_ids = config.get('camera', {}).get('ids', [0, 2, 4, 6, 8, 10, 12, 14])
        self.get_logger().info(f'Loaded cameras: {self.camera_ids}')
        
        self.frames = {cam_id: None for cam_id in self.camera_ids}
        self.lock = threading.Lock()

        # 使用与发布端匹配的 QoS (BEST_EFFORT)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 订阅所有话题
        self.subs = []
        for cam_id in self.camera_ids:
            topic = f'/helmet/cam{cam_id}/image_raw'
            sub = self.create_subscription(
                Image,
                topic,
                self.make_callback(cam_id),
                qos_profile
            )
            self.subs.append(sub)
            self.get_logger().info(f'Subscribed to {topic}')

        # 启动显示循环线程
        self.stop_event = threading.Event()
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.start()

    def make_callback(self, cam_id):
        def callback(msg):
            try:
                # ── 手动转换图像消息，避开 cv_bridge 可能的 numpy 版本冲突 ──
                # 假设发布端是 bgr8
                img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                
                with self.lock:
                    self.frames[cam_id] = img_data.copy()
            except Exception as e:
                self.get_logger().error(f'Error processing image from cam{cam_id}: {e}')
        return callback

    def display_loop(self):
        cv2.namedWindow("Helmet Multi-Camera Viewer", cv2.WINDOW_NORMAL)
        
        while not self.stop_event.is_set():
            display_grid = []
            
            with self.lock:
                # 将图像分成两行四列显示 (2x4)
                row1 = []
                row2 = []
                
                # 第一行: 0, 2, 4, 6
                for i in range(4):
                    cam_id = self.camera_ids[i]
                    frame = self.frames[cam_id]
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, f"Cam{cam_id} No Signal", (50, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    row1.append(frame)
                
                # 第二行: 8, 10, 12, 14
                for i in range(4, 8):
                    cam_id = self.camera_ids[i]
                    frame = self.frames[cam_id]
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, f"Cam{cam_id} No Signal", (50, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    row2.append(frame)

            # 拼接网格
            top_row = np.hstack(row1)
            bottom_row = np.hstack(row2)
            combined = np.vstack([top_row, bottom_row])

            # 缩放以便适应屏幕 (可选)
            small_combined = cv2.resize(combined, (1280, 480))
            
            cv2.imshow("Helmet Multi-Camera Viewer", small_combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_event.set()
        node.display_thread.join()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
