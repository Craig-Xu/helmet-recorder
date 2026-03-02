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
import cv2
import numpy as np
import threading
import yaml
from pathlib import Path

# ── 把项目根目录加入路径，以便读取 config.yaml ──
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))


def _resolve_cam_to_video(camera_ids: list[int], raw_index_map: dict) -> dict[int, int]:
    """统一 index_map 为 cam_id -> /dev/video_id，兼容新旧两种配置方向。"""
    ids = [int(x) for x in camera_ids]
    id_set = set(ids)
    raw = {int(k): int(v) for k, v in (raw_index_map or {}).items()}
    if not raw:
        return {cid: cid for cid in ids}

    keys = set(raw.keys())
    vals = set(raw.values())

    if vals.issubset(id_set):
        return {cam: vid for cam, vid in raw.items() if vid in id_set}
    if keys.issubset(id_set):
        return {cam: vid for vid, cam in raw.items() if vid in id_set}
    return {cam: vid for cam, vid in raw.items()}

class MultiCameraVisualizer(Node):
    def __init__(self):
        super().__init__('multi_camera_visualizer')
        
        # 加载配置
        config_path = _HERE / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 定义要订阅的相机 ID
        cam_cfg = config.get('camera', {})
        capture_ids = [int(x) for x in cam_cfg.get('ids', [0, 2, 4, 6, 8, 10, 12, 14])]
        cam_to_video = _resolve_cam_to_video(capture_ids, cam_cfg.get('index_map', {}))
        self.topic_camera_ids = sorted(cam_to_video.keys())
        self.topic_to_capture = {int(cam): int(vid) for cam, vid in cam_to_video.items()}
        self.get_logger().info(
            f'Loaded cameras (display order by cam*): {self.topic_camera_ids}'
        )
        
        self.frames = {cam_id: None for cam_id in self.topic_camera_ids}
        self.lock = threading.Lock()

        # 使用与发布端匹配的 QoS (BEST_EFFORT)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 订阅所有话题
        self.subs = []
        for cam_id in self.topic_camera_ids:
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

    def _draw_cam_label(self, frame: np.ndarray, topic_cam_id: int) -> np.ndarray:
        capture_id = self.topic_to_capture.get(topic_cam_id, topic_cam_id)
        label = f"Cam{topic_cam_id} (/dev/video{capture_id})"
        cv2.putText(frame, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

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
                    cam_id = self.topic_camera_ids[i]
                    frame = self.frames[cam_id]
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, f"Cam{cam_id} No Signal", (50, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    frame = self._draw_cam_label(frame, cam_id)
                    row1.append(frame)
                
                # 第二行: 8, 10, 12, 14
                for i in range(4, 8):
                    cam_id = self.topic_camera_ids[i]
                    frame = self.frames[cam_id]
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, f"Cam{cam_id} No Signal", (50, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    frame = self._draw_cam_label(frame, cam_id)
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
