#!/usr/bin/env python3

import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

from .common import load_yaml, resolve_cam_to_video, resolve_config_path


class MultiCameraVisualizer(Node):
    def __init__(self):
        super().__init__('multi_camera_visualizer')

        self.declare_parameter('config_path', '')
        config_path = resolve_config_path(str(self.get_parameter('config_path').value or ''))
        config = load_yaml(config_path)

        cam_cfg = config.get('camera', {})
        capture_ids = [int(x) for x in cam_cfg.get('ids', [0, 2, 4, 6, 8, 10, 12, 14])]
        cam_to_video = resolve_cam_to_video(capture_ids, cam_cfg.get('index_map', {}))
        self.topic_camera_ids = sorted(cam_to_video.keys())
        self.topic_to_capture = {int(cam): int(vid) for cam, vid in cam_to_video.items()}

        self.frames = {cam_id: None for cam_id in self.topic_camera_ids}
        self.lock = threading.Lock()

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.subs = []
        for cam_id in self.topic_camera_ids:
            topic = f'/helmet/cam{cam_id}/image_raw'
            sub = self.create_subscription(Image, topic, self.make_callback(cam_id), qos_profile)
            self.subs.append(sub)

        self.stop_event = threading.Event()
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()

    def make_callback(self, cam_id: int):
        def callback(msg: Image):
            try:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                with self.lock:
                    self.frames[cam_id] = img.copy()
            except Exception:
                pass

        return callback

    def draw_label(self, frame: np.ndarray, topic_cam_id: int) -> np.ndarray:
        capture_id = self.topic_to_capture.get(topic_cam_id, topic_cam_id)
        label = f'Cam{topic_cam_id} (/dev/video{capture_id})'
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def display_loop(self):
        cv2.namedWindow('Helmet Multi-Camera Viewer', cv2.WINDOW_NORMAL)

        while not self.stop_event.is_set():
            with self.lock:
                ids = list(self.topic_camera_ids)
                rows = []
                for i in range(0, len(ids), 4):
                    row = []
                    for cam_id in ids[i:i + 4]:
                        frame = self.frames[cam_id]
                        if frame is None:
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(frame, f'Cam{cam_id} No Signal', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        row.append(self.draw_label(frame, cam_id))
                    while len(row) < 4:
                        row.append(np.zeros((480, 640, 3), dtype=np.uint8))
                    rows.append(np.hstack(row))

            if not rows:
                continue
            combined = np.vstack(rows)
            small = cv2.resize(combined, (1280, 480 * len(rows) // 2))
            cv2.imshow('Helmet Multi-Camera Viewer', small)

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
        node.display_thread.join(timeout=2.0)
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
