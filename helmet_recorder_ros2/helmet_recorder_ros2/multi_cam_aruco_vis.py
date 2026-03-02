#!/usr/bin/env python3

import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

from .common import load_yaml, resolve_cam_to_video, resolve_config_path


class MultiCameraArucoVisualizer(Node):
    def __init__(self):
        super().__init__('multi_camera_aruco_visualizer')

        self.declare_parameter('config_path', '')
        config_path = resolve_config_path(str(self.get_parameter('config_path').value or ''))
        config = load_yaml(config_path)

        cam_cfg = config.get('camera', {})
        capture_ids = [int(x) for x in cam_cfg.get('ids', [0, 2, 4, 6, 8, 10, 12, 14])]
        cam_to_video = resolve_cam_to_video(capture_ids, cam_cfg.get('index_map', {}))
        self.topic_camera_ids = sorted(cam_to_video.keys())

        self.frames = {cam_id: None for cam_id in self.topic_camera_ids}
        self.lock = threading.Lock()

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)

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
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1).copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            cv2.putText(frame, f'Cam{cam_id}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            with self.lock:
                self.frames[cam_id] = frame

        return callback

    def display_loop(self):
        ncols = 4
        cv2.namedWindow('Helmet ArUco Viewer', cv2.WINDOW_NORMAL)

        while not self.stop_event.is_set():
            with self.lock:
                cells = []
                for cam_id in self.topic_camera_ids:
                    frame = self.frames[cam_id]
                    if frame is None:
                        frame = np.zeros((240, 320, 3), dtype=np.uint8)
                        cv2.putText(frame, f'Cam{cam_id} No Signal', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
                    else:
                        frame = cv2.resize(frame, (320, 240))
                    cells.append(frame)

            while len(cells) % ncols:
                cells.append(np.zeros((240, 320, 3), dtype=np.uint8))
            rows = [np.hstack(cells[i:i + ncols]) for i in range(0, len(cells), ncols)]
            grid = np.vstack(rows)
            cv2.imshow('Helmet ArUco Viewer', grid)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraArucoVisualizer()
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
