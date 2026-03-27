#!/usr/bin/env python3

import threading
import time
from collections import deque

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

from helmet_recorder_ros.common import load_yaml, resolve_cam_to_video, resolve_config_path


class MultiCameraVisualizer:
    def __init__(self):
        config_path = resolve_config_path(rospy.get_param('~config_path', ''))
        config = load_yaml(config_path)

        cam_cfg = config.get('camera', {})
        capture_ids = [int(x) for x in cam_cfg.get('ids', [0, 2, 4, 6, 8, 10, 12, 14])]
        cam_to_video = resolve_cam_to_video(capture_ids, cam_cfg.get('index_map', {}))
        self.topic_camera_ids = sorted(cam_to_video.keys())
        self.topic_to_capture = {int(cam): int(vid) for cam, vid in cam_to_video.items()}

        # 显示配置：根据 high_res 参数选择分辨率
        high_res = rospy.get_param('~high_res', True)
        if isinstance(high_res, str):
            high_res = high_res.lower() == 'true'

        if high_res:
            self.tile_w = 640
            self.tile_h = 480
            rospy.loginfo('高分辨率模式: 640x480')
        else:
            self.tile_w = 160
            self.tile_h = 120
            rospy.loginfo('低分辨率模式（优先帧率）: 160x120')
        self.max_cols = 4

        self.frames = {cam_id: None for cam_id in self.topic_camera_ids}
        self.lock = threading.Lock()
        self.empty_tile = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.uint8)

        # 性能监控
        self.display_fps = 30
        self.display_interval = 1.0 / self.display_fps
        self.last_display_time = 0.0
        self.fps_counter = deque(maxlen=30)
        self.fullscreen = False

        # 订阅相机话题
        for cam_id in self.topic_camera_ids:
            topic = f'/helmet/cam{cam_id}/image_raw'
            rospy.Subscriber(topic, Image, self.make_callback(cam_id), queue_size=1)

        self.stop_event = threading.Event()
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()

    def make_callback(self, cam_id: int):
        def callback(msg: Image):
            try:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                if img.shape[:2] != (self.tile_h, self.tile_w):
                    img = cv2.resize(img, (self.tile_w, self.tile_h), interpolation=cv2.INTER_NEAREST)
                else:
                    img = img.copy()
                with self.lock:
                    self.frames[cam_id] = img
            except Exception:
                pass
        return callback

    def draw_label(self, frame: np.ndarray, topic_cam_id: int) -> np.ndarray:
        label = f'Cam{topic_cam_id}'
        cv2.putText(frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return frame

    def display_loop(self):
        window_name = 'Helmet Multi-Camera Viewer'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        num_rows = (len(self.topic_camera_ids) + self.max_cols - 1) // self.max_cols
        content_width = self.tile_w * self.max_cols
        content_height = self.tile_h * num_rows
        scale_factor = min(1920 / content_width, 1080 / content_height, 2.5)
        window_width = int(content_width * scale_factor)
        window_height = int(content_height * scale_factor)
        cv2.resizeWindow(window_name, window_width, window_height)

        try:
            cv2.moveWindow(window_name, (1920 - window_width) // 2, (1080 - window_height) // 2)
        except Exception:
            pass

        while not self.stop_event.is_set():
            current_time = time.time()
            if current_time - self.last_display_time < self.display_interval:
                time.sleep(0.001)
                continue

            if self.last_display_time > 0:
                self.fps_counter.append(1.0 / (current_time - self.last_display_time))
            self.last_display_time = current_time

            with self.lock:
                snapshot = {cid: self.frames[cid] for cid in self.topic_camera_ids}

            ids = list(self.topic_camera_ids)
            rows = []
            for i in range(0, len(ids), self.max_cols):
                row = []
                for cam_id in ids[i:i + self.max_cols]:
                    frame = snapshot[cam_id]
                    if frame is None:
                        frame = self.empty_tile.copy()
                        cv2.putText(frame, f'Cam{cam_id}', (5, self.tile_h // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    else:
                        frame = self.draw_label(frame, cam_id)
                    row.append(frame)
                while len(row) < self.max_cols:
                    row.append(self.empty_tile)
                rows.append(np.hstack(row))

            if not rows:
                continue

            combined = np.vstack(rows)

            if len(self.fps_counter) > 0:
                actual_fps = sum(self.fps_counter) / len(self.fps_counter)
                cv2.setWindowTitle(window_name, f'{window_name} - {actual_fps:.1f} FPS')

            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.loginfo('User pressed q, shutting down...')
                rospy.signal_shutdown('user quit')
                break
            elif key == ord('f'):
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        cv2.destroyAllWindows()


def main():
    rospy.init_node('multi_camera_visualizer')
    node = MultiCameraVisualizer()

    def _shutdown():
        node.stop_event.set()
        node.display_thread.join(timeout=2.0)

    rospy.on_shutdown(_shutdown)
    rospy.spin()


if __name__ == '__main__':
    main()
