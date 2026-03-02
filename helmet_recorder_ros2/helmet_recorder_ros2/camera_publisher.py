#!/usr/bin/env python3

import os
import threading
import time

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as RosTime
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from .common import load_yaml, resolve_cam_to_video, resolve_config_path

_QOS_REALTIME = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def _open_camera(camera_id: int, width: int, height: int, fps: int):
    backend = cv2.CAP_V4L2 if os.name == 'posix' and hasattr(cv2, 'CAP_V4L2') else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    for _ in range(4):
        cap.grab()
    return cap


def _make_image_msg(width: int, height: int, frame_id: str) -> Image:
    msg = Image()
    msg.header = Header()
    msg.header.stamp = RosTime()
    msg.header.frame_id = frame_id
    msg.height = height
    msg.width = width
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step = width * 3
    msg.data = bytearray(height * width * 3)
    return msg


class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('helmet_camera_publisher')

        self.declare_parameter('config_path', '')
        config_path = resolve_config_path(str(self.get_parameter('config_path').value or ''))
        config = load_yaml(config_path)
        cam_cfg = config.get('camera', {})

        self.declare_parameter('camera_ids', cam_cfg.get('ids', [0]))
        self.declare_parameter('width', cam_cfg.get('width', 640))
        self.declare_parameter('height', cam_cfg.get('height', 480))
        self.declare_parameter('fps', 30)

        camera_ids = [int(x) for x in self.get_parameter('camera_ids').value]
        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        fps = int(self.get_parameter('fps').value)
        cam_to_video = resolve_cam_to_video(camera_ids, cam_cfg.get('index_map', {}))

        self.get_logger().info(
            f'config={config_path} cam_to_video={cam_to_video} {width}x{height}@{fps}fps'
        )

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        for topic_cam_id in sorted(cam_to_video.keys()):
            camera_id = int(cam_to_video[topic_cam_id])
            topic = f'/helmet/cam{topic_cam_id}/image_raw'
            pub = self.create_publisher(Image, topic, qos_profile=_QOS_REALTIME)
            self.get_logger().info(f'Publishing /dev/video{camera_id} -> {topic}')

            thread = threading.Thread(
                target=self._camera_loop,
                args=(camera_id, topic_cam_id, width, height, fps, pub),
                daemon=True,
                name=f'cam{camera_id}',
            )
            self._threads.append(thread)
            thread.start()

    def _camera_loop(self, camera_id: int, topic_cam_id: int, width: int, height: int, fps: int, pub):
        cap = _open_camera(camera_id, width, height, fps)
        if cap is None:
            self.get_logger().error(f'failed to open /dev/video{camera_id}')
            return

        msg = _make_image_msg(width, height, f'cam{topic_cam_id}')
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)

        frames_published = 0
        t_report = time.monotonic()

        while not self._stop_event.is_set():
            if pub.get_subscription_count() == 0:
                cap.grab()
                time.sleep(0.01)
                continue

            if not cap.grab():
                time.sleep(0.1)
                continue

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue

            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            np.copyto(buf, frame)
            msg.header.stamp = self.get_clock().now().to_msg()
            pub.publish(msg)

            frames_published += 1
            now = time.monotonic()
            if now - t_report >= 5.0:
                self.get_logger().info(f'/dev/video{camera_id}: {frames_published / (now - t_report):.1f} fps')
                frames_published = 0
                t_report = now

        cap.release()

    def destroy_node(self):
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=3.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
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
