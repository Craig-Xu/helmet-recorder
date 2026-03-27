#!/usr/bin/env python3

import os
import threading
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from helmet_recorder_ros.common import load_yaml, resolve_cam_to_video, resolve_config_path


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
    msg.header.frame_id = frame_id
    msg.height = height
    msg.width = width
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step = width * 3
    msg.data = bytearray(height * width * 3)
    return msg


def _camera_loop(camera_id: int, topic_cam_id: int, width: int, height: int,
                 fps: int, pub, stop_event: threading.Event):
    cap = _open_camera(camera_id, width, height, fps)
    if cap is None:
        rospy.logerr(f'failed to open /dev/video{camera_id}')
        return

    msg = _make_image_msg(width, height, f'cam{topic_cam_id}')
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)

    frames_published = 0
    t_report = time.monotonic()

    while not stop_event.is_set() and not rospy.is_shutdown():
        if pub.get_num_connections() == 0:
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
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)

        frames_published += 1
        now = time.monotonic()
        if now - t_report >= 5.0:
            rospy.loginfo(f'/dev/video{camera_id}: {frames_published / (now - t_report):.1f} fps')
            frames_published = 0
            t_report = now

    cap.release()


def main():
    rospy.init_node('helmet_camera_publisher')

    config_path = resolve_config_path(rospy.get_param('~config_path', ''))
    config = load_yaml(config_path)
    cam_cfg = config.get('camera', {})

    camera_ids = [int(x) for x in rospy.get_param('~camera_ids', cam_cfg.get('ids', [0]))]
    width = int(rospy.get_param('~width', cam_cfg.get('width', 640)))
    height = int(rospy.get_param('~height', cam_cfg.get('height', 480)))
    fps = int(rospy.get_param('~fps', 30))
    cam_to_video = resolve_cam_to_video(camera_ids, cam_cfg.get('index_map', {}))

    rospy.loginfo(f'config={config_path} cam_to_video={cam_to_video} {width}x{height}@{fps}fps')

    stop_event = threading.Event()
    threads = []

    for topic_cam_id in sorted(cam_to_video.keys()):
        camera_id = int(cam_to_video[topic_cam_id])
        topic = f'/helmet/cam{topic_cam_id}/image_raw'
        pub = rospy.Publisher(topic, Image, queue_size=1)
        rospy.loginfo(f'Publishing /dev/video{camera_id} -> {topic}')

        thread = threading.Thread(
            target=_camera_loop,
            args=(camera_id, topic_cam_id, width, height, fps, pub, stop_event),
            daemon=True,
            name=f'cam{camera_id}',
        )
        threads.append(thread)
        thread.start()

    def _shutdown():
        stop_event.set()
        for t in threads:
            t.join(timeout=3.0)

    rospy.on_shutdown(_shutdown)
    rospy.spin()


if __name__ == '__main__':
    main()
