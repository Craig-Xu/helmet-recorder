#!/usr/bin/env python3

from collections import deque

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster

from .common import load_yaml, resolve_cam_to_video, resolve_config_path


def quat_mean(quats: np.ndarray) -> np.ndarray:
    q = np.array(quats)
    q[q[:, 3] < 0] *= -1
    _, vecs = np.linalg.eigh(q.T @ q)
    avg = vecs[:, -1]
    return avg / np.linalg.norm(avg)


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    return quat / np.linalg.norm(quat)


class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_aruco_calib_node')

        self.declare_parameter('config_path', '')
        config_path = resolve_config_path(str(self.get_parameter('config_path').value or ''))
        config = load_yaml(config_path)

        cam_cfg = config.get('camera', {})
        capture_ids = [int(x) for x in cam_cfg.get('ids', [0, 2, 4, 6, 8, 10, 12, 14])]
        self.cam_to_video = resolve_cam_to_video(capture_ids, cam_cfg.get('index_map', {}))
        self.camera_ids = sorted(self.cam_to_video.keys())
        self.marker_size = 0.1

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate = 0.02
        params.polygonalApproxAccuracyRate = 0.05
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.01
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)

        self.camera_matrix = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        h = self.marker_size / 2
        self.obj_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

        self.buffer_size = 7
        self.max_reproj_err = 1.5
        self.max_jump_t = 0.05
        self.max_jump_r = 10.0

        self.pose_buf = {cid: deque(maxlen=self.buffer_size) for cid in self.camera_ids}
        self.last_tvec = {}
        self.last_quat = {}
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.tf_broadcaster = TransformBroadcaster(self)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        for cam_id in self.camera_ids:
            topic = f'/helmet/cam{cam_id}/image_raw'
            self.create_subscription(Image, topic, self.make_callback(cam_id), qos_profile)
            self.get_logger().info(f'Subscribed: {topic}')

    def make_callback(self, cam_id: int):
        def callback(msg: Image):
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            gray = self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            corners, ids, _ = self.detector.detectMarkers(gray)
            if ids is None:
                return

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id != 0:
                    continue

                image_points = corners[i].reshape(-1, 1, 2).astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if not ok:
                    continue

                rvec, tvec = cv2.solvePnPRefineLM(
                    self.obj_points, image_points, self.camera_matrix, self.dist_coeffs, rvec, tvec
                )

                proj, _ = cv2.projectPoints(self.obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)))
                if err > self.max_reproj_err:
                    continue

                rot_matrix, _ = cv2.Rodrigues(rvec)
                rot_inv = rot_matrix.T
                t_inv = (-rot_inv @ tvec).flatten()
                quat = matrix_to_quat_xyzw(rot_inv)

                if cam_id in self.last_tvec:
                    dt = np.linalg.norm(t_inv - self.last_tvec[cam_id])
                    dq = self.last_quat[cam_id]
                    dangle = 2 * np.degrees(np.arccos(np.clip(abs(np.dot(quat, dq)), 0, 1)))
                    if dt > self.max_jump_t or dangle > self.max_jump_r:
                        continue

                self.last_tvec[cam_id] = t_inv
                self.last_quat[cam_id] = quat

                self.pose_buf[cam_id].append((t_inv, quat))
                samples = list(self.pose_buf[cam_id])
                tvecs = np.array([p[0] for p in samples])
                quats = np.array([p[1] for p in samples])
                t_smooth = np.median(tvecs, axis=0)
                q_smooth = quat_mean(quats)

                self.publish_tf(t_smooth, q_smooth, cam_id)

        return callback

    def publish_tf(self, tvec: np.ndarray, quat: np.ndarray, cam_id: int):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'marker_0'
        transform.child_frame_id = f'cam{cam_id}'
        transform.transform.translation.x = float(tvec[0])
        transform.transform.translation.y = float(tvec[1])
        transform.transform.translation.z = float(tvec[2])
        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])
        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
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
