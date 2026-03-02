#!/usr/bin/env python3
"""
ArUco 相机标定与 TF 发布节点
订阅多个相机图像，检测 ArUco 标记并发布相机相对于标记（World/Marker）的 TF。

稳定性措施:
  - CLAHE 自适应直方图均衡，应对光照变化
  - 亚像素角点精化
  - solvePnPRefineLM 二阶精化
  - 重投影误差过滤，拒绝异常帧
  - 跳变检测，拒绝位姿突变帧
  - N 帧缓冲 + 四元数平均 + tvec 中值滤波

用法:
    python3 scripts/ros2_camera_aruco_calib.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from collections import deque
import yaml
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))


def quat_mean(quats: np.ndarray) -> np.ndarray:
    """对四元数列表求平均（eigenvalue 法，适用于小角度偏差）"""
    Q = np.array(quats)
    # 保证半球一致性
    Q[Q[:, 3] < 0] *= -1
    M = Q.T @ Q
    _, vecs = np.linalg.eigh(M)
    avg = vecs[:, -1]
    return avg / np.linalg.norm(avg)


class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_aruco_calib_node')

        # 1. 加载配置
        config_path = _HERE / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.camera_ids = config.get('camera', {}).get('ids', [0, 2, 4, 6, 8, 10, 12, 14])
        self.marker_size = 0.1  # 标记边长（米），根据实际打印尺寸修改

        # 2. ArUco 检测器 + 调优参数
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        params = cv2.aruco.DetectorParameters()
        # 提升弱光/模糊环境下的检出率
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

        # 3. 相机内参（生产环境应替换为各相机实际标定值）
        self.camera_matrix = np.array([[500.0, 0, 320.0],
                                       [0, 500.0, 240.0],
                                       [0, 0, 1.0]], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # 4. 预计算标记 3D 角点
        h = self.marker_size / 2
        self.obj_points = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0]
        ], dtype=np.float32)

        # 5. 稳定性参数
        self.BUFFER_SIZE   = 7      # 用于平均的帧缓冲数量
        self.MAX_REPROJ_ERR = 1.5   # 允许的最大重投影像素误差
        self.MAX_JUMP_T    = 0.05   # 位移跳变阈值（米）
        self.MAX_JUMP_R    = 10.0   # 旋转跳变阈值（度）

        # 每个相机独立的缓冲区和上一帧状态
        self.pose_buf  = {cid: deque(maxlen=self.BUFFER_SIZE) for cid in self.camera_ids}
        self.last_tvec = {}
        self.last_quat = {}

        # 6. CLAHE（自适应直方图均衡）对象，改善光照不均匀
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 7. TF 广播器
        self.tf_broadcaster = TransformBroadcaster(self)

        # 8. 订阅相机流
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        for cam_id in self.camera_ids:
            topic = f'/helmet/cam{cam_id}/image_raw'
            self.create_subscription(Image, topic, self.make_callback(cam_id), qos_profile)
            self.get_logger().info(f'Subscribed: {topic}')

    # ── 图像回调 ─────────────────────────────────────────────────────────────
    def make_callback(self, cam_id):
        def callback(msg):
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # CLAHE 均衡，提升对比度
            gray = self.clahe.apply(gray)

            corners, ids, _ = self.detector.detectMarkers(gray)
            if ids is None:
                return

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id != 0:
                    continue

                image_points = corners[i].reshape(-1, 1, 2).astype(np.float32)

                # 初始 PnP 估计
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_points, image_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not ok:
                    continue

                # LM 精化
                rvec, tvec = cv2.solvePnPRefineLM(
                    self.obj_points, image_points,
                    self.camera_matrix, self.dist_coeffs,
                    rvec, tvec
                )

                # ── 重投影误差过滤 ────────────────────────────────────────
                proj, _ = cv2.projectPoints(
                    self.obj_points, rvec, tvec,
                    self.camera_matrix, self.dist_coeffs
                )
                err = float(np.mean(np.linalg.norm(
                    proj.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1
                )))
                if err > self.MAX_REPROJ_ERR:
                    self.get_logger().debug(
                        f'Cam{cam_id} reproj err {err:.2f}px > {self.MAX_REPROJ_ERR}, skipped')
                    continue

                # 转为 marker→camera 逆变换
                R, _ = cv2.Rodrigues(rvec)
                R_inv = R.T
                t_inv = (-R_inv @ tvec).flatten()
                quat = Rotation.from_matrix(R_inv).as_quat()  # xyzw

                # ── 跳变检测 ─────────────────────────────────────────────
                if cam_id in self.last_tvec:
                    dt = np.linalg.norm(t_inv - self.last_tvec[cam_id])
                    dq = self.last_quat[cam_id]
                    dangle = 2 * np.degrees(np.arccos(
                        np.clip(abs(np.dot(quat, dq)), 0, 1)))
                    if dt > self.MAX_JUMP_T or dangle > self.MAX_JUMP_R:
                        self.get_logger().debug(
                            f'Cam{cam_id} jump dt={dt:.3f}m dR={dangle:.1f}°, skipped')
                        continue

                self.last_tvec[cam_id] = t_inv
                self.last_quat[cam_id] = quat

                # ── 缓冲区 + 平均 ─────────────────────────────────────────
                self.pose_buf[cam_id].append((t_inv, quat))
                buf = list(self.pose_buf[cam_id])

                tvecs = np.array([p[0] for p in buf])
                quats = np.array([p[1] for p in buf])

                t_smooth = np.median(tvecs, axis=0)       # 中值滤波位移
                q_smooth = quat_mean(quats)               # 四元数均值旋转

                self.publish_tf(t_smooth, q_smooth, cam_id)
        return callback

    # ── 发布 TF ──────────────────────────────────────────────────────────────
    def publish_tf(self, tvec: np.ndarray, quat: np.ndarray, cam_id: int):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'marker_0'
        t.child_frame_id = f'cam{cam_id}'

        t.transform.translation.x = float(tvec[0])
        t.transform.translation.y = float(tvec[1])
        t.transform.translation.z = float(tvec[2])

        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
