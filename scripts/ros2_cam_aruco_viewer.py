#!/usr/bin/env python3
"""
ROS2 多相机图像可视化节点（含 ArUco 检测叠加显示）
订阅多个 /helmet/cam{id}/image_raw 话题，在每帧上叠加 ArUco 检测结果，
并在一个窗口中网格化显示。

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

# ── 把项目根目录加入路径，以便读取配置文件 ──
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
        config_path = _HERE / "config" / "config.yaml"
        if not config_path.exists():
            config_path = _HERE / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 定义要订阅的相机 ID
        cam_cfg = config.get('camera', {})
        capture_ids = [int(x) for x in cam_cfg.get('ids', [0, 2, 4, 6, 8, 10, 12, 14])]
        cam_to_video = _resolve_cam_to_video(capture_ids, cam_cfg.get('index_map', {}))
        self.topic_camera_ids = sorted(cam_to_video.keys())
        self.get_logger().info(
            f'Loaded cameras (display order by cam*): {self.topic_camera_ids}'
        )

        self.frames = {cam_id: None for cam_id in self.topic_camera_ids}
        self.lock = threading.Lock()

        # ── ArUco 检测器（与标定节点参数保持一致）──
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
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
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)

        # 相机内参（估算值，仅用于绘制坐标轴）
        self.camera_matrix = np.array([[500.0, 0, 320.0],
                                       [0, 500.0, 240.0],
                                       [0, 0, 1.0]], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self.marker_size = 0.1
        h = self.marker_size / 2
        self.obj_points = np.array([
            [-h,  h, 0], [ h,  h, 0],
            [ h, -h, 0], [-h, -h, 0]
        ], dtype=np.float32)

        # CLAHE，改善低对比度图像的检出率
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1).copy()
                annotated = self._detect_and_draw(frame, cam_id)
                with self.lock:
                    self.frames[cam_id] = annotated
            except Exception as e:
                self.get_logger().error(f'Cam{cam_id} error: {e}')
        return callback

    def _detect_and_draw(self, frame: np.ndarray, cam_id: int) -> np.ndarray:
        """在图像上叠加 ArUco 检测框、ID 和坐标轴"""
        gray = self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        # 相机 ID 标签（始终显示）
        cv2.putText(frame, f'Cam{cam_id}', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if ids is None:
            cv2.putText(frame, 'No Marker', (8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 1)
            return frame

        # 绘制所有检测到的标记边框
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i].reshape(-1, 2)
            cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())

            # 标记 ID 文字
            cv2.putText(frame, f'ID:{marker_id}', (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 60, 0), 2)

            # 姿态估计 + 坐标轴（仅在内参可用时绘制）
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_points,
                    corners[i].reshape(-1, 1, 2).astype(np.float32),
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if ok:
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                      rvec, tvec, self.marker_size * 0.5)
                    # 距离标注
                    dist = float(np.linalg.norm(tvec))
                    cv2.putText(frame, f'{dist:.2f}m', (cx - 20, cy + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
            except Exception:
                pass

        return frame

    def display_loop(self):
        ncols = 4
        nrows = (len(self.topic_camera_ids) + ncols - 1) // ncols
        win_w, win_h = 1280, 240 * nrows * 2  # 每格 320x240，拼接后缩放
        cv2.namedWindow("Helmet ArUco Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Helmet ArUco Viewer", win_w, win_h)

        while not self.stop_event.is_set():
            with self.lock:
                frames_snap = {cid: f.copy() if f is not None else None
                               for cid, f in self.frames.items()}

            cells = []
            for cam_id in self.topic_camera_ids:
                frame = frames_snap[cam_id]
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f'Cam{cam_id}', (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, 'No Signal', (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 220), 2)
                cells.append(cv2.resize(frame, (320, 240)))

            # 填充到 ncols 的倍数
            while len(cells) % ncols:
                cells.append(np.zeros((240, 320, 3), dtype=np.uint8))

            rows = [np.hstack(cells[r*ncols:(r+1)*ncols]) for r in range(nrows)]
            grid = np.vstack(rows)

            cv2.imshow("Helmet ArUco Viewer", grid)
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
