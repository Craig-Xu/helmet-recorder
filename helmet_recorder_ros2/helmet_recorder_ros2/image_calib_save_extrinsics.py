#!/usr/bin/env python3
"""
image_calib_save_extrinsics.py
─────────────────────────────
基于 ROS2 图像话题的多相机 ArUco 标定脚本。

行为与 scripts/cam_calib_standalone.py 保持一致：
- OpenCV 交互窗口预览
- 按 y 收集样本并保存 index_map + extrinsics
- 按 q 退出

唯一差异：输入来源由 /dev/video* 改为订阅 /helmet/cam*/image_raw。
"""

import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

from .common import load_yaml, resolve_cam_to_video, resolve_config_path, save_yaml

# ── ArUco 参数 ──
ARUCO_DICT = cv2.aruco.DICT_6X6_1000
TARGET_MARKER = 0
MARKER_SIZE = 0.1

# ── 相机参数（话题输入）──
COLLECT_SECONDS = 5.0
MIN_SAMPLES = 10

# ── 稳定性参数 ──
BUFFER_SIZE = 7
MAX_REPROJ_ERR = 1.5
MAX_JUMP_T = 0.05
MAX_JUMP_R = 10.0

# ── 排序参数 ──
INDEX_AXIS = 0
INDEX_DESC = False
MIDDLE_COUNT = 4
MIDDLE_Z_ASC = True
SWAP_PAIRS = [(2, 3), (4, 5)]
REF_CAM_ID = 3


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


def quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    return np.array([-x, -y, -z, w], dtype=np.float64)


def quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


def quat_rotate_vec_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = quat / np.linalg.norm(quat)
    q_vec = np.array([vec[0], vec[1], vec[2], 0.0], dtype=np.float64)
    q_inv = quat_conjugate_xyzw(q)
    rotated = quat_multiply_xyzw(quat_multiply_xyzw(q, q_vec), q_inv)
    return rotated[:3]


def rebase_extrinsics(results: dict, ref_cam_name: str) -> dict:
    if ref_cam_name not in results:
        return results

    ref = results[ref_cam_name]
    q_ref = np.array(ref['rotation'], dtype=np.float64)
    q_ref = q_ref / np.linalg.norm(q_ref)
    t_ref = np.array(ref['translation'])
    q_ref_inv = quat_conjugate_xyzw(q_ref)

    rebased = {}
    for cam_name, item in results.items():
        q_cam = np.array(item['rotation'], dtype=np.float64)
        q_cam = q_cam / np.linalg.norm(q_cam)
        t_cam = np.array(item['translation'])

        q_rel = quat_multiply_xyzw(q_ref_inv, q_cam)
        q_rel = q_rel / np.linalg.norm(q_rel)
        t_rel = quat_rotate_vec_xyzw(q_ref_inv, t_cam - t_ref)

        rebased[cam_name] = {
            **item,
            'translation': [round(float(v), 6) for v in t_rel],
            'rotation': [round(float(v), 8) for v in q_rel],
        }
    return rebased


def build_index_map(results: dict, video_ids_per_cam: dict):
    sortable = []
    for cam_name, item in results.items():
        t = item['translation']
        vid = video_ids_per_cam.get(cam_name)
        if vid is None:
            continue
        sortable.append((cam_name, float(t[INDEX_AXIS]), float(t[2]), vid))

    sortable.sort(key=lambda x: x[1], reverse=INDEX_DESC)

    n = len(sortable)
    if n >= MIDDLE_COUNT:
        ms = (n - MIDDLE_COUNT) // 2
        me = ms + MIDDLE_COUNT
        mid = sortable[ms:me]
        mid.sort(key=lambda x: x[2], reverse=not MIDDLE_Z_ASC)
        sortable = sortable[:ms] + mid + sortable[me:]

    idx_map = {i: int(vid) for i, (_, _, _, vid) in enumerate(sortable)}

    for a, b in SWAP_PAIRS:
        if a in idx_map and b in idx_map:
            idx_map[a], idx_map[b] = idx_map[b], idx_map[a]

    cam_order = [name for name, _, _, _ in sortable]
    for a, b in SWAP_PAIRS:
        if a < len(cam_order) and b < len(cam_order):
            cam_order[a], cam_order[b] = cam_order[b], cam_order[a]

    return idx_map, cam_order


class ArucoDetector:
    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
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
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.camera_matrix = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        h = MARKER_SIZE / 2
        self.obj_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    def detect(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)

        corners, ids, _ = self.detector.detectMarkers(gray)
        vis = frame.copy()

        if ids is None:
            return None, None, vis

        for i, mid in enumerate(ids.flatten()):
            if mid != TARGET_MARKER:
                continue

            img_pts = corners[i].reshape(-1, 1, 2).astype(np.float32)

            ok, rvec, tvec = cv2.solvePnP(
                self.obj_points, img_pts, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                continue

            rvec, tvec = cv2.solvePnPRefineLM(
                self.obj_points, img_pts, self.camera_matrix, self.dist_coeffs, rvec, tvec
            )

            proj, _ = cv2.projectPoints(self.obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)))
            if err > MAX_REPROJ_ERR:
                continue

            rot_matrix, _ = cv2.Rodrigues(rvec)
            rot_inv = rot_matrix.T
            t_inv = (-rot_inv @ tvec).flatten()
            quat = matrix_to_quat_xyzw(rot_inv)

            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.drawFrameAxes(vis, self.camera_matrix, self.dist_coeffs, rvec, tvec, MARKER_SIZE * 0.5)
            return t_inv, quat, vis

        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        return None, None, vis


class TopicCameraStream:
    def __init__(self, cam_id: int, aruco: ArucoDetector):
        self.cam_id = cam_id
        self.aruco = aruco
        self.alive = True

        self.pose_buf = deque(maxlen=BUFFER_SIZE)
        self.last_tvec = None
        self.last_quat = None

        self.collecting = False
        self.samples_t = []
        self.samples_q = []

        self._frame = None
        self._tvec_smooth = None
        self._quat_smooth = None
        self._detected = False

    def on_image(self, frame: np.ndarray):
        t_inv, quat, vis = self.aruco.detect(frame)

        t_smooth, q_smooth = self._tvec_smooth, self._quat_smooth
        detected = t_inv is not None

        if t_inv is not None:
            if self.collecting:
                self.samples_t.append(t_inv.copy())
                self.samples_q.append(quat.copy())

            skip = False
            if self.last_tvec is not None:
                dt = np.linalg.norm(t_inv - self.last_tvec)
                dangle = 2 * np.degrees(np.arccos(np.clip(abs(np.dot(quat, self.last_quat)), 0, 1)))
                if dt > MAX_JUMP_T or dangle > MAX_JUMP_R:
                    skip = True

            if not skip:
                self.last_tvec = t_inv
                self.last_quat = quat
                self.pose_buf.append((t_inv, quat))

                buf = list(self.pose_buf)
                tvecs = np.array([p[0] for p in buf])
                quats = np.array([p[1] for p in buf])
                t_smooth = np.median(tvecs, axis=0)
                q_smooth = quat_mean(quats)

        self._frame = vis
        self._tvec_smooth = t_smooth
        self._quat_smooth = q_smooth
        self._detected = detected

    def get_state(self):
        return (
            self._frame.copy() if self._frame is not None else None,
            self._tvec_smooth.copy() if self._tvec_smooth is not None else None,
            self._quat_smooth.copy() if self._quat_smooth is not None else None,
            self._detected,
        )

    def start_collect(self):
        self.samples_t.clear()
        self.samples_q.clear()
        self.collecting = True

    def stop_collect(self):
        self.collecting = False


class ImageCalibSaveNode(Node):
    def __init__(self):
        super().__init__('image_calib_save_extrinsics')

        self.declare_parameter('config_path', '')
        self.config_path = resolve_config_path(str(self.get_parameter('config_path').value or ''))

        cfg = load_yaml(self.config_path)
        cam_cfg = cfg.get('camera', {})
        capture_ids = [int(x) for x in cam_cfg.get('ids', [0, 2, 4, 6, 8, 10, 12, 14])]
        self.cam_to_video = resolve_cam_to_video(capture_ids, cam_cfg.get('index_map', {}))
        self.camera_ids = sorted(self.cam_to_video.keys())

        self.aruco = ArucoDetector()
        self.streams = {cam_id: TopicCameraStream(cam_id, self.aruco) for cam_id in self.camera_ids}

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.subs = []
        for cam_id in self.camera_ids:
            topic = f'/helmet/cam{cam_id}/image_raw'
            sub = self.create_subscription(Image, topic, self._make_callback(cam_id), qos_profile)
            self.subs.append(sub)
            self.get_logger().info(f'Subscribed: {topic}')

    def _make_callback(self, cam_id: int):
        def callback(msg: Image):
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1).copy()
            self.streams[cam_id].on_image(frame)

        return callback


def save_results(streams: dict[int, TopicCameraStream], config_path):
    print(f"\n{'='*50}")
    print('处理标定结果...')

    cfg = load_yaml(config_path)

    results = {}
    vid_map = {}
    missing = []

    for cam_id in sorted(streams.keys()):
        stream = streams[cam_id]
        n = len(stream.samples_t)
        if n < 1:
            missing.append(cam_id)
            print(f'  ⚠ cam{cam_id}: 无样本（未检测到 ArUco）')
            continue

        tvecs = np.array(stream.samples_t)
        quats = np.array(stream.samples_q)
        tvec = np.median(tvecs, axis=0)
        quat = quat_mean(quats)

        name = f'video{cam_id}'
        results[name] = {
            'translation': [round(float(v), 6) for v in tvec],
            'rotation': [round(float(v), 8) for v in quat],
            'n_samples': n,
        }
        vid_map[name] = cam_id
        print(f'  cam{cam_id}: {n} samples  t=[{tvec[0]:.4f}, {tvec[1]:.4f}, {tvec[2]:.4f}]')

    if not results:
        print('没有有效标定数据，放弃保存')
        return

    if missing:
        print(f'\n  ⚠ 以下相机无样本，将保留旧的外参数据: {missing}')

    idx_map, cam_order = build_index_map(results, vid_map)

    print('\n自动排序生成 index_map:')
    for cam_id, video_id in sorted(idx_map.items()):
        print(f'  cam{cam_id} <- cam{video_id}')

    extrinsics = {}
    for new_cam_id, old_name in enumerate(cam_order):
        item = results[old_name]
        extrinsics[old_name] = {
            'physical_id': new_cam_id,
            **item,
        }

    ref_name = cam_order[REF_CAM_ID] if REF_CAM_ID < len(cam_order) else None
    if ref_name and ref_name in extrinsics:
        print(f'\n以 {ref_name} (新编号 cam{REF_CAM_ID}) 为基准，转换为相对坐标系')
        extrinsics = rebase_extrinsics(extrinsics, ref_name)
    else:
        print(f'  ⚠ 未找到 cam{REF_CAM_ID} 的数据，外参保持 marker_0 坐标系')

    final_ext = {}
    for new_cam_id, old_name in enumerate(cam_order):
        final_ext[f'cam{new_cam_id}'] = extrinsics[old_name]

    old_ext = cfg.get('camera', {}).get('extrinsics', {})
    old_idx_map = cfg.get('camera', {}).get('index_map', {})

    next_id = len(final_ext)
    for cam_id in missing:
        found_old = False
        for ck, cv in old_idx_map.items():
            if int(cv) == cam_id:
                old_cam_key = f'cam{ck}'
                if old_cam_key in old_ext:
                    final_ext[f'cam{next_id}'] = {
                        **old_ext[old_cam_key],
                        'physical_id': next_id,
                        'n_samples': 0,
                    }
                    idx_map[next_id] = cam_id
                    print(f'  cam{next_id} <- cam{cam_id} (保留旧外参)')
                    next_id += 1
                    found_old = True
                break

        if not found_old:
            final_ext[f'cam{next_id}'] = {
                'physical_id': next_id,
                'translation': [0.0, 0.0, 0.0],
                'rotation': [0.0, 0.0, 0.0, 1.0],
                'n_samples': 0,
            }
            idx_map[next_id] = cam_id
            print(f'  cam{next_id} <- cam{cam_id} (无旧数据，填 identity)')
            next_id += 1

    cam_cfg = cfg.setdefault('camera', {})
    cam_cfg['index_map'] = {int(k): int(v) for k, v in idx_map.items()}
    cam_cfg['extrinsics'] = final_ext

    save_yaml(config_path, cfg)

    total = len(streams)
    ok_count = total - len(missing)
    print(f'\n✓ 已保存到 {config_path}')
    print(f'  标定成功: {ok_count}/{total} 个相机')
    print(f'  index_map: {len(idx_map)} 个映射')
    print(f'  extrinsics: {len(final_ext)} 个相机')
    if missing:
        print(f'  ⚠ 无样本相机: cam{missing} (已用旧数据或identity填充)')
    print(f"{'='*50}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageCalibSaveNode()

    print(f'订阅 {len(node.camera_ids)} 路话题: {node.camera_ids}')
    print('实时预览中…  按 y 保存标定 | 按 q 退出\n')

    n_cams = len(node.camera_ids)
    cols = min(4, n_cams)
    rows = (n_cams + cols - 1) // cols
    thumb_w, thumb_h = 320, 240

    saving = False
    save_start = 0.0

    cv2.namedWindow('ROS2 ArUco Calibration', cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)

            tiles = []
            for cam_id in sorted(node.streams.keys()):
                stream = node.streams[cam_id]
                frame, tvec, _quat, detected = stream.get_state()

                if frame is None:
                    tile = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
                else:
                    tile = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_LINEAR)

                color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.putText(tile, f'cam{cam_id}', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

                if detected and tvec is not None:
                    t_str = f"t=[{tvec[0]:+.3f},{tvec[1]:+.3f},{tvec[2]:+.3f}]"
                    cv2.putText(tile, t_str, (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

                if saving:
                    n_samples = len(stream.samples_t)
                    cv2.putText(tile, f'Collecting: {n_samples}', (5, thumb_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

                tiles.append(tile)

            while len(tiles) < rows * cols:
                tiles.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))

            grid_rows = []
            for r in range(rows):
                row_tiles = tiles[r * cols:(r + 1) * cols]
                grid_rows.append(np.hstack(row_tiles))
            grid = np.vstack(grid_rows)

            bar_h = 30
            bar = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
            if saving:
                elapsed = time.monotonic() - save_start
                remaining = max(0.0, COLLECT_SECONDS - elapsed)
                msg = f'  Collecting samples... {remaining:.1f}s remaining  |  Press q to cancel'
                cv2.putText(bar, msg, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
            else:
                msg = '  Press [y] to calibrate & save  |  Press [q] to quit'
                cv2.putText(bar, msg, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            grid = np.vstack([grid, bar])

            cv2.imshow('ROS2 ArUco Calibration', grid)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('y') and not saving:
                saving = True
                save_start = time.monotonic()
                for stream in node.streams.values():
                    stream.start_collect()
                print(f'开始收集样本，持续 {COLLECT_SECONDS:.0f} 秒...')

            if saving and (time.monotonic() - save_start) >= COLLECT_SECONDS:
                for stream in node.streams.values():
                    stream.stop_collect()
                save_results(node.streams, node.config_path)
                saving = False
                print('\n可以继续预览，再按 y 重新标定，按 q 退出\n')

            if key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
