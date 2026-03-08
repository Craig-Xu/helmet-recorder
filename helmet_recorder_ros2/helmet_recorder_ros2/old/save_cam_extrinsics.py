#!/usr/bin/env python3

import time
from collections import defaultdict

import numpy as np
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node

from .common import load_yaml, resolve_cam_to_video, resolve_config_path, save_yaml

COLLECT_SECONDS = 5.0
MIN_SAMPLES = 10
PARENT_FRAME = 'marker_0'
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


def build_index_map_from_extrinsics(results: dict, cam_to_video: dict, axis: int = INDEX_AXIS, desc: bool = INDEX_DESC):
    sortable = []
    for cam_name, item in results.items():
        try:
            cam_id = int(cam_name.replace('cam', ''))
            t = item['translation']
            video_id = int(cam_to_video.get(cam_id, cam_id))
            sortable.append((cam_id, float(t[axis]), float(t[2]), video_id))
        except Exception:
            continue

    sortable.sort(key=lambda x: x[1], reverse=desc)

    n = len(sortable)
    if n >= MIDDLE_COUNT:
        start = (n - MIDDLE_COUNT) // 2
        end = start + MIDDLE_COUNT
        middle = sortable[start:end]
        middle.sort(key=lambda x: x[2], reverse=not MIDDLE_Z_ASC)
        sortable = sortable[:start] + middle + sortable[end:]

    index_map = {idx: int(video_id) for idx, (_, _, _, video_id) in enumerate(sortable)}
    for a, b in SWAP_PAIRS:
        if a in index_map and b in index_map:
            index_map[a], index_map[b] = index_map[b], index_map[a]
    return index_map


class ExtrinsicsSaver(Node):
    def __init__(self, camera_ids: list[int], cam_to_video: dict[int, int], config_path):
        super().__init__('extrinsics_saver')
        self.camera_ids = camera_ids
        self.cam_to_video = cam_to_video
        self.config_path = config_path

        self._tf_buf = tf2_ros.Buffer(cache_time=Duration(seconds=30))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        self._samples = defaultdict(lambda: {'tvecs': [], 'quats': []})
        self._done = set()
        self._t_start = time.monotonic()
        self._timer = self.create_timer(0.05, self._poll_tf)

    def _poll_tf(self):
        now = rclpy.time.Time()
        for vid in self.camera_ids:
            if vid in self._done:
                continue
            child = f'cam{vid}'
            try:
                tf = self._tf_buf.lookup_transform(PARENT_FRAME, child, now, timeout=Duration(seconds=0.01))
                t = tf.transform.translation
                q = tf.transform.rotation
                self._samples[vid]['tvecs'].append([t.x, t.y, t.z])
                self._samples[vid]['quats'].append([q.x, q.y, q.z, q.w])
            except Exception:
                pass

        elapsed = time.monotonic() - self._t_start
        for vid in [v for v in self.camera_ids if v not in self._done]:
            if len(self._samples[vid]['tvecs']) >= MIN_SAMPLES and elapsed >= COLLECT_SECONDS:
                self._done.add(vid)

        if elapsed >= COLLECT_SECONDS:
            self._finish()

    def _finish(self):
        self._timer.cancel()

        results = {}
        for vid in self.camera_ids:
            samples = self._samples[vid]
            if not samples['tvecs']:
                continue

            tvec = np.median(np.array(samples['tvecs']), axis=0)
            quat = quat_mean(np.array(samples['quats']))
            results[f'cam{vid}'] = {
                'physical_id': int(vid),
                'translation': [round(float(v), 6) for v in tvec],
                'rotation': [round(float(v), 8) for v in quat],
                'n_samples': len(samples['tvecs']),
            }

        new_index_map = build_index_map_from_extrinsics(results, self.cam_to_video)
        old_cam_to_new_cam = {}
        for new_cam, video_id in new_index_map.items():
            for old_cam, old_video in self.cam_to_video.items():
                if int(old_video) == int(video_id):
                    old_cam_to_new_cam[int(old_cam)] = int(new_cam)
                    break

        for cam_name, item in results.items():
            old_cam = int(cam_name.replace('cam', ''))
            if old_cam in old_cam_to_new_cam:
                item['physical_id'] = int(old_cam_to_new_cam[old_cam])

        ref_video = new_index_map.get(REF_CAM_ID)
        ref_old_cam_name = None
        if ref_video is not None:
            for old_cam, old_video in self.cam_to_video.items():
                if int(old_video) == int(ref_video):
                    ref_old_cam_name = f'cam{old_cam}'
                    break

        if ref_old_cam_name and ref_old_cam_name in results:
            results = rebase_extrinsics(results, ref_old_cam_name)

        cfg = load_yaml(self.config_path)
        cam_cfg = cfg.setdefault('camera', {})
        cam_cfg['extrinsics'] = results
        if new_index_map:
            cam_cfg['index_map'] = {int(k): int(v) for k, v in new_index_map.items()}
        save_yaml(self.config_path, cfg)

        self.get_logger().info(f'extrinsics saved: {self.config_path}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    temp_node = Node('save_extrinsics_args')
    temp_node.declare_parameter('config_path', '')
    config_path = resolve_config_path(str(temp_node.get_parameter('config_path').value or ''))
    temp_node.destroy_node()

    cfg = load_yaml(config_path)
    cam_cfg = cfg.get('camera', {})
    capture_ids = [int(x) for x in cam_cfg.get('ids', [0, 2, 4, 6, 8, 10, 12, 14])]
    cam_to_video = resolve_cam_to_video(capture_ids, cam_cfg.get('index_map', {}))
    cam_ids = sorted(cam_to_video.keys())

    node = ExtrinsicsSaver(cam_ids, cam_to_video, config_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
