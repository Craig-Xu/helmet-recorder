#!/usr/bin/env python3
"""
ros2_save_cam_extrinsics.py
───────────────────────────
收集 ArUco 标定节点发布的 TF (marker_0 → cam{v4l2_id})，
取若干帧平均后，将所有相机外参写入 config.yaml。

前提条件:
  - ros2_camera_publisher.py   已在运行（发布图像）
  - ros2_camera_aruco_calib.py 已在运行（持续发布 TF）
  - config.yaml 中 camera.index_map 已正确填写

用法:
    python3 scripts/ros2_save_cam_extrinsics.py

成功后会在 config.yaml 的 camera.extrinsics 节点下写入：
    cam{v4l2_id}:
      physical_id: {物理编号}
      translation: [tx, ty, tz]          # 米
      rotation:    [qx, qy, qz, qw]
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros
from scipy.spatial.transform import Rotation

# ── 参数 ──────────────────────────────────────────────────────────────────────
COLLECT_SECONDS = 5.0    # 每个相机收集多少秒的 TF 样本
MIN_SAMPLES     = 10     # 至少需要多少样本才认为该相机已完成
PARENT_FRAME    = 'marker_0'
CONFIG_PATH     = _HERE / 'config.yaml'
INDEX_AXIS      = 0      # 0:x, 1:y, 2:z
INDEX_DESC      = False  # False: x 从小到大 -> physical_id/cam_id 从 0 到 N-1
MIDDLE_COUNT    = 4      # 中间相机数量
MIDDLE_Z_ASC    = True   # 中间 4 个按 z 从小到大
SWAP_CAM_4_5    = True   # 业务修正：交换 cam4 / cam5 顺序


def quat_mean(quats: np.ndarray) -> np.ndarray:
    """特征向量法求四元数均值（适用于小角度偏差）"""
    Q = np.array(quats)
    Q[Q[:, 3] < 0] *= -1
    _, vecs = np.linalg.eigh(Q.T @ Q)
    q = vecs[:, -1]
    return q / np.linalg.norm(q)


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


def build_index_map_from_extrinsics(results: dict, cam_to_video: dict, axis: int = INDEX_AXIS, desc: bool = INDEX_DESC) -> dict:
    """
    根据外参平移量对相机做排序，并生成 cam_id -> /dev/video_id 映射。
    默认策略：按 translation[0] (x) 排序，且 x 越大 cam_id 越小（右->左）。
    """
    sortable = []
    for cam_name, item in results.items():
        try:
            cam_id = int(cam_name.replace('cam', ''))
            t = item['translation']
            video_id = int(cam_to_video.get(cam_id, cam_id))
            sortable.append((cam_id, float(t[axis]), float(t[2]), video_id))
        except Exception:
            continue

    # 1) 先按 x 轴排序
    sortable.sort(key=lambda x: x[1], reverse=desc)

    # 2) 中间 4 个再按 z 轴排序
    n = len(sortable)
    if n >= MIDDLE_COUNT:
        mid_start = (n - MIDDLE_COUNT) // 2
        mid_end = mid_start + MIDDLE_COUNT
        middle = sortable[mid_start:mid_end]
        middle.sort(key=lambda x: x[2], reverse=not MIDDLE_Z_ASC)
        sortable = sortable[:mid_start] + middle + sortable[mid_end:]

    index_map = {idx: int(video_id) for idx, (_, _, _, video_id) in enumerate(sortable)}

    # 业务修正：仅交换 cam4 与 cam5，其它顺序保持不变
    if SWAP_CAM_4_5 and 4 in index_map and 5 in index_map:
        index_map[4], index_map[5] = index_map[5], index_map[4]

    return index_map


class ExtrinsicsSaver(Node):
    def __init__(self, camera_ids: list, cam_to_video: dict):
        super().__init__('extrinsics_saver')
        self.camera_ids = camera_ids
        self.cam_to_video  = cam_to_video  # cam_id -> /dev/video_id

        # TF 缓冲
        self._tf_buf = tf2_ros.Buffer(cache_time=Duration(seconds=30))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        # 样本收集: {v4l2_id: {'tvecs': [], 'quats': []}}
        self._samples: dict = defaultdict(lambda: {'tvecs': [], 'quats': []})
        self._done: set = set()

        self._t_start = time.monotonic()
        self._timer = self.create_timer(0.05, self._poll_tf)  # 20 Hz 轮询

        self.get_logger().info(
            f'开始收集 TF，持续 {COLLECT_SECONDS:.0f}s ...\n'
            f'等待 {len(camera_ids)} 个相机: {camera_ids}'
        )

    def _poll_tf(self):
        now = rclpy.time.Time()
        for vid in self.camera_ids:
            if vid in self._done:
                continue
            child = f'cam{vid}'
            try:
                tf = self._tf_buf.lookup_transform(
                    PARENT_FRAME, child, now,
                    timeout=Duration(seconds=0.01)
                )
                t = tf.transform.translation
                q = tf.transform.rotation
                self._samples[vid]['tvecs'].append([t.x, t.y, t.z])
                self._samples[vid]['quats'].append([q.x, q.y, q.z, q.w])
            except Exception:
                pass  # TF 尚未就绪，忽略

        # 检查是否超时或所有相机已就绪
        elapsed = time.monotonic() - self._t_start
        pending = [v for v in self.camera_ids if v not in self._done]

        # 标记样本充足的相机为完成
        for vid in list(pending):
            n = len(self._samples[vid]['tvecs'])
            if n >= MIN_SAMPLES and elapsed >= COLLECT_SECONDS:
                self._done.add(vid)
                self.get_logger().info(
                    f'  cam{vid} (/dev/video{self.cam_to_video.get(vid, "?")})'
                    f' — {n} 样本 OK'
                )

        if elapsed >= COLLECT_SECONDS:
            self._finish()

    def _finish(self):
        self._timer.cancel()

        results: dict = {}
        missing = []

        for vid in self.camera_ids:
            s = self._samples[vid]
            n = len(s['tvecs'])
            if n < 1:
                missing.append(vid)
                self.get_logger().warn(
                    f'cam{vid} 无样本 — 未检测到 ArUco 或 TF 未发布，跳过'
                )
                continue

            tvec = np.median(np.array(s['tvecs']), axis=0)
            quat = quat_mean(np.array(s['quats']))

            results[f'cam{vid}'] = {
                'physical_id':  int(vid),
                'translation':  [round(float(v), 6) for v in tvec],
                'rotation':     [round(float(v), 8) for v in quat],
                'n_samples':    n,
            }
            self.get_logger().info(
                f'cam{vid} (/dev/video{self.cam_to_video.get(vid, "?")}): '
                f't=[{tvec[0]:.4f}, {tvec[1]:.4f}, {tvec[2]:.4f}] '
                f'q=[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]'
            )

        # 基于外参自动重建 index_map(cam->video)，并回填 physical_id
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

        if new_index_map:
            self.get_logger().info('根据 extrinsics 自动更新 index_map:')
            for cam_id, video_id in sorted(new_index_map.items(), key=lambda kv: kv[0]):
                self.get_logger().info(f'  /helmet/cam{cam_id} <- /dev/video{video_id}')

        # 写回 config.yaml
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f) or {}

        cam_cfg = cfg.setdefault('camera', {})
        cam_cfg['extrinsics'] = results
        if new_index_map:
            cam_cfg['index_map'] = {int(k): int(v) for k, v in new_index_map.items()}

        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        total = len(self.camera_ids)
        saved = total - len(missing)
        self.get_logger().info(
            f'\n{"="*50}\n'
            f'外参保存完毕: {saved}/{total} 个相机写入 {CONFIG_PATH}\n'
            + (f'未能标定: {missing}\n' if missing else '') +
            f'{"="*50}'
        )
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    cam_cfg   = cfg.get('camera', {})
    capture_ids = [int(x) for x in cam_cfg.get('ids', [0, 2, 4, 6, 8, 10, 12, 14])]
    cam_to_video = _resolve_cam_to_video(capture_ids, cam_cfg.get('index_map', {}))
    cam_ids = sorted(cam_to_video.keys())

    node = ExtrinsicsSaver(cam_ids, cam_to_video)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
