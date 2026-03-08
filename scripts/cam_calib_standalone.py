#!/usr/bin/env python3
"""
standalone_calib.py
───────────────────
独立的多相机 ArUco 标定脚本（无需 ROS2）。

直接打开所有 /dev/video* 相机，实时检测 ArUco marker_0，
在 OpenCV 窗口中显示检测结果和各相机状态。

按键:
    y  —  保存标定结果（index_map + extrinsics）写入 config/config.yaml
    q  —  退出

用法:
    python3 scripts/standalone_calib.py
"""

import os
import time
import threading
from collections import defaultdict, deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from common import CONFIG_PATH, load_yaml, save_yaml

# ── ArUco 参数 ──
ARUCO_DICT      = cv2.aruco.DICT_6X6_1000
TARGET_MARKER   = 0
MARKER_SIZE     = 0.1        # 标记边长 (米)

# ── 相机参数 ──
FPS             = 30
COLLECT_SECONDS = 5.0        # 按 y 后收集多少秒
MIN_SAMPLES     = 10

# ── 稳定性参数 ──
BUFFER_SIZE     = 7
MAX_REPROJ_ERR  = 1.5        # px
MAX_JUMP_T      = 0.05       # 米
MAX_JUMP_R      = 10.0       # 度

# ── 排序参数 ──
INDEX_AXIS      = 0          # 0=x
INDEX_DESC      = False
MIDDLE_COUNT    = 4
MIDDLE_Z_ASC    = True
SWAP_PAIRS      = []   # 业务修正：排序后交换这些 cam 对
REF_CAM_ID      = 3          # 主摄像头新编号


# ═════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════════════════

def _load_config() -> dict:
    return load_yaml(CONFIG_PATH)


def _open_camera(dev_id: int, width: int, height: int, fps: int):
    backend = (cv2.CAP_V4L2
               if os.name == "posix" and hasattr(cv2, "CAP_V4L2")
               else cv2.CAP_ANY)
    cap = cv2.VideoCapture(dev_id, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(dev_id)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    for _ in range(3):
        cap.grab()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  /dev/video{dev_id}: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
    return cap


def quat_mean(quats: np.ndarray) -> np.ndarray:
    Q = np.array(quats)
    Q[Q[:, 3] < 0] *= -1
    _, vecs = np.linalg.eigh(Q.T @ Q)
    q = vecs[:, -1]
    return q / np.linalg.norm(q)


def _rebase_extrinsics(results: dict, ref_cam_name: str) -> dict:
    if ref_cam_name not in results:
        return results
    ref = results[ref_cam_name]
    R_ref = Rotation.from_quat(ref["rotation"])
    t_ref = np.array(ref["translation"])
    R_ref_inv = R_ref.inv()

    rebased = {}
    for cam_name, item in results.items():
        R_cam = Rotation.from_quat(item["rotation"])
        t_cam = np.array(item["translation"])
        R_rel = R_ref_inv * R_cam
        t_rel = R_ref_inv.apply(t_cam - t_ref)
        rebased[cam_name] = {
            **item,
            "translation": [round(float(v), 6) for v in t_rel],
            "rotation":    [round(float(v), 8) for v in R_rel.as_quat()],
        }
    return rebased


def build_index_map(results: dict, video_ids_per_cam: dict) -> dict:
    """基于 marker 坐标系下的 x 排序，生成 cam_id -> video_id 映射。"""
    sortable = []
    for cam_name, item in results.items():
        t = item["translation"]
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

    # 业务修正：交换指定 cam 对
    for a, b in SWAP_PAIRS:
        if a in idx_map and b in idx_map:
            idx_map[a], idx_map[b] = idx_map[b], idx_map[a]

    # 新 cam_id -> 原 cam name 映射（用于 rebase 和 physical_id 回填）
    cam_order = [name for name, _, _, _ in sortable]
    for a, b in SWAP_PAIRS:
        if a < len(cam_order) and b < len(cam_order):
            cam_order[a], cam_order[b] = cam_order[b], cam_order[a]

    return idx_map, cam_order


# ═════════════════════════════════════════════════════════════════════════════
# ArUco 检测器
# ═════════════════════════════════════════════════════════════════════════════

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

        self.camera_matrix = np.array([
            [500.0, 0, 320.0],
            [0, 500.0, 240.0],
            [0, 0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        h = MARKER_SIZE / 2
        self.obj_points = np.array([
            [-h,  h, 0], [ h,  h, 0],
            [ h, -h, 0], [-h, -h, 0],
        ], dtype=np.float32)

    def detect(self, frame: np.ndarray):
        """检测 marker_0，返回 (t_inv, quat, annotated_frame) 或 (None, None, frame)。"""
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
                self.obj_points, img_pts,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            rvec, tvec = cv2.solvePnPRefineLM(
                self.obj_points, img_pts,
                self.camera_matrix, self.dist_coeffs,
                rvec, tvec,
            )

            # 重投影误差
            proj, _ = cv2.projectPoints(
                self.obj_points, rvec, tvec,
                self.camera_matrix, self.dist_coeffs,
            )
            err = float(np.mean(np.linalg.norm(
                proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)))
            if err > MAX_REPROJ_ERR:
                continue

            # marker→camera 逆变换
            R, _ = cv2.Rodrigues(rvec)
            R_inv = R.T
            t_inv = (-R_inv @ tvec).flatten()
            quat = Rotation.from_matrix(R_inv).as_quat()  # xyzw

            # 画检测框 + 轴
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.drawFrameAxes(vis, self.camera_matrix, self.dist_coeffs,
                              rvec, tvec, MARKER_SIZE * 0.5)
            return t_inv, quat, vis

        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        return None, None, vis


# ═════════════════════════════════════════════════════════════════════════════
# 单相机采集线程
# ═════════════════════════════════════════════════════════════════════════════

class CameraThread:
    def __init__(self, dev_id: int, width: int, height: int, aruco: ArucoDetector):
        self.dev_id = dev_id
        self.aruco = aruco
        self.cap = _open_camera(dev_id, width, height, FPS)
        self.alive = self.cap is not None

        # 平滑缓冲
        self.pose_buf: deque = deque(maxlen=BUFFER_SIZE)
        self.last_tvec = None
        self.last_quat = None

        # 样本收集 (按y后启动)
        self.collecting = False
        self.samples_t: list = []
        self.samples_q: list = []

        # 最新帧 + 位姿（供主线程读取）
        self._lock = threading.Lock()
        self._frame = None
        self._tvec_smooth = None
        self._quat_smooth = None
        self._detected = False

        self._stop = threading.Event()
        if self.alive:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            if not self.cap.grab():
                time.sleep(0.05)
                continue
            ret, frame = self.cap.retrieve()
            if not ret:
                continue

            t_inv, quat, vis = self.aruco.detect(frame)

            t_smooth, q_smooth = self._tvec_smooth, self._quat_smooth
            detected = t_inv is not None

            if t_inv is not None:
                # 收集样本（不受跳变过滤影响，最终用中值 + 四元数均值去噪）
                if self.collecting:
                    self.samples_t.append(t_inv.copy())
                    self.samples_q.append(quat.copy())

                # 跳变检测（仅影响实时预览平滑，不影响样本收集）
                skip = False
                if self.last_tvec is not None:
                    dt = np.linalg.norm(t_inv - self.last_tvec)
                    dangle = 2 * np.degrees(np.arccos(
                        np.clip(abs(np.dot(quat, self.last_quat)), 0, 1)))
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

            with self._lock:
                self._frame = vis
                self._tvec_smooth = t_smooth
                self._quat_smooth = q_smooth
                self._detected = detected

    def get_state(self):
        with self._lock:
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

    def stop(self):
        self._stop.set()
        if self.alive and self.cap:
            self.cap.release()


# ═════════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════════

def main():
    cfg = _load_config()
    cam_cfg = cfg.get("camera", {})
    dev_ids = [int(x) for x in cam_cfg.get("ids", [0, 2, 4, 6, 8, 10, 12, 14])]
    width   = int(cam_cfg.get("width", 640))
    height  = int(cam_cfg.get("height", 480))

    print(f"打开 {len(dev_ids)} 个相机: {dev_ids}")
    aruco = ArucoDetector()

    threads: dict[int, CameraThread] = {}
    for did in dev_ids:
        ct = CameraThread(did, width, height, aruco)
        if ct.alive:
            threads[did] = ct
        else:
            print(f"  ⚠ /dev/video{did} 打开失败，跳过")

    if not threads:
        print("没有可用相机，退出")
        return

    print(f"\n成功打开 {len(threads)} 个相机")
    print("实时预览中…  按 y 保存标定 | 按 q 退出\n")

    # 预览缩略图尺寸
    n_cams = len(threads)
    cols = min(4, n_cams)
    rows = (n_cams + cols - 1) // cols
    thumb_w, thumb_h = 320, 240

    saving = False
    save_start = 0.0

    try:
        while True:
            tiles = []

            for did in sorted(threads.keys()):
                ct = threads[did]
                frame, tvec, quat, detected = ct.get_state()

                if frame is None:
                    tile = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
                else:
                    tile = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_LINEAR)

                # 状态文字
                label = f"/dev/video{did}"
                color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.putText(tile, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 1, cv2.LINE_AA)

                if detected and tvec is not None:
                    t_str = f"t=[{tvec[0]:+.3f},{tvec[1]:+.3f},{tvec[2]:+.3f}]"
                    cv2.putText(tile, t_str, (5, 38), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (200, 200, 200), 1, cv2.LINE_AA)

                if saving:
                    n_samples = len(ct.samples_t)
                    cv2.putText(tile, f"Collecting: {n_samples}", (5, thumb_h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

                tiles.append(tile)

            # 补齐空位
            while len(tiles) < rows * cols:
                tiles.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))

            # 拼接网格
            grid_rows = []
            for r in range(rows):
                row_tiles = tiles[r * cols: (r + 1) * cols]
                grid_rows.append(np.hstack(row_tiles))
            grid = np.vstack(grid_rows)

            # 底部状态栏
            bar_h = 30
            bar = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
            if saving:
                elapsed = time.monotonic() - save_start
                remaining = max(0, COLLECT_SECONDS - elapsed)
                msg = f"  Collecting samples... {remaining:.1f}s remaining  |  Press q to cancel"
                cv2.putText(bar, msg, (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 200, 255), 1, cv2.LINE_AA)
            else:
                msg = "  Press [y] to calibrate & save  |  Press [q] to quit"
                cv2.putText(bar, msg, (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (200, 200, 200), 1, cv2.LINE_AA)
            grid = np.vstack([grid, bar])

            cv2.imshow("ArUco Calibration", grid)
            key = cv2.waitKey(30) & 0xFF

            # ── 按 y 开始收集 ──
            if key == ord("y") and not saving:
                saving = True
                save_start = time.monotonic()
                for ct in threads.values():
                    ct.start_collect()
                print(f"开始收集样本，持续 {COLLECT_SECONDS:.0f} 秒...")

            # ── 收集超时 → 保存 ──
            if saving and (time.monotonic() - save_start) >= COLLECT_SECONDS:
                for ct in threads.values():
                    ct.stop_collect()
                _save_results(threads)
                saving = False
                print("\n可以继续预览，再按 y 重新标定，按 q 退出\n")

            # ── 按 q 退出 ──
            if key == ord("q"):
                break

    finally:
        for ct in threads.values():
            ct.stop()
        cv2.destroyAllWindows()


def _save_results(threads: dict[int, "CameraThread"]):
    """从各线程收集样本，计算外参，排序，rebase，保存到 config/config.yaml。"""
    print(f"\n{'='*50}")
    print("处理标定结果...")

    # 0. 重新读取 config/config.yaml，防止覆盖中途修改
    cfg = _load_config()

    # 1. 汇总各相机在 marker 坐标系下的平均位姿
    results = {}          # key: "video{did}"
    vid_map = {}          # "video{did}" -> did
    missing = []

    for did in sorted(threads.keys()):
        ct = threads[did]
        n = len(ct.samples_t)
        if n < 1:
            missing.append(did)
            print(f"  ⚠ /dev/video{did}: 无样本（未检测到 ArUco）")
            continue

        tvecs = np.array(ct.samples_t)
        quats = np.array(ct.samples_q)
        tvec = np.median(tvecs, axis=0)
        quat = quat_mean(quats)

        name = f"video{did}"
        results[name] = {
            "translation": [round(float(v), 6) for v in tvec],
            "rotation":    [round(float(v), 8) for v in quat],
            "n_samples":   n,
        }
        vid_map[name] = did
        print(f"  /dev/video{did}: {n} samples  t=[{tvec[0]:.4f}, {tvec[1]:.4f}, {tvec[2]:.4f}]")

    if not results:
        print("没有有效标定数据，放弃保存")
        return

    if missing:
        print(f"\n  ⚠ 以下相机无样本，将保留旧的外参数据: {missing}")

    # 2. 基于 marker 坐标系 x 排序，生成 index_map
    idx_map, cam_order = build_index_map(results, vid_map)

    print(f"\n自动排序生成 index_map:")
    for cam_id, video_id in sorted(idx_map.items()):
        print(f"  cam{cam_id} <- /dev/video{video_id}")

    # 3. 用原 marker 坐标系数据构建 extrinsics（先用旧 key），回填 physical_id
    extrinsics = {}
    for new_cam_id, old_name in enumerate(cam_order):
        item = results[old_name]
        extrinsics[old_name] = {
            "physical_id": new_cam_id,
            **item,
        }

    # 4. 找到参考相机，做 rebase
    ref_name = cam_order[REF_CAM_ID] if REF_CAM_ID < len(cam_order) else None
    if ref_name and ref_name in extrinsics:
        print(f"\n以 {ref_name} (新编号 cam{REF_CAM_ID}) 为基准，转换为相对坐标系")
        extrinsics = _rebase_extrinsics(extrinsics, ref_name)
    else:
        print(f"  ⚠ 未找到 cam{REF_CAM_ID} 的数据，外参保持 marker_0 坐标系")

    # 5. 重命名 key 为 cam{new_id}
    final_ext = {}
    for new_cam_id, old_name in enumerate(cam_order):
        final_ext[f"cam{new_cam_id}"] = extrinsics[old_name]

    # 5b. 补充无样本相机的旧外参（如果 config 已有）
    old_ext = cfg.get("camera", {}).get("extrinsics", {})
    next_id = len(final_ext)
    for did in missing:
        # 查找旧 config 中该 video_id 对应的外参
        found_old = False
        for old_key, old_val in old_ext.items():
            old_idx_map = cfg.get("camera", {}).get("index_map", {})
            for ck, cv in old_idx_map.items():
                if int(cv) == did:
                    # 找到旧编号
                    old_cam_key = f"cam{ck}"
                    if old_cam_key in old_ext:
                        final_ext[f"cam{next_id}"] = {
                            **old_ext[old_cam_key],
                            "physical_id": next_id,
                            "n_samples": 0,
                        }
                        idx_map[next_id] = did
                        print(f"  cam{next_id} <- /dev/video{did} (保留旧外参)")
                        next_id += 1
                        found_old = True
                    break
            if found_old:
                break
        if not found_old:
            # 没有旧数据，填 identity
            final_ext[f"cam{next_id}"] = {
                "physical_id": next_id,
                "translation": [0.0, 0.0, 0.0],
                "rotation":    [0.0, 0.0, 0.0, 1.0],
                "n_samples":   0,
            }
            idx_map[next_id] = did
            print(f"  cam{next_id} <- /dev/video{did} (无旧数据，填 identity)")
            next_id += 1

    # 6. 写入 config/config.yaml
    cam_cfg = cfg.setdefault("camera", {})
    cam_cfg["index_map"] = {int(k): int(v) for k, v in idx_map.items()}
    cam_cfg["extrinsics"] = final_ext

    save_yaml(CONFIG_PATH, cfg)

    total = len(threads)
    ok_count = total - len(missing)
    print(f"\n✓ 已保存到 {CONFIG_PATH}")
    print(f"  标定成功: {ok_count}/{total} 个相机")
    print(f"  index_map: {len(idx_map)} 个映射")
    print(f"  extrinsics: {len(final_ext)} 个相机")
    if missing:
        print(f"  ⚠ 无样本相机: /dev/video{missing} (已用旧数据或identity填充)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
