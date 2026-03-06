#!/usr/bin/env python3
"""
cam_intrinsics_from_video.py
────────────────────────────
从 ~/recordings 下的录制视频 (*.mp4) 中检测棋盘格角点，
对每个视频独立进行相机内参标定，输出每个视频对应相机的内参矩阵和畸变系数。

棋盘格参数（默认）:
    内角点数: 8 × 11  （即棋盘格 9列×12行 方块，内部交叉点 8×11）
    方块边长: 20 mm

高级检测策略:
    1. 优先使用 findChessboardCornersSB (OpenCV 4.0+ 的鲁棒检测器)
    2. 多尺度检测: 原尺寸 + 缩小到 50%/75%
    3. 多种预处理: 原图/CLAHE/锐化/去噪/对比度拉伸/自适应二值化
    4. 自动尝试两个方向 (cols×rows 和 rows×cols)
    5. 多种 flag 组合

可视化输出:
    - 实时弹窗显示每帧检测结果（绿框=成功, 红字=失败）
    - 保存每帧检测图到  <output>/vis/<name>/
    - 标定成功后保存去畸变前后对比图

用法:
    # 标定某次录制的所有视频（带可视化窗口）
    python3 scripts/cam_intrinsics_from_video.py ~/recordings/recording_20260305_225913

    # 不弹窗，只保存可视化图片
    python3 scripts/cam_intrinsics_from_video.py ~/recordings --no-show

    # 可选参数
    python3 scripts/cam_intrinsics_from_video.py ~/recordings \\
        --cols 8 --rows 11 --square-size 20 --sample-interval 10 --output-dir ./calib_results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# 导入项目配置
from common import CONFIG_PATH, load_yaml

# ── 默认棋盘格参数 ──
DEFAULT_COLS = 7           # 内角点列数
DEFAULT_ROWS = 10          # 内角点行数
DEFAULT_SQUARE_SIZE = 20   # 方块边长 (mm)

# ── 可视化缩放 ──
VIS_MAX_WIDTH = 960


# ─────────────────────────────────────────────────────────
#  辅助函数
# ─────────────────────────────────────────────────────────

def build_object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    """生成棋盘格三维世界坐标（z=0 平面，单位 mm）。"""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm
    return objp


def resize_for_display(img: np.ndarray, max_w: int = VIS_MAX_WIDTH) -> np.ndarray:
    """将图片等比缩放到 max_w 宽度以方便显示。"""
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


class ChessboardDetector:
    """
    高效棋盘格检测器。
    
    优化策略:
    1. 缓存上一次成功的方法/方向，优先尝试
    2. 渐进式检测：先用最快的方法，失败再逐步升级
    3. 快速预检：使用 FAST_CHECK 排除无棋盘帧
    4. 惰性计算：只在需要时生成预处理图像
    """
    
    def __init__(self, pattern_size: tuple[int, int]):
        self.default_pattern = pattern_size
        
        # 缓存上一次成功的配置
        self._last_orient: tuple[int, int] | None = None
        self._last_method: str | None = None  # "sb", "classic", "sb@0.5", "classic@0.5"
        self._last_preproc: str | None = None  # "raw", "clahe", "eq"
        
        # 统计
        self.cache_hits = 0
        self.total_detections = 0
    
    def _get_orientations(self) -> list[tuple[int, int]]:
        """返回要尝试的方向列表，优先使用上次成功的方向。"""
        cols, rows = self.default_pattern
        all_orients = [(cols, rows)]
        if cols != rows:
            all_orients.append((rows, cols))
        
        if self._last_orient and self._last_orient in all_orients:
            # 把上次成功的方向放到最前面
            all_orients.remove(self._last_orient)
            all_orients.insert(0, self._last_orient)
        
        return all_orients
    
    def _lazy_preprocess(self, gray: np.ndarray, name: str) -> np.ndarray:
        """惰性计算预处理图像。"""
        if name == "raw":
            return gray
        elif name == "clahe":
            return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        elif name == "eq":
            return cv2.equalizeHist(gray)
        return gray
    
    def _try_detect(self, gray: np.ndarray, orient: tuple[int, int], 
                    method: str, preproc: str, scale: float = 1.0
                   ) -> tuple[bool, np.ndarray | None]:
        """尝试单次检测。"""
        # 缩放处理
        if scale < 1.0:
            h, w = gray.shape[:2]
            small = cv2.resize(gray, (int(w * scale), int(h * scale)), 
                             interpolation=cv2.INTER_AREA)
            img = self._lazy_preprocess(small, preproc)
        else:
            img = self._lazy_preprocess(gray, preproc)
        
        try:
            if method.startswith("sb"):
                found, corners = cv2.findChessboardCornersSB(img, orient)
            else:
                flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | 
                        cv2.CALIB_CB_NORMALIZE_IMAGE | 
                        cv2.CALIB_CB_FAST_CHECK)
                found, corners = cv2.findChessboardCorners(img, orient, flags)
            
            if found and scale < 1.0:
                corners = corners / scale
            return found, corners if found else None
        except:
            return False, None
    
    def detect(self, gray: np.ndarray) -> tuple[bool, np.ndarray | None, tuple[int, int], str]:
        """
        检测棋盘格角点。
        
        Returns:
            (found, corners, used_pattern_size, method_description)
        """
        orientations = self._get_orientations()
        
        # ─── 阶段1: 尝试缓存的配置（最快路径）───
        if self._last_orient and self._last_method and self._last_preproc:
            scale = 0.5 if "@0.5" in self._last_method else 1.0
            method_base = self._last_method.replace("@0.5", "")
            
            found, corners = self._try_detect(
                gray, self._last_orient, method_base, self._last_preproc, scale
            )
            if found:
                self.cache_hits += 1
                self.total_detections += 1
                desc = f"{self._last_method}+{self._last_preproc}"
                return True, corners, self._last_orient, desc
        
        # ─── 阶段2: 快速检测（SB + raw，原尺寸）───
        for orient in orientations:
            found, corners = self._try_detect(gray, orient, "sb", "raw", 1.0)
            if found:
                self._update_cache(orient, "sb", "raw")
                return True, corners, orient, "SB+raw"
        
        # ─── 阶段3: 经典检测器 + 快速预检（更快排除）───
        for orient in orientations:
            found, corners = self._try_detect(gray, orient, "classic", "raw", 1.0)
            if found:
                self._update_cache(orient, "classic", "raw")
                return True, corners, orient, "classic+raw"
        
        # ─── 阶段4: 增强预处理（CLAHE/直方图均衡）───
        for preproc in ["clahe", "eq"]:
            for orient in orientations:
                # 优先 SB
                found, corners = self._try_detect(gray, orient, "sb", preproc, 1.0)
                if found:
                    self._update_cache(orient, "sb", preproc)
                    return True, corners, orient, f"SB+{preproc}"
                # 然后 classic
                found, corners = self._try_detect(gray, orient, "classic", preproc, 1.0)
                if found:
                    self._update_cache(orient, "classic", preproc)
                    return True, corners, orient, f"classic+{preproc}"
        
        # ─── 阶段5: 缩小到50%（高分辨率图像有效）───
        for preproc in ["raw", "clahe"]:
            for orient in orientations:
                found, corners = self._try_detect(gray, orient, "sb", preproc, 0.5)
                if found:
                    self._update_cache(orient, "sb@0.5", preproc)
                    return True, corners, orient, f"SB+{preproc}@0.5"
                found, corners = self._try_detect(gray, orient, "classic", preproc, 0.5)
                if found:
                    self._update_cache(orient, "classic@0.5", preproc)
                    return True, corners, orient, f"classic+{preproc}@0.5"
        
        return False, None, self.default_pattern, ""
    
    def _update_cache(self, orient: tuple[int, int], method: str, preproc: str):
        """更新缓存的成功配置。"""
        self._last_orient = orient
        self._last_method = method
        self._last_preproc = preproc
        self.total_detections += 1
    
    def get_stats(self) -> str:
        """返回检测统计信息。"""
        if self.total_detections == 0:
            return "无成功检测"
        hit_rate = self.cache_hits / self.total_detections * 100
        return f"总检测 {self.total_detections}, 缓存命中 {self.cache_hits} ({hit_rate:.1f}%)"


# 保留旧函数接口以兼容
def try_find_chessboard(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None, tuple[int, int], str]:
    """
    快速棋盘格检测（兼容接口）。
    返回 (found, corners, used_pattern_size, method_used)。
    """
    detector = ChessboardDetector(pattern_size)
    return detector.detect(gray)


# ─────────────────────────────────────────────────────────
#  核心逻辑
# ─────────────────────────────────────────────────────────

def collect_corners_from_video(
    video_path: Path,
    pattern_size: tuple[int, int],
    sample_interval: int = 10,
    show: bool = True,
    save_vis_dir: Path | None = None,
) -> tuple[list[np.ndarray], tuple[int, int], tuple[int, int]]:
    """
    从视频中按帧间隔采样，检测棋盘格角点，同时可视化。

    Returns:
        corners_list:      成功检测到角点的列表
        image_size:        (width, height)
        actual_pattern:    实际匹配到的 pattern_size
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ 无法打开视频: {video_path}")
        return [], (0, 0), pattern_size

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = (w, h)

    duration = total_frames / fps if fps > 0 else 0
    expect_samples = total_frames // max(sample_interval, 1)
    print(f"  视频信息: {w}×{h}, {fps:.1f}fps, {total_frames} 帧 ({duration:.1f}s)")
    print(f"  采样: 每 {sample_interval} 帧, 预计采样 ~{expect_samples} 帧")

    if save_vis_dir:
        save_vis_dir.mkdir(parents=True, exist_ok=True)

    win_name = f"Calibration - {video_path.name}"
    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        print("  按键说明: [Y] 接受此帧 | [N] 跳过 | [Q] 退出当前视频")

    corners_list: list[np.ndarray] = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    actual_pattern = pattern_size

    # 创建检测器实例以利用方法缓存
    detector = ChessboardDetector(pattern_size)

    frame_idx = 0
    sampled = 0
    detected = 0
    accepted = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        is_sample = (frame_idx % sample_interval == 0)

        if is_sample:
            sampled += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found, corners, orient, method = detector.detect(gray)

            vis = frame.copy()

            if found:
                actual_pattern = orient
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                detected += 1

                # 画角点
                cv2.drawChessboardCorners(vis, orient, corners_sub, True)
                status_color = (0, 255, 0)
                status_text = f"DETECTED #{detected} [{method}]"

                # 叠加状态信息和提示
                cv2.putText(vis, f"Frame {frame_idx}/{total_frames}  {status_text}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.putText(vis, f"Accepted: {accepted}  Detected: {detected}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(vis, "[Y] Accept  [N] Skip  [Q] Quit",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 保存可视化图片
                if save_vis_dir:
                    cv2.imwrite(str(save_vis_dir / f"frame_{frame_idx:06d}_detected.jpg"), vis)

                # 等待用户确认
                if show:
                    cv2.imshow(win_name, resize_for_display(vis))
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('y') or key == ord('Y'):
                            corners_list.append(corners_sub)
                            accepted += 1
                            print(f"    ✓ Frame {frame_idx} 已接受 (共 {accepted} 帧)")
                            break
                        elif key == ord('n') or key == ord('N'):
                            print(f"    ✗ Frame {frame_idx} 已跳过")
                            break
                        elif key == ord('q') or key == ord('Q'):
                            print("  用户退出当前视频")
                            cap.release()
                            cv2.destroyWindow(win_name)
                            elapsed = time.time() - t0
                            print(f"  采样 {sampled} 帧, 检测到 {detected} 帧, 接受 {accepted} 帧, 耗时 {elapsed:.1f}s")
                            return corners_list, image_size, actual_pattern
                else:
                    # 不显示窗口时自动接受
                    corners_list.append(corners_sub)
                    accepted += 1

            else:
                status_color = (0, 0, 255)
                status_text = "NOT FOUND"

                # 叠加状态信息
                cv2.putText(vis, f"Frame {frame_idx}/{total_frames}  {status_text}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.putText(vis, f"Accepted: {accepted}  Detected: {detected}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 保存可视化图片
                if save_vis_dir:
                    cv2.imwrite(str(save_vis_dir / f"frame_{frame_idx:06d}_fail.jpg"), vis)

                # 未检测到时快速跳过
                if show:
                    cv2.imshow(win_name, resize_for_display(vis))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("  用户中断 (q)")
                        break

        frame_idx += 1

    cap.release()
    if show:
        cv2.destroyWindow(win_name)

    elapsed = time.time() - t0
    print(f"  采样 {sampled} 帧, 检测到 {detected} 帧, 接受 {accepted} 帧, 耗时 {elapsed:.1f}s")
    print(f"  检测器统计: {detector.get_stats()}")

    return corners_list, image_size, actual_pattern


def calibrate_camera(
    corners_list: list[np.ndarray],
    objp: np.ndarray,
    image_size: tuple[int, int],
) -> dict | None:
    """执行相机标定，返回内参结果字典。"""
    if len(corners_list) < 3:
        print(f"  ⚠ 有效角点帧数不足 ({len(corners_list)}, 至少需要 3 帧)，跳过标定")
        return None

    obj_points = [objp] * len(corners_list)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, corners_list, image_size, None, None
    )

    # 计算重投影误差
    total_error = 0.0
    for i in range(len(obj_points)):
        img_pts, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(corners_list[i], img_pts, cv2.NORM_L2) / len(img_pts)
        total_error += error
    mean_error = total_error / len(obj_points)

    result = {
        "image_size": list(image_size),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "rms_reprojection_error": float(ret),
        "mean_reprojection_error": float(mean_error),
        "num_frames_used": len(corners_list),
    }
    return result


def print_result(name: str, result: dict) -> None:
    """打印标定结果。"""
    print(f"\n  ── {name} 标定结果 ──")
    print(f"  图像尺寸 : {result['image_size'][0]} × {result['image_size'][1]}")
    print(f"  fx       : {result['fx']:.4f}")
    print(f"  fy       : {result['fy']:.4f}")
    print(f"  cx       : {result['cx']:.4f}")
    print(f"  cy       : {result['cy']:.4f}")
    print(f"  畸变系数 : {np.array(result['dist_coeffs']).round(6).tolist()}")
    print(f"  RMS 重投影误差    : {result['rms_reprojection_error']:.6f}")
    print(f"  平均重投影误差    : {result['mean_reprojection_error']:.6f}")
    print(f"  使用帧数          : {result['num_frames_used']}")


# ─────────────────────────────────────────────────────────
#  实时相机标定
# ─────────────────────────────────────────────────────────

def collect_corners_from_live_camera(
    video_id: int,
    cam_name: str,
    pattern_size: tuple[int, int],
    width: int = 640,
    height: int = 480,
    min_frames: int = 15,
    output_dir: Path | None = None,
) -> tuple[list[np.ndarray], tuple[int, int], tuple[int, int]]:
    """
    从实时摄像头采集标定图像。

    按键:
        [SPACE] 检测并确认当前帧
        [S]     跳过当前帧
        [D]     完成采集，开始标定
        [Q]     退出

    Returns:
        corners_list, image_size, actual_pattern
    """
    cap = cv2.VideoCapture(video_id)
    if not cap.isOpened():
        print(f"  ✗ 无法打开相机 /dev/video{video_id}")
        return [], (0, 0), pattern_size

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = (w, h)

    win_name = f"Live Calibration - {cam_name} (video{video_id})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print(f"\n  实时标定: {cam_name} -> /dev/video{video_id} ({w}×{h})")
    print(f"  按键: [SPACE] 捕获 | [D] 完成 | [Q] 退出")
    print(f"  目标: 至少 {min_frames} 帧，多角度拍摄棋盘格")

    corners_list: list[np.ndarray] = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    actual_pattern = pattern_size
    accepted = 0

    # 创建检测器实例以利用方法缓存
    detector = ChessboardDetector(pattern_size)

    if output_dir:
        vis_dir = output_dir / "vis" / cam_name
        vis_dir.mkdir(parents=True, exist_ok=True)
    else:
        vis_dir = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners, orient, method = detector.detect(gray)

        vis = frame.copy()

        if found:
            actual_pattern = orient
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis, orient, corners_sub, True)
            cv2.putText(vis, f"DETECTED [{method}] - Press SPACE to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "Searching for chessboard...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(vis, f"Captured: {accepted}/{min_frames}  [SPACE]=Capture [D]=Done [Q]=Quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow(win_name, vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):  # SPACE - 捕获
            if found:
                corners_list.append(corners_sub)
                accepted += 1
                print(f"    ✓ 捕获第 {accepted} 帧")
                if vis_dir:
                    cv2.imwrite(str(vis_dir / f"capture_{accepted:03d}.jpg"), vis)
            else:
                print(f"    ✗ 未检测到棋盘格，无法捕获")

        elif key == ord('d') or key == ord('D'):  # D - 完成
            if accepted >= 3:
                print(f"  完成采集，共 {accepted} 帧")
                break
            else:
                print(f"    ⚠ 至少需要 3 帧，当前 {accepted} 帧")

        elif key == ord('q') or key == ord('Q'):  # Q - 退出
            print(f"  用户退出")
            break

    cap.release()
    cv2.destroyWindow(win_name)
    print(f"  检测器统计: {detector.get_stats()}")

    return corners_list, image_size, actual_pattern


def process_live_camera(
    video_id: int,
    cam_name: str,
    pattern_size: tuple[int, int],
    square_size: float,
    width: int,
    height: int,
    output_dir: Path,
) -> dict | None:
    """处理单个实时相机，返回标定结果。"""
    print(f"\n{'='*60}")
    print(f"处理相机: {cam_name} -> /dev/video{video_id}")
    print(f"{'='*60}")

    corners_list, image_size, actual_pattern = collect_corners_from_live_camera(
        video_id, cam_name, pattern_size, width, height, output_dir=output_dir
    )

    if not corners_list:
        print(f"  ✗ {cam_name}: 未采集到任何有效帧")
        return None

    if actual_pattern != pattern_size:
        print(f"  ℹ 使用方向 {actual_pattern[0]}×{actual_pattern[1]}")
    objp = build_object_points(actual_pattern[0], actual_pattern[1], square_size)

    result = calibrate_camera(corners_list, objp, image_size)
    if result is None:
        return None

    result["video_id"] = video_id
    result["pattern_used"] = f"{actual_pattern[0]}x{actual_pattern[1]}"
    print_result(cam_name, result)

    return result


def run_live_calibration(
    pattern_size: tuple[int, int],
    square_size: float,
    output_dir: Path,
) -> None:
    """从 config.yaml 读取 index_map，逐个相机进行实时标定。"""
    config = load_yaml(CONFIG_PATH)
    cam_config = config.get("camera", {})

    index_map = cam_config.get("index_map", {})
    width = cam_config.get("width", 640)
    height = cam_config.get("height", 480)

    if not index_map:
        print("✗ config.yaml 中未找到 camera.index_map")
        print("  请先运行标定脚本生成 index_map，或手动配置")
        sys.exit(1)

    # index_map: cam_id -> video_id
    cameras = [(int(cam_id), int(video_id)) for cam_id, video_id in index_map.items()]
    cameras.sort(key=lambda x: x[0])

    print(f"\n从 config.yaml 加载相机配置:")
    print(f"  分辨率: {width}×{height}")
    print(f"  相机映射:")
    for cam_id, video_id in cameras:
        print(f"    cam{cam_id} -> /dev/video{video_id}")

    print(f"\n棋盘格: 内角点 {pattern_size[0]}×{pattern_size[1]}, 方块 {square_size}mm")
    print(f"输出目录: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for cam_id, video_id in cameras:
        cam_name = f"cam{cam_id}"
        result = process_live_camera(
            video_id, cam_name, pattern_size, square_size, width, height, output_dir
        )
        if result is not None:
            all_results[cam_name] = result

    if all_results:
        save_results(all_results, output_dir)
        print(f"\n{'='*60}")
        print(f"实时标定完成! 共成功标定 {len(all_results)}/{len(cameras)} 个相机")
        print(f"结果保存在: {output_dir}")
        print(f"{'='*60}")
    else:
        print("\n✗ 所有相机均未能成功标定")
        sys.exit(1)


def save_undistort_sample(
    video_path: Path,
    result: dict,
    output_dir: Path,
    name: str,
) -> None:
    """从视频中取一帧，保存原图和去畸变对比图。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return

    K = np.array(result["camera_matrix"], dtype=np.float64)
    D = np.array(result["dist_coeffs"], dtype=np.float64)
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, K, D, None, new_K)

    # 左右拼接对比
    compare = np.hstack([frame, undistorted])
    cv2.putText(compare, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(compare, "Undistorted", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out_path = output_dir / f"{name}_undistort_compare.jpg"
    cv2.imwrite(str(out_path), compare)
    print(f"  ✓ 去畸变对比图: {out_path}")


def process_video(
    video_path: Path,
    pattern_size: tuple[int, int],
    square_size: float,
    sample_interval: int,
    show: bool,
    output_dir: Path,
    name: str,
) -> dict | None:
    """处理单个视频文件，返回标定结果。"""
    print(f"\n{'='*60}")
    print(f"处理视频: {video_path.name}  (路径: {video_path})")
    print(f"{'='*60}")

    vis_dir = output_dir / "vis" / name
    corners_list, image_size, actual_pattern = collect_corners_from_video(
        video_path, pattern_size, sample_interval,
        show=show,
        save_vis_dir=vis_dir,
    )

    if not corners_list:
        print(f"  ✗ {name}: 未检测到任何棋盘格角点，跳过")
        print(f"     检查事项:")
        print(f"       - 视频中是否有清晰的 {pattern_size[0]}×{pattern_size[1]} 棋盘格？")
        print(f"       - 棋盘格是否完整出现在画面中（不能被裁剪）？")
        print(f"       - 可调低 --sample-interval 以增加采样帧数")
        print(f"     检查采样帧: {vis_dir}/ (*_fail.jpg)")
        return None

    # 根据实际匹配上的方向构建 objp
    if actual_pattern != pattern_size:
        print(f"  ℹ 使用方向 {actual_pattern[0]}×{actual_pattern[1]} 匹配成功")
    objp = build_object_points(actual_pattern[0], actual_pattern[1], square_size)

    result = calibrate_camera(corners_list, objp, image_size)
    if result is None:
        return None

    result["video_file"] = str(video_path)
    result["pattern_used"] = f"{actual_pattern[0]}x{actual_pattern[1]}"
    print_result(name, result)

    # 保存去畸变对比
    save_undistort_sample(video_path, result, output_dir, name)

    return result


def save_results(results: dict[str, dict], output_dir: Path) -> None:
    """将所有标定结果保存为 JSON 和 OpenCV YAML。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 汇总 JSON
    summary_path = output_dir / "intrinsics_all.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 汇总结果已保存: {summary_path}")

    # 2. 每个相机单独 OpenCV YAML
    for name, res in results.items():
        yaml_path = output_dir / f"{name}_intrinsics.yaml"
        fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_WRITE)
        fs.write("image_width", res["image_size"][0])
        fs.write("image_height", res["image_size"][1])
        fs.write("camera_matrix", np.array(res["camera_matrix"], dtype=np.float64))
        fs.write("dist_coeffs", np.array(res["dist_coeffs"], dtype=np.float64))
        fs.write("rms_reprojection_error", res["rms_reprojection_error"])
        fs.write("num_frames_used", res["num_frames_used"])
        fs.release()
        print(f"  ✓ {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="相机内参标定（支持录制视频或实时相机）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_path", type=str, nargs="?",
        default=str(Path.home() / "recordings"),
        help="录制目录路径（可以是 ~/recordings 或某次录制子目录）",
    )
    parser.add_argument("--live", action="store_true",
                        help="实时模式: 从摄像头采集图像（按 config.yaml 的 index_map）")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS,
                        help=f"棋盘格内角点列数 (默认 {DEFAULT_COLS})")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help=f"棋盘格内角点行数 (默认 {DEFAULT_ROWS})")
    parser.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE,
                        help=f"方块边长 mm (默认 {DEFAULT_SQUARE_SIZE})")
    parser.add_argument("--sample-interval", type=int, default=20,
                        help="每隔多少帧采样一次 (默认 20, 仅视频模式)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: calib_intrinsics/)")
    parser.add_argument("--no-show", action="store_true",
                        help="不弹出实时可视化窗口（仅视频模式）")

    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)

    # === 实时模式 ===
    if args.live:
        output_dir = Path(args.output_dir) if args.output_dir else Path("calib_intrinsics")
        run_live_calibration(pattern_size, args.square_size, output_dir)
        return

    # === 视频模式 ===
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"✗ 路径不存在: {input_path}")
        sys.exit(1)

    # 查找所有 mp4 文件
    video_files = sorted(input_path.rglob("*.mp4")) + sorted(input_path.rglob("*.MP4"))
    seen = set()
    unique_videos = []
    for v in video_files:
        key = str(v).lower()
        if key not in seen:
            seen.add(key)
            unique_videos.append(v)
    video_files = sorted(unique_videos)

    if not video_files:
        print(f"✗ 在 {input_path} 下未找到任何 .mp4 文件")
        sys.exit(1)

    print(f"找到 {len(video_files)} 个视频文件:")
    for vf in video_files:
        print(f"  {vf}")

    # 参数
    output_dir = Path(args.output_dir) if args.output_dir else input_path / "calib_intrinsics"
    output_dir.mkdir(parents=True, exist_ok=True)
    show = not args.no_show

    print(f"\n棋盘格参数: 内角点 {args.cols}×{args.rows}, 方块边长 {args.square_size} mm")
    print(f"采样间隔  : 每 {args.sample_interval} 帧")
    print(f"可视化窗口: {'开' if show else '关（图片仍保存到 vis/）'}")
    print(f"输出目录  : {output_dir}")

    # 逐个视频标定
    all_results: dict[str, dict] = {}
    for video_path in video_files:
        rel = video_path.relative_to(input_path)
        if len(rel.parts) > 1:
            key = f"{rel.parts[-2]}_{rel.stem}"
        else:
            key = rel.stem

        result = process_video(
            video_path, pattern_size, args.square_size,
            args.sample_interval, show, output_dir, key,
        )
        if result is not None:
            all_results[key] = result

    # 保存
    if all_results:
        save_results(all_results, output_dir)
        print(f"\n{'='*60}")
        print(f"标定完成! 共成功标定 {len(all_results)}/{len(video_files)} 个视频")
        print(f"结果保存在: {output_dir}")
        print(f"\n输出内容:")
        print(f"  intrinsics_all.json          ── 所有相机汇总")
        print(f"  <name>_intrinsics.yaml       ── 每个相机 OpenCV 格式")
        print(f"  <name>_undistort_compare.jpg  ── 去畸变前后对比")
        print(f"  vis/<name>/                   ── 每帧检测可视化 (ok/fail)")
        print(f"{'='*60}")
    else:
        print("\n✗ 所有视频均未能成功标定")
        print("  可能的原因:")
        print("    1. 棋盘格内角点数不对 (当前 --cols/--rows)")
        print("    2. 视频画面中棋盘格不完整或太模糊")
        print("    3. 尝试降低 --sample-interval 增加采样")
        print(f"\n  检查 {output_dir}/vis/ 下的 *_fail.jpg 查看未检测到的帧")
        sys.exit(1)


if __name__ == "__main__":
    main()
