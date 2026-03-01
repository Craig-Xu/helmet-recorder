#!/usr/bin/env python3
"""
录制数据回放工具 - GUI 版本
可视化界面选择录制文件夹，同步播放多相机视频 + IMU 可视化
"""

import cv2
import numpy as np
import time
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import threading
from PIL import Image, ImageTk


# ============== 数据结构 ==============

@dataclass
class IMUFrame:
    """IMU 数据帧"""
    timestamp: float
    roll: float
    pitch: float
    yaw: float
    q0: float
    q1: float
    q2: float
    q3: float
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


@dataclass
class RecordingInfo:
    """录制信息"""
    path: Path
    name: str
    date_str: str
    video_count: int
    has_imu: bool
    duration_sec: float = 0.0
    frame_count: int = 0


# ============== 工具函数 ==============

def load_imu_data(filepath: Path) -> list[IMUFrame]:
    """加载 IMU 数据文件"""
    frames = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                parts = line.split(',')
                if len(parts) >= 14:
                    frame = IMUFrame(
                        timestamp=float(parts[0]),
                        roll=float(parts[1]),
                        pitch=float(parts[2]),
                        yaw=float(parts[3]),
                        q0=float(parts[4]),
                        q1=float(parts[5]),
                        q2=float(parts[6]),
                        q3=float(parts[7]),
                        acc_x=float(parts[8]),
                        acc_y=float(parts[9]),
                        acc_z=float(parts[10]),
                        gyro_x=float(parts[11]),
                        gyro_y=float(parts[12]),
                        gyro_z=float(parts[13]),
                    )
                    frames.append(frame)
            except (ValueError, IndexError):
                continue
    return frames


def quaternion_to_rotation_matrix(q0, q1, q2, q3):
    """四元数转旋转矩阵"""
    norm = np.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
    if norm < 1e-10:
        return np.eye(3)
    q0, q1, q2, q3 = q0/norm, q1/norm, q2/norm, q3/norm
    R = np.array([
        [1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2)]
    ])
    return R


def scan_recordings(base_dir: Path) -> list[RecordingInfo]:
    """扫描录制目录"""
    recordings = []
    if not base_dir.exists():
        return recordings
    
    for folder in sorted(base_dir.iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        if not folder.name.startswith('recording_'):
            continue
        
        video_files = list(folder.glob('*.mp4'))
        imu_file = folder / 'imu_data.txt'
        
        if not video_files:
            continue
        
        # 解析日期
        try:
            date_part = folder.name.replace('recording_', '')
            dt = datetime.strptime(date_part, '%Y%m%d_%H%M%S')
            date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            date_str = folder.name
        
        # 获取视频信息
        duration = 0.0
        frame_count = 0
        try:
            cap = cv2.VideoCapture(str(video_files[0]))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                duration = frame_count / fps
                cap.release()
        except:
            pass
        
        recordings.append(RecordingInfo(
            path=folder,
            name=folder.name,
            date_str=date_str,
            video_count=len(video_files),
            has_imu=imu_file.exists(),
            duration_sec=duration,
            frame_count=frame_count
        ))
    
    return recordings


# ============== 视频播放器 ==============

class VideoPlayer:
    """高性能多视频播放器"""
    
    def __init__(self, video_paths: list[Path], fps: float = 30.0, grid_width: int = 800):
        self.caps = []
        self.video_names = []
        
        for path in video_paths:
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                self.caps.append(cap)
                self.video_names.append(path.stem)
        
        self.num_videos = len(self.caps)
        self.fps = fps
        self.current_frame = 0
        self.total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in self.caps) if self.caps else 0
        
        # 计算网格布局
        if self.num_videos <= 2:
            self.cols = self.num_videos
        elif self.num_videos <= 4:
            self.cols = 2
        elif self.num_videos <= 9:
            self.cols = 3
        else:
            self.cols = 4
        self.rows = (self.num_videos + self.cols - 1) // self.cols
        
        # 计算单个视频尺寸
        if self.caps:
            orig_w = int(self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.thumb_w = grid_width // self.cols
            self.thumb_h = int(self.thumb_w * orig_h / orig_w)
        else:
            self.thumb_w, self.thumb_h = 320, 240
        
        # 预分配帧缓冲
        self.frame_buffer = np.zeros(
            (self.rows * self.thumb_h, self.cols * self.thumb_w, 3),
            dtype=np.uint8
        )
    
    def get_current_time(self) -> float:
        return self.current_frame / self.fps
    
    def read_combined_frame(self) -> np.ndarray | None:
        """读取所有视频帧并合并成网格"""
        self.frame_buffer.fill(0)
        
        for i, cap in enumerate(self.caps):
            ret, frame = cap.read()
            if ret:
                row = i // self.cols
                col = i % self.cols
                y1 = row * self.thumb_h
                x1 = col * self.thumb_w
                
                small = cv2.resize(frame, (self.thumb_w, self.thumb_h), interpolation=cv2.INTER_NEAREST)
                cv2.putText(small, f"Cam {self.video_names[i]}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                self.frame_buffer[y1:y1+self.thumb_h, x1:x1+self.thumb_w] = small
        
        self.current_frame += 1
        return self.frame_buffer
    
    def seek(self, frame_num: int):
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.current_frame = frame_num
    
    def release(self):
        for cap in self.caps:
            cap.release()


# ============== IMU 可视化 ==============

class IMUVisualizer:
    """IMU 可视化 - OpenCV 渲染"""
    
    def __init__(self, imu_data: list[IMUFrame], width: int = 600, height: int = 450):
        self.imu_data = imu_data
        self.width = width
        self.height = height
        
        # 计算 IMU 数据的起始时间偏移（用于与视频时间同步）
        self.start_timestamp = imu_data[0].timestamp if imu_data else 0.0
        
        self.history_size = 200
        self.time_history = deque(maxlen=self.history_size)
        self.roll_history = deque(maxlen=self.history_size)
        self.pitch_history = deque(maxlen=self.history_size)
        self.yaw_history = deque(maxlen=self.history_size)
        self.acc_x_history = deque(maxlen=self.history_size)
        self.acc_y_history = deque(maxlen=self.history_size)
        self.acc_z_history = deque(maxlen=self.history_size)
        self.gyro_x_history = deque(maxlen=self.history_size)
        self.gyro_y_history = deque(maxlen=self.history_size)
        self.gyro_z_history = deque(maxlen=self.history_size)
        
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.plot_w = width // 2
        self.plot_h = height // 2
    
    def _find_frame_at_time(self, video_time: float) -> int:
        """根据视频时间查找对应的 IMU 帧索引
        
        video_time: 视频播放时间（从 0 开始）
        返回对应的 IMU 帧索引
        """
        if not self.imu_data:
            return 0
        
        # 将视频时间转换为 IMU 时间戳
        target_time = self.start_timestamp + video_time
        
        left, right = 0, len(self.imu_data) - 1
        while left < right:
            mid = (left + right) // 2
            if self.imu_data[mid].timestamp < target_time:
                left = mid + 1
            else:
                right = mid
        return left
    
    def _draw_plot(self, canvas, x, y, w, h, title: str, 
                   data_list: list[tuple[deque, tuple]], 
                   y_min: float, y_max: float, times: list):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 1)
        cv2.putText(canvas, title, (x + 10, y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if not times or len(times) < 2:
            return
        
        margin = 25
        px, py = x + margin, y + margin
        pw, ph = w - margin * 2, h - margin * 2
        
        t_min, t_max = times[0], times[-1]
        if t_max <= t_min:
            t_max = t_min + 1
        
        for i in range(5):
            gy = py + int(ph * i / 4)
            cv2.line(canvas, (px, gy), (px + pw, gy), (50, 50, 50), 1)
        
        for data, color in data_list:
            if len(data) < 2:
                continue
            points = []
            for t, v in zip(times, data):
                px_ = px + int((t - t_min) / (t_max - t_min) * pw)
                py_ = py + int((1 - (v - y_min) / (y_max - y_min)) * ph)
                py_ = max(py, min(py + ph, py_))
                points.append((px_, py_))
            
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1], color, 1, cv2.LINE_AA)
    
    def _draw_3d_box(self, canvas, x, y, w, h, R):
        """绘制3D姿态可视化 - 北西天(NWU)坐标系
        
        世界坐标系：X=北(前), Y=西(左), Z=天(上)
        视角：从东南上方45度俯视
        """
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 1)
        cv2.putText(canvas, "3D Orientation (NWU)", (x + 10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cx, cy = x + w // 2, y + h // 2 + 15  # 稍微下移，给标签留空间
        scale = min(w, h) // 4
        
        # 等距投影变换矩阵 - 从东南上方45度俯视
        # 这样可以同时看到北(X)、西(Y)、天(Z)三个方向
        angle_h = np.radians(30)  # 水平旋转角度
        angle_v = np.radians(25)  # 俯视角度
        
        def isometric_project(p3d):
            """等距投影：NWU坐标系 -> 屏幕坐标"""
            x3d, y3d, z3d = p3d
            # 旋转变换：先绕Z轴旋转（水平视角），再绕X轴旋转（俯视）
            # 屏幕X = -Y*cos + X*sin（西向右，北向右上）
            # 屏幕Y = -Z*cos_v + (X*cos + Y*sin)*sin_v（天向上）
            screen_x = -y3d * np.cos(angle_h) + x3d * np.sin(angle_h)
            screen_y = -z3d * np.cos(angle_v) - (x3d * np.cos(angle_h) + y3d * np.sin(angle_h)) * np.sin(angle_v)
            return (int(cx + screen_x * scale), int(cy + screen_y * scale))
        
        # 物体局部坐标系的盒子 (模拟传感器/头盔形状)
        sx, sy, sz = 1.5, 1.0, 0.4  # 长宽高比例
        vertices = np.array([
            [-sx/2, -sy/2, -sz/2], [sx/2, -sy/2, -sz/2],
            [sx/2, sy/2, -sz/2], [-sx/2, sy/2, -sz/2],
            [-sx/2, -sy/2, sz/2], [sx/2, -sy/2, sz/2],
            [sx/2, sy/2, sz/2], [-sx/2, sy/2, sz/2],
        ])
        
        # 应用旋转矩阵（R的列是局部坐标轴在全局坐标系中的表示）
        rotated = (R @ vertices.T).T
        projected = [isometric_project(v) for v in rotated]
        
        # 绘制盒子边（根据深度着色）
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i, j in edges:
            # 计算边的平均深度用于着色
            avg_depth = (rotated[i][0] * np.cos(angle_h) + rotated[i][1] * np.sin(angle_h) + 
                        rotated[j][0] * np.cos(angle_h) + rotated[j][1] * np.sin(angle_h)) / 2
            brightness = int(150 + avg_depth * 40)
            brightness = max(80, min(255, brightness))
            color = (brightness, brightness, brightness)
            cv2.line(canvas, projected[i], projected[j], color, 2, cv2.LINE_AA)
        
        # 绘制前面标记（红色箭头指向物体前方）
        front_center = R @ np.array([sx/2 + 0.2, 0, 0])
        front_p = isometric_project(front_center)
        front_base = R @ np.array([sx/2, 0, 0])
        front_base_p = isometric_project(front_base)
        cv2.arrowedLine(canvas, front_base_p, front_p, (0, 100, 255), 2, cv2.LINE_AA, tipLength=0.5)
        
        # 绘制世界坐标系参考轴（固定不动）
        origin = np.array([0, 0, 0])
        axis_length = 1.3
        world_axes = [
            (np.array([axis_length, 0, 0]), (0, 0, 255), "N"),   # X=北，红色
            (np.array([0, axis_length, 0]), (0, 255, 0), "W"),   # Y=西，绿色
            (np.array([0, 0, axis_length]), (255, 100, 0), "U"), # Z=天，蓝色
        ]
        
        p_origin = isometric_project(origin)
        for axis_vec, color, label in world_axes:
            p_end = isometric_project(axis_vec)
            # 绘制虚线表示世界坐标系
            cv2.line(canvas, p_origin, p_end, color, 1, cv2.LINE_AA)
            cv2.putText(canvas, label, (p_end[0] + 3, p_end[1] + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        # 绘制物体坐标系轴（R的列是局部轴在全局坐标系中的方向）
        body_axis_length = 1.0
        body_axes = [
            (R[:, 0] * body_axis_length, (100, 100, 255), "x"),  # 物体X轴
            (R[:, 1] * body_axis_length, (100, 255, 100), "y"),  # 物体Y轴
            (R[:, 2] * body_axis_length, (255, 150, 100), "z"),  # 物体Z轴
        ]
        
        for axis_vec, color, label in body_axes:
            p_end = isometric_project(axis_vec)
            cv2.arrowedLine(canvas, p_origin, p_end, color, 2, cv2.LINE_AA, tipLength=0.15)
    
    def render(self, video_time: float) -> np.ndarray:
        idx = self._find_frame_at_time(video_time)
        if idx >= len(self.imu_data):
            idx = len(self.imu_data) - 1
        if idx < 0 or not self.imu_data:
            self.canvas.fill(0)
            return self.canvas
        
        frame = self.imu_data[idx]
        
        if not self.time_history or frame.timestamp > self.time_history[-1]:
            self.time_history.append(frame.timestamp)
            self.roll_history.append(frame.roll)
            self.pitch_history.append(frame.pitch)
            self.yaw_history.append(frame.yaw)
            self.acc_x_history.append(frame.acc_x)
            self.acc_y_history.append(frame.acc_y)
            self.acc_z_history.append(frame.acc_z)
            self.gyro_x_history.append(frame.gyro_x)
            self.gyro_y_history.append(frame.gyro_y)
            self.gyro_z_history.append(frame.gyro_z)
        
        self.canvas.fill(0)
        times = list(self.time_history)
        
        R = quaternion_to_rotation_matrix(frame.q0, frame.q1, frame.q2, frame.q3)
        self._draw_3d_box(self.canvas, 0, 0, self.plot_w, self.plot_h, R)
        
        self._draw_plot(
            self.canvas, self.plot_w, 0, self.plot_w, self.plot_h,
            f"Euler R:{frame.roll:.0f} P:{frame.pitch:.0f} Y:{frame.yaw:.0f}",
            [(self.roll_history, (0, 0, 255)), (self.pitch_history, (0, 255, 0)), 
             (self.yaw_history, (255, 0, 0))],
            -180, 180, times
        )
        
        self._draw_plot(
            self.canvas, 0, self.plot_h, self.plot_w, self.plot_h,
            f"Accel X:{frame.acc_x:.1f} Y:{frame.acc_y:.1f} Z:{frame.acc_z:.1f}",
            [(self.acc_x_history, (0, 0, 255)), (self.acc_y_history, (0, 255, 0)),
             (self.acc_z_history, (255, 0, 0))],
            -3, 3, times
        )
        
        self._draw_plot(
            self.canvas, self.plot_w, self.plot_h, self.plot_w, self.plot_h,
            f"Gyro X:{frame.gyro_x:.0f} Y:{frame.gyro_y:.0f} Z:{frame.gyro_z:.0f}",
            [(self.gyro_x_history, (0, 0, 255)), (self.gyro_y_history, (0, 255, 0)),
             (self.gyro_z_history, (255, 0, 0))],
            -500, 500, times
        )
        
        return self.canvas
    
    def reset(self):
        self.time_history.clear()
        self.roll_history.clear()
        self.pitch_history.clear()
        self.yaw_history.clear()
        self.acc_x_history.clear()
        self.acc_y_history.clear()
        self.acc_z_history.clear()
        self.gyro_x_history.clear()
        self.gyro_y_history.clear()
        self.gyro_z_history.clear()


# ============== 可交互 3D 姿态窗口 ==============

class Interactive3DWindow:
    """可交互的3D姿态可视化窗口
    
    支持：
    - 鼠标左键拖拽旋转视角
    - 滚轮缩放
    - R键重置视角
    - ESC关闭窗口
    """
    
    def __init__(self, window_name: str = "3D Attitude (Interactive)"):
        self.window_name = window_name
        self.width = 600
        self.height = 500
        
        # 视角参数（可交互调整）
        self.angle_h = 30.0   # 水平旋转角度（度）
        self.angle_v = 25.0   # 俯视角度（度）
        self.scale = 80       # 缩放比例
        
        # 鼠标状态
        self.dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # 当前姿态（四元数）
        self.q0, self.q1, self.q2, self.q3 = 1.0, 0.0, 0.0, 0.0
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0
        
        # 窗口状态
        self.is_open = False
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def open(self):
        """打开窗口"""
        if self.is_open:
            return
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self.is_open = True
    
    def close(self):
        """关闭窗口"""
        if self.is_open:
            cv2.destroyWindow(self.window_name)
            self.is_open = False
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse_x = x
            self.last_mouse_y = y
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                dx = x - self.last_mouse_x
                dy = y - self.last_mouse_y
                self.angle_h += dx * 0.5
                self.angle_v += dy * 0.5
                # 限制俯仰角度
                self.angle_v = max(-85, min(85, self.angle_v))
                self.last_mouse_x = x
                self.last_mouse_y = y
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # 滚轮缩放
            if flags > 0:
                self.scale = min(200, self.scale + 5)
            else:
                self.scale = max(30, self.scale - 5)
    
    def reset_view(self):
        """重置视角"""
        self.angle_h = 30.0
        self.angle_v = 25.0
        self.scale = 80
    
    def update_attitude(self, q0, q1, q2, q3, roll=0.0, pitch=0.0, yaw=0.0):
        """更新姿态数据"""
        self.q0, self.q1, self.q2, self.q3 = q0, q1, q2, q3
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
    
    def _isometric_project(self, p3d, cx, cy):
        """等距投影"""
        x3d, y3d, z3d = p3d
        angle_h_rad = np.radians(self.angle_h)
        angle_v_rad = np.radians(self.angle_v)
        
        screen_x = -y3d * np.cos(angle_h_rad) + x3d * np.sin(angle_h_rad)
        screen_y = -z3d * np.cos(angle_v_rad) - (x3d * np.cos(angle_h_rad) + y3d * np.sin(angle_h_rad)) * np.sin(angle_v_rad)
        return (int(cx + screen_x * self.scale), int(cy + screen_y * self.scale))
    
    def render(self) -> np.ndarray:
        """渲染3D姿态"""
        self.canvas.fill(20)
        
        cx, cy = self.width // 2, self.height // 2 + 20
        
        # 计算旋转矩阵
        R = quaternion_to_rotation_matrix(self.q0, self.q1, self.q2, self.q3)
        
        # 绘制网格地面（XY平面）
        grid_size = 2.0
        grid_step = 0.5
        for i in np.arange(-grid_size, grid_size + 0.1, grid_step):
            # X方向线
            p1 = self._isometric_project([i, -grid_size, 0], cx, cy)
            p2 = self._isometric_project([i, grid_size, 0], cx, cy)
            cv2.line(self.canvas, p1, p2, (40, 40, 40), 1)
            # Y方向线
            p1 = self._isometric_project([-grid_size, i, 0], cx, cy)
            p2 = self._isometric_project([grid_size, i, 0], cx, cy)
            cv2.line(self.canvas, p1, p2, (40, 40, 40), 1)
        
        # 绘制世界坐标系轴
        origin = np.array([0, 0, 0])
        axis_length = 1.8
        world_axes = [
            (np.array([axis_length, 0, 0]), (0, 0, 255), "N (X)"),
            (np.array([0, axis_length, 0]), (0, 255, 0), "W (Y)"),
            (np.array([0, 0, axis_length]), (255, 150, 0), "U (Z)"),
        ]
        
        p_origin = self._isometric_project(origin, cx, cy)
        for axis_vec, color, label in world_axes:
            p_end = self._isometric_project(axis_vec, cx, cy)
            cv2.arrowedLine(self.canvas, p_origin, p_end, color, 2, cv2.LINE_AA, tipLength=0.08)
            cv2.putText(self.canvas, label, (p_end[0] + 5, p_end[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # 绘制物体盒子
        sx, sy, sz = 1.2, 0.8, 0.35
        vertices = np.array([
            [-sx/2, -sy/2, -sz/2], [sx/2, -sy/2, -sz/2],
            [sx/2, sy/2, -sz/2], [-sx/2, sy/2, -sz/2],
            [-sx/2, -sy/2, sz/2], [sx/2, -sy/2, sz/2],
            [sx/2, sy/2, sz/2], [-sx/2, sy/2, sz/2],
        ])
        
        # 应用旋转矩阵（R的列是局部坐标轴在全局坐标系中的表示）
        rotated = (R @ vertices.T).T
        projected = [self._isometric_project(v, cx, cy) for v in rotated]
        
        # 计算每个面的平均深度，用于排序
        faces = [
            ([0, 1, 2, 3], (80, 80, 100)),    # 底面
            ([4, 5, 6, 7], (120, 120, 150)),  # 顶面
            ([0, 1, 5, 4], (100, 100, 120)),  # 前面
            ([2, 3, 7, 6], (90, 90, 110)),    # 后面
            ([0, 3, 7, 4], (95, 95, 115)),    # 左面
            ([1, 2, 6, 5], (105, 105, 125)),  # 右面
        ]
        
        # 按深度排序面（从远到近绘制）
        def face_depth(face):
            indices = face[0]
            return sum(rotated[i][0] * np.cos(np.radians(self.angle_h)) + 
                      rotated[i][1] * np.sin(np.radians(self.angle_h)) for i in indices)
        
        faces_sorted = sorted(faces, key=face_depth)
        
        for indices, base_color in faces_sorted:
            pts = np.array([projected[i] for i in indices], np.int32)
            # 根据深度调整颜色
            depth = face_depth((indices, base_color)) / 4
            brightness = max(0.5, min(1.5, 1.0 + depth * 0.15))
            color = tuple(int(c * brightness) for c in base_color)
            cv2.fillPoly(self.canvas, [pts], color)
            cv2.polylines(self.canvas, [pts], True, (180, 180, 200), 2, cv2.LINE_AA)
        
        # 绘制前方标记（橙色箭头）
        front_base = R @ np.array([sx/2, 0, 0])
        front_tip = R @ np.array([sx/2 + 0.35, 0, 0])
        p_base = self._isometric_project(front_base, cx, cy)
        p_tip = self._isometric_project(front_tip, cx, cy)
        cv2.arrowedLine(self.canvas, p_base, p_tip, (0, 140, 255), 3, cv2.LINE_AA, tipLength=0.4)
        
        # 绘制物体坐标轴（R的列是局部轴在全局坐标系中的方向）
        body_axis_length = 0.9
        body_axes = [
            (R[:, 0] * body_axis_length, (150, 150, 255), "x"),
            (R[:, 1] * body_axis_length, (150, 255, 150), "y"),
            (R[:, 2] * body_axis_length, (255, 180, 150), "z"),
        ]
        
        for axis_vec, color, label in body_axes:
            p_end = self._isometric_project(axis_vec, cx, cy)
            cv2.arrowedLine(self.canvas, p_origin, p_end, color, 2, cv2.LINE_AA, tipLength=0.12)
        
        # 绘制信息面板
        info_y = 30
        cv2.putText(self.canvas, "3D Attitude Viewer (NWU)", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += 25
        cv2.putText(self.canvas, f"Roll: {self.roll:7.2f}  Pitch: {self.pitch:7.2f}  Yaw: {self.yaw:7.2f}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        info_y += 22
        cv2.putText(self.canvas, f"View: H={self.angle_h:.0f}  V={self.angle_v:.0f}  Scale={self.scale}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        # 操作提示
        cv2.putText(self.canvas, "Drag: Rotate | Scroll: Zoom | R: Reset | ESC: Close",
                    (10, self.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        return self.canvas
    
    def show(self) -> int:
        """显示并返回按键"""
        if not self.is_open:
            self.open()
        
        frame = self.render()
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r') or key == ord('R'):
            self.reset_view()
        
        return key


# ============== GUI 应用 ==============

class PlaybackGUI:
    """回放工具 GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("录制回放工具")
        self.root.geometry("1400x800")
        
        # 默认录制目录
        self.recordings_dir = Path.home() / "recordings"
        self.recordings: list[RecordingInfo] = []
        self.selected_recording: RecordingInfo | None = None
        
        # 播放状态
        self.player: VideoPlayer | None = None
        self.imu_viz: IMUVisualizer | None = None
        self.imu_data: list[IMUFrame] = []
        self.is_playing = False
        self.speed = 1.0
        self.playback_thread: threading.Thread | None = None
        self.stop_flag = False
        
        # 可交互3D姿态窗口
        self.interactive_3d: Interactive3DWindow | None = None
        self.current_imu_frame: IMUFrame | None = None
        
        self._setup_ui()
        self._scan_recordings()
    
    def _setup_ui(self):
        # 左侧：录制列表
        left_frame = ttk.Frame(self.root, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)
        
        # 目录选择
        dir_frame = ttk.Frame(left_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dir_frame, text="录制目录:").pack(side=tk.LEFT)
        self.dir_var = tk.StringVar(value=str(self.recordings_dir))
        dir_entry = ttk.Entry(dir_frame, textvariable=self.dir_var, width=25)
        dir_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_frame, text="浏览", command=self._browse_dir, width=6).pack(side=tk.LEFT)
        
        # 刷新按钮
        ttk.Button(left_frame, text="刷新列表", command=self._scan_recordings).pack(fill=tk.X, pady=5)
        
        # 录制列表
        ttk.Label(left_frame, text="录制列表:", font=('', 11, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.recording_list = ttk.Treeview(list_frame, columns=('date', 'videos', 'duration'), 
                                           show='headings', height=20)
        self.recording_list.heading('date', text='日期时间')
        self.recording_list.heading('videos', text='视频')
        self.recording_list.heading('duration', text='时长')
        self.recording_list.column('date', width=140)
        self.recording_list.column('videos', width=50)
        self.recording_list.column('duration', width=60)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.recording_list.yview)
        self.recording_list.configure(yscrollcommand=scrollbar.set)
        
        self.recording_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.recording_list.bind('<<TreeviewSelect>>', self._on_recording_select)
        self.recording_list.bind('<Double-1>', self._on_double_click)
        
        # 详情
        detail_frame = ttk.LabelFrame(left_frame, text="详情")
        detail_frame.pack(fill=tk.X, pady=10)
        
        self.detail_label = ttk.Label(detail_frame, text="请选择一个录制", wraplength=300)
        self.detail_label.pack(padx=10, pady=10)
        
        # 播放按钮
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.play_btn = ttk.Button(btn_frame, text="播放", command=self._start_playback, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self._stop_playback, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        # 3D视图按钮
        self.view_3d_btn = ttk.Button(btn_frame, text="3D姿态", command=self._toggle_3d_view, state=tk.DISABLED)
        self.view_3d_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        # 右侧：预览区
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 视频预览
        video_frame = ttk.LabelFrame(right_frame, text="视频预览")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.video_canvas = tk.Canvas(video_frame, bg='black', width=800, height=400)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # IMU 预览
        imu_frame = ttk.LabelFrame(right_frame, text="IMU 数据")
        imu_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.imu_canvas = tk.Canvas(imu_frame, bg='black', width=800, height=300)
        self.imu_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 控制栏
        ctrl_frame = ttk.Frame(right_frame)
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        # 进度条
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_scale = ttk.Scale(ctrl_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                         variable=self.progress_var, command=self._on_seek)
        self.progress_scale.pack(fill=tk.X, pady=5)
        
        # 播放控制
        ctrl_btn_frame = ttk.Frame(ctrl_frame)
        ctrl_btn_frame.pack()
        
        self.pause_btn = ttk.Button(ctrl_btn_frame, text="暂停", command=self._toggle_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(ctrl_btn_frame, text="<<", command=lambda: self._seek_relative(-30), width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_btn_frame, text="<", command=lambda: self._seek_relative(-5), width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_btn_frame, text=">", command=lambda: self._seek_relative(5), width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_btn_frame, text=">>", command=lambda: self._seek_relative(30), width=4).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(ctrl_btn_frame, text="  速度:").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.StringVar(value="1.0x")
        speed_combo = ttk.Combobox(ctrl_btn_frame, textvariable=self.speed_var, 
                                    values=["0.25x", "0.5x", "1.0x", "1.5x", "2.0x", "4.0x"], width=6)
        speed_combo.pack(side=tk.LEFT)
        speed_combo.bind('<<ComboboxSelected>>', self._on_speed_change)
        
        # 时间显示
        self.time_label = ttk.Label(ctrl_btn_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.LEFT, padx=20)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _browse_dir(self):
        path = filedialog.askdirectory(initialdir=self.recordings_dir)
        if path:
            self.recordings_dir = Path(path)
            self.dir_var.set(str(self.recordings_dir))
            self._scan_recordings()
    
    def _scan_recordings(self):
        self.recordings_dir = Path(self.dir_var.get())
        self.recordings = scan_recordings(self.recordings_dir)
        
        # 清空列表
        for item in self.recording_list.get_children():
            self.recording_list.delete(item)
        
        # 填充列表
        for rec in self.recordings:
            duration_str = f"{int(rec.duration_sec // 60)}:{int(rec.duration_sec % 60):02d}"
            videos_str = f"{rec.video_count}" + ("+" if rec.has_imu else "")
            self.recording_list.insert('', tk.END, iid=rec.name, 
                                       values=(rec.date_str, videos_str, duration_str))
        
        self.status_var.set(f"找到 {len(self.recordings)} 个录制")
    
    def _on_recording_select(self, event):
        selection = self.recording_list.selection()
        if not selection:
            return
        
        name = selection[0]
        for rec in self.recordings:
            if rec.name == name:
                self.selected_recording = rec
                break
        
        if self.selected_recording:
            rec = self.selected_recording
            detail = f"路径: {rec.path.name}\n"
            detail += f"视频数: {rec.video_count}\n"
            detail += f"帧数: {rec.frame_count}\n"
            detail += f"时长: {rec.duration_sec:.1f}秒\n"
            detail += f"IMU数据: {'有' if rec.has_imu else '无'}"
            self.detail_label.config(text=detail)
            self.play_btn.config(state=tk.NORMAL)
    
    def _on_double_click(self, event):
        if self.selected_recording:
            self._start_playback()
    
    def _start_playback(self):
        if not self.selected_recording or self.is_playing:
            return
        
        self._stop_playback()
        
        rec = self.selected_recording
        video_files = sorted(rec.path.glob('*.mp4'))
        imu_file = rec.path / 'imu_data.txt'
        
        if not video_files:
            messagebox.showerror("错误", "未找到视频文件")
            return
        
        # 初始化播放器
        self.player = VideoPlayer(video_files, fps=30.0, grid_width=800)
        
        # 加载 IMU 数据
        if imu_file.exists():
            self.imu_data = load_imu_data(imu_file)
            if self.imu_data:
                self.imu_viz = IMUVisualizer(self.imu_data, width=600, height=300)
        
        self.is_playing = True
        self.stop_flag = False
        
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL)
        # 如果有IMU数据，启用3D视图按钮
        if self.imu_data:
            self.view_3d_btn.config(state=tk.NORMAL)
        
        # 启动播放线程
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        
        self.status_var.set(f"正在播放: {rec.name}")
    
    def _playback_loop(self):
        """播放循环（在独立线程中运行）"""
        fps = 30.0
        frame_interval = 1.0 / fps
        last_frame_time = time.perf_counter()
        paused = False
        
        while not self.stop_flag and self.player and self.player.current_frame < self.player.total_frames:
            # 检查暂停状态
            if hasattr(self, '_paused') and self._paused:
                time.sleep(0.05)
                last_frame_time = time.perf_counter()
                continue
            
            now = time.perf_counter()
            elapsed = now - last_frame_time
            target = frame_interval / self.speed
            
            if elapsed >= target:
                # 读取帧
                video_frame = self.player.read_combined_frame()
                video_time = self.player.get_current_time()
                
                # 更新视频画面
                if video_frame is not None:
                    self._update_video_canvas(video_frame)
                
                # 更新 IMU 画面
                if self.imu_viz:
                    imu_frame = self.imu_viz.render(video_time)
                    self._update_imu_canvas(imu_frame)
                    # 保存当前IMU帧用于3D窗口
                    idx = self.imu_viz._find_frame_at_time(video_time)
                    if 0 <= idx < len(self.imu_data):
                        self.current_imu_frame = self.imu_data[idx]
                
                # 更新进度
                progress = (self.player.current_frame / self.player.total_frames) * 100
                self._update_progress(progress, video_time, self.player.total_frames / fps)
                
                last_frame_time = now
            else:
                time.sleep(0.001)
        
        # 播放结束
        self.root.after(0, self._on_playback_end)
    
    def _update_video_canvas(self, frame):
        """更新视频画布"""
        def update():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            
            # 适应画布大小
            canvas_w = self.video_canvas.winfo_width()
            canvas_h = self.video_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                scale = min(canvas_w / img.width, canvas_h / img.height)
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                img = img.resize((new_w, new_h), Image.NEAREST)
            
            self._video_photo = ImageTk.PhotoImage(img)
            self.video_canvas.delete("all")
            self.video_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._video_photo)
        
        self.root.after(0, update)
    
    def _update_imu_canvas(self, frame):
        """更新 IMU 画布"""
        def update():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            
            canvas_w = self.imu_canvas.winfo_width()
            canvas_h = self.imu_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                scale = min(canvas_w / img.width, canvas_h / img.height)
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                img = img.resize((new_w, new_h), Image.NEAREST)
            
            self._imu_photo = ImageTk.PhotoImage(img)
            self.imu_canvas.delete("all")
            self.imu_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._imu_photo)
        
        self.root.after(0, update)
    
    def _update_progress(self, progress, current_time, total_time):
        """更新进度条和时间"""
        def update():
            self.progress_var.set(progress)
            cur_str = f"{int(current_time // 60)}:{int(current_time % 60):02d}"
            tot_str = f"{int(total_time // 60)}:{int(total_time % 60):02d}"
            self.time_label.config(text=f"{cur_str} / {tot_str}")
        
        self.root.after(0, update)
    
    def _on_playback_end(self):
        """播放结束回调"""
        self.is_playing = False
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.status_var.set("播放完成")
    
    def _stop_playback(self):
        """停止播放"""
        self.stop_flag = True
        self._paused = False
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
        
        if self.player:
            self.player.release()
            self.player = None
        
        self.imu_viz = None
        self.imu_data = []
        self.is_playing = False
        
        # 关闭3D窗口
        if self.interactive_3d:
            self.interactive_3d.close()
            self.interactive_3d = None
        self.view_3d_btn.config(state=tk.DISABLED, text="3D姿态")
        
        self.play_btn.config(state=tk.NORMAL if self.selected_recording else tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED, text="暂停")
        
        self.video_canvas.delete("all")
        self.imu_canvas.delete("all")
        self.progress_var.set(0)
        self.time_label.config(text="00:00 / 00:00")
        
        self.status_var.set("已停止")
    
    def _toggle_pause(self):
        """切换暂停状态"""
        if not hasattr(self, '_paused'):
            self._paused = False
        
        self._paused = not self._paused
        self.pause_btn.config(text="继续" if self._paused else "暂停")
        self.status_var.set("已暂停" if self._paused else "播放中")
    
    def _on_seek(self, value):
        """进度条拖动"""
        if self.player and self.is_playing:
            frame = int(float(value) / 100 * self.player.total_frames)
            self.player.seek(frame)
            if self.imu_viz:
                self.imu_viz.reset()
    
    def _seek_relative(self, seconds):
        """相对跳转"""
        if self.player:
            frames = int(seconds * 30)
            self.player.seek(self.player.current_frame + frames)
            if self.imu_viz:
                self.imu_viz.reset()
    
    def _on_speed_change(self, event):
        """速度变更"""
        speed_str = self.speed_var.get()
        self.speed = float(speed_str.replace('x', ''))
    
    def _toggle_3d_view(self):
        """切换可交互3D姿态窗口"""
        if not self.imu_data:
            return
        
        if self.interactive_3d and self.interactive_3d.is_open:
            # 关闭窗口
            self.interactive_3d.close()
            self.interactive_3d = None
            self.view_3d_btn.config(text="3D姿态")
        else:
            # 打开窗口
            self.interactive_3d = Interactive3DWindow()
            self.interactive_3d.open()
            self.view_3d_btn.config(text="关闭3D")
            # 启动3D窗口更新定时器
            self._update_3d_window()
    
    def _update_3d_window(self):
        """在主线程中更新3D窗口（定时器回调）"""
        if not self.interactive_3d or not self.interactive_3d.is_open:
            return
        
        # 更新姿态数据
        if self.current_imu_frame:
            f = self.current_imu_frame
            self.interactive_3d.update_attitude(f.q0, f.q1, f.q2, f.q3, f.roll, f.pitch, f.yaw)
        
        # 渲染并显示
        key = self.interactive_3d.show()
        if key == 27:  # ESC
            self.interactive_3d.close()
            self.interactive_3d = None
            self.view_3d_btn.config(text="3D姿态")
            return
        
        # 继续定时更新（约30fps）
        self.root.after(33, self._update_3d_window)
    
    def run(self):
        self.root.mainloop()


def main():
    app = PlaybackGUI()
    app.run()


if __name__ == "__main__":
    main()
