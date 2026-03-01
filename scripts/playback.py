#!/usr/bin/env python3
"""
录制数据回放工具 - 高性能版本
同步播放多相机视频 + IMU 可视化

优化：
- 多相机合并到单窗口网格显示
- 纯 OpenCV 绘制 IMU（无 matplotlib，无线程问题）
- 视频播放保持 30fps
"""

import cv2
import numpy as np
import time
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from collections import deque


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


class VideoPlayer:
    """高性能多视频播放器 - 单窗口网格显示"""
    
    def __init__(self, video_paths: list[Path], fps: float = 30.0, grid_width: int = 960):
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
        self.is_playing = True
        self.is_paused = False
        
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
        """读取所有视频帧并合并成网格（优化版）"""
        self.frame_buffer.fill(0)  # 清空缓冲
        
        for i, cap in enumerate(self.caps):
            ret, frame = cap.read()
            if ret:
                # 计算在网格中的位置
                row = i // self.cols
                col = i % self.cols
                y1 = row * self.thumb_h
                x1 = col * self.thumb_w
                
                # 缩放并写入缓冲
                small = cv2.resize(frame, (self.thumb_w, self.thumb_h), interpolation=cv2.INTER_NEAREST)
                cv2.putText(small, f"Cam {self.video_names[i]}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                self.frame_buffer[y1:y1+self.thumb_h, x1:x1+self.thumb_w] = small
        
        self.current_frame += 1
        
        # 添加时间戳
        video_time = self.get_current_time()
        cv2.putText(
            self.frame_buffer,
            f"t={video_time:.2f}s  Frame {self.current_frame}/{self.total_frames}",
            (10, self.frame_buffer.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        return self.frame_buffer
    
    def seek(self, frame_num: int):
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.current_frame = frame_num
    
    def release(self):
        for cap in self.caps:
            cap.release()


class IMUVisualizer:
    """IMU 可视化 - 纯 OpenCV 渲染（无 matplotlib）"""
    
    def __init__(self, imu_data: list[IMUFrame], width: int = 800, height: int = 600):
        self.imu_data = imu_data
        self.width = width
        self.height = height
        
        # 计算 IMU 数据的起始时间偏移（用于与视频时间同步）
        self.start_timestamp = imu_data[0].timestamp if imu_data else 0.0
        
        # 历史数据
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
        
        # 预分配画布
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 子区域尺寸
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
        """在指定区域绘制曲线图"""
        # 背景
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 1)
        
        # 标题
        cv2.putText(canvas, title, (x + 10, y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not times or len(times) < 2:
            return
        
        # 绘图区域（留边距）
        margin = 30
        px, py = x + margin, y + margin
        pw, ph = w - margin * 2, h - margin * 2
        
        t_min, t_max = times[0], times[-1]
        if t_max <= t_min:
            t_max = t_min + 1
        
        # 绘制网格线
        for i in range(5):
            gy = py + int(ph * i / 4)
            cv2.line(canvas, (px, gy), (px + pw, gy), (50, 50, 50), 1)
        
        # 绘制数据曲线
        for data, color in data_list:
            if len(data) < 2:
                continue
            points = []
            for i, (t, v) in enumerate(zip(times, data)):
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cx, cy = x + w // 2, y + h // 2 + 15  # 稍微下移，给标签留空间
        scale = min(w, h) // 4
        
        # 等距投影变换矩阵 - 从东南上方45度俯视
        angle_h = np.radians(30)  # 水平旋转角度
        angle_v = np.radians(25)  # 俯视角度
        
        def isometric_project(p3d):
            """等距投影：NWU坐标系 -> 屏幕坐标"""
            x3d, y3d, z3d = p3d
            screen_x = -y3d * np.cos(angle_h) + x3d * np.sin(angle_h)
            screen_y = -z3d * np.cos(angle_v) - (x3d * np.cos(angle_h) + y3d * np.sin(angle_h)) * np.sin(angle_v)
            return (int(cx + screen_x * scale), int(cy + screen_y * scale))
        
        # 物体局部坐标系的盒子
        sx, sy, sz = 1.5, 1.0, 0.4
        vertices = np.array([
            [-sx/2, -sy/2, -sz/2], [sx/2, -sy/2, -sz/2],
            [sx/2, sy/2, -sz/2], [-sx/2, sy/2, -sz/2],
            [-sx/2, -sy/2, sz/2], [sx/2, -sy/2, sz/2],
            [sx/2, sy/2, sz/2], [-sx/2, sy/2, sz/2],
        ])
        
        rotated = (R @ vertices.T).T
        projected = [isometric_project(v) for v in rotated]
        
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i, j in edges:
            avg_depth = (rotated[i][0] * np.cos(angle_h) + rotated[i][1] * np.sin(angle_h) + 
                        rotated[j][0] * np.cos(angle_h) + rotated[j][1] * np.sin(angle_h)) / 2
            brightness = int(150 + avg_depth * 40)
            brightness = max(80, min(255, brightness))
            color = (brightness, brightness, brightness)
            cv2.line(canvas, projected[i], projected[j], color, 2, cv2.LINE_AA)
        
        # 绘制前面标记
        front_center = R @ np.array([sx/2 + 0.2, 0, 0])
        front_p = isometric_project(front_center)
        front_base = R @ np.array([sx/2, 0, 0])
        front_base_p = isometric_project(front_base)
        cv2.arrowedLine(canvas, front_base_p, front_p, (0, 100, 255), 2, cv2.LINE_AA, tipLength=0.5)
        
        # 绘制世界坐标系参考轴
        origin = np.array([0, 0, 0])
        axis_length = 1.3
        world_axes = [
            (np.array([axis_length, 0, 0]), (0, 0, 255), "N"),   # X=北
            (np.array([0, axis_length, 0]), (0, 255, 0), "W"),   # Y=西
            (np.array([0, 0, axis_length]), (255, 100, 0), "U"), # Z=天
        ]
        
        p_origin = isometric_project(origin)
        for axis_vec, color, label in world_axes:
            p_end = isometric_project(axis_vec)
            cv2.line(canvas, p_origin, p_end, color, 1, cv2.LINE_AA)
            cv2.putText(canvas, label, (p_end[0] + 3, p_end[1] + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 绘制物体坐标系轴（R的列是局部轴在全局坐标系中的方向）
        body_axis_length = 1.0
        body_axes = [
            (R[:, 0] * body_axis_length, (100, 100, 255), "x"),
            (R[:, 1] * body_axis_length, (100, 255, 100), "y"),
            (R[:, 2] * body_axis_length, (255, 150, 100), "z"),
        ]
        
        for axis_vec, color, label in body_axes:
            p_end = isometric_project(axis_vec)
            cv2.arrowedLine(canvas, p_origin, p_end, color, 2, cv2.LINE_AA, tipLength=0.15)
    
    def render(self, video_time: float) -> np.ndarray:
        """渲染 IMU 可视化画面"""
        idx = self._find_frame_at_time(video_time)
        if idx >= len(self.imu_data):
            idx = len(self.imu_data) - 1
        if idx < 0 or not self.imu_data:
            self.canvas.fill(0)
            return self.canvas
        
        frame = self.imu_data[idx]
        
        # 更新历史数据
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
        
        # 左上：3D 姿态
        R = quaternion_to_rotation_matrix(frame.q0, frame.q1, frame.q2, frame.q3)
        self._draw_3d_box(self.canvas, 0, 0, self.plot_w, self.plot_h, R)
        
        # 右上：欧拉角
        self._draw_plot(
            self.canvas, self.plot_w, 0, self.plot_w, self.plot_h,
            f"Euler (R:{frame.roll:.1f} P:{frame.pitch:.1f} Y:{frame.yaw:.1f})",
            [
                (self.roll_history, (0, 0, 255)),    # Red
                (self.pitch_history, (0, 255, 0)),  # Green
                (self.yaw_history, (255, 0, 0)),    # Blue
            ],
            -180, 180, times
        )
        
        # 左下：加速度
        self._draw_plot(
            self.canvas, 0, self.plot_h, self.plot_w, self.plot_h,
            f"Accel (X:{frame.acc_x:.2f} Y:{frame.acc_y:.2f} Z:{frame.acc_z:.2f})",
            [
                (self.acc_x_history, (0, 0, 255)),
                (self.acc_y_history, (0, 255, 0)),
                (self.acc_z_history, (255, 0, 0)),
            ],
            -3, 3, times
        )
        
        # 右下：角速度
        self._draw_plot(
            self.canvas, self.plot_w, self.plot_h, self.plot_w, self.plot_h,
            f"Gyro (X:{frame.gyro_x:.0f} Y:{frame.gyro_y:.0f} Z:{frame.gyro_z:.0f})",
            [
                (self.gyro_x_history, (0, 0, 255)),
                (self.gyro_y_history, (0, 255, 0)),
                (self.gyro_z_history, (255, 0, 0)),
            ],
            -500, 500, times
        )
        
        # 时间戳
        cv2.putText(self.canvas, f"t={frame.timestamp:.2f}s",
                    (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return self.canvas
    
    def reset(self):
        """重置历史数据"""
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


def find_recording_files(directory: Path):
    video_files = sorted(directory.glob('*.mp4'))
    imu_file = directory / 'imu_data.txt'
    return video_files, imu_file if imu_file.exists() else None


def main():
    parser = argparse.ArgumentParser(description='录制数据回放（高性能版）')
    parser.add_argument('directory', type=str, help='录制目录路径')
    parser.add_argument('--fps', type=float, default=30.0, help='播放帧率')
    parser.add_argument('--width', type=int, default=960, help='视频网格总宽度')
    args = parser.parse_args()
    
    recording_dir = Path(args.directory)
    if not recording_dir.exists():
        print(f"错误: 目录不存在: {recording_dir}")
        sys.exit(1)
    
    video_files, imu_file = find_recording_files(recording_dir)
    
    print(f"目录: {recording_dir}")
    print(f"视频: {len(video_files)} 个")
    
    if not video_files:
        print("错误: 未找到视频文件")
        sys.exit(1)
    
    player = VideoPlayer(video_files, args.fps, args.width)
    print(f"帧数: {player.total_frames}, 时长: {player.total_frames/args.fps:.1f}s")
    print(f"布局: {player.rows}x{player.cols}")
    
    imu_viz = None
    if imu_file:
        print("加载 IMU...")
        imu_data = load_imu_data(imu_file)
        print(f"IMU: {len(imu_data)} 条")
        if imu_data:
            imu_viz = IMUVisualizer(imu_data, width=640, height=480)
    
    print("\n空格=暂停 q=退出 ←/→=快退/快进 ↑/↓=变速 r=重置\n")
    
    frame_interval = 1.0 / args.fps
    speed = 1.0
    last_frame_time = time.perf_counter()
    
    cv2.namedWindow('Video Playback', cv2.WINDOW_NORMAL)
    if imu_viz:
        cv2.namedWindow('IMU Visualization', cv2.WINDOW_NORMAL)
    
    try:
        while player.is_playing and player.current_frame < player.total_frames:
            now = time.perf_counter()
            
            if not player.is_paused:
                elapsed = now - last_frame_time
                target = frame_interval / speed
                
                if elapsed >= target:
                    # 视频帧
                    combined = player.read_combined_frame()
                    if combined is not None:
                        cv2.imshow('Video Playback', combined)
                    
                    # IMU 帧（同步渲染）
                    if imu_viz:
                        imu_frame = imu_viz.render(player.get_current_time())
                        cv2.imshow('IMU Visualization', imu_frame)
                    
                    last_frame_time = now
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                player.is_paused = not player.is_paused
                print("暂停" if player.is_paused else "继续")
            elif key == ord('r'):
                player.seek(0)
                if imu_viz:
                    imu_viz.reset()
                print("重置")
            elif key == 81 or key == 2:  # 左箭头
                player.seek(player.current_frame - int(args.fps))
                if imu_viz:
                    imu_viz.reset()
            elif key == 83 or key == 3:  # 右箭头
                player.seek(player.current_frame + int(args.fps))
            elif key == 82 or key == 0:  # 上箭头
                speed = min(4.0, speed * 1.5)
                print(f"速度: {speed:.1f}x")
            elif key == 84 or key == 1:  # 下箭头
                speed = max(0.25, speed / 1.5)
                print(f"速度: {speed:.1f}x")
    
    finally:
        player.release()
        cv2.destroyAllWindows()
    
    print("播放结束")


if __name__ == "__main__":
    main()
