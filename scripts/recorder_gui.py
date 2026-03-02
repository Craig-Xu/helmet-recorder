#!/usr/bin/env python3
"""
多相机+IMU数据采集录制系统 - GUI版本 (多进程架构)
同时录制多个相机视频和IMU数据

架构说明：
- 每个相机在独立进程中运行，完全绑过 Python GIL
- 采集和写入在同一进程中完成，无跨进程大数据传输
- 预览帧通过共享内存传递（低开销）
"""

import cv2
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from datetime import datetime
import queue
import yaml
import sys
import os
import multiprocessing as mp
from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parent.parent

# 添加项目根目录，确保可导入 camera / imu 包
sys.path.insert(0, str(REPO_ROOT))

from camera.multi_camera import list_cameras
from camera.mp_camera import MultiCameraManager  # 多进程相机管理器
from imu.imu_manager import IMUManager
from scripts.common import resolve_cam_to_video
from scripts.playback_gui import scan_recordings, load_imu_data, VideoPlayer, IMUVisualizer


# 注意: SyncController 和 VideoRecorder 已被多进程版本 MultiCameraManager 替代
# 多进程版本位于 camera/mp_camera.py，每个相机在独立进程中运行


class IMURecorder:
    """IMU数据录制器"""
    
    def __init__(self, port, baudrate=460800):
        self.port = port
        self.baudrate = baudrate
        self.manager = None
        self.is_recording = False
        self.data_queue = queue.Queue()
        self.start_time = None
        self.file = None
        self.data_count = 0
        
    def start(self, output_path):
        """开始录制"""
        try:
            # 创建IMU管理器
            config = {
                'port': self.port,
                'baudrate': self.baudrate,
                'debug': False
            }
            self.manager = IMUManager(config)
            
            # 连接设备
            if not self.manager.connect():
                return False
            
            # 设置数据回调
            self.manager.set_data_callback(self._on_imu_data)
            
            # 打开文件
            output_file = output_path / "imu_data.txt"
            self.file = open(output_file, 'w', encoding='utf-8')
            
            # 写入文件头
            self._write_header()
            
            # 启动IMU采集
            if not self.manager.start():
                self.file.close()
                return False
            
            self.is_recording = True
            self.start_time = time.perf_counter()
            self.data_count = 0
            return True
            
        except Exception as e:
            print(f"IMU启动失败: {e}")
            if self.file:
                self.file.close()
            return False
    
    def _write_header(self):
        """写入CSV格式的文件头"""
        header = (
            "# IMU Data Recording\n"
            f"# Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Port: {self.port}, Baudrate: {self.baudrate}\n"
            "# Format: timestamp(s), roll(deg), pitch(deg), yaw(deg), "
            "q0, q1, q2, q3, "
            "acc_x(g), acc_y(g), acc_z(g), "
            "gyro_x(deg/s), gyro_y(deg/s), gyro_z(deg/s), "
            "mag_x, mag_y, mag_z, temp(C)\n"
        )
        self.file.write(header)
        self.file.flush()
    
    def _on_imu_data(self, data):
        """IMU数据回调"""
        if self.is_recording and self.start_time:
            timestamp = time.perf_counter() - self.start_time
            # 对齐到统一同步点之前的数据直接丢弃
            if timestamp >= 0:
                self.data_queue.put((timestamp, data))
    
    def flush_data(self):
        """将队列中的数据写入文件"""
        if not self.is_recording:
            return 0
        
        count = 0
        while not self.data_queue.empty():
            try:
                timestamp, data = self.data_queue.get_nowait()
                
                # 格式化数据行
                line = (
                    f"{timestamp:.6f},"
                    f"{data['roll']:.3f},{data['pitch']:.3f},{data['yaw']:.3f},"
                    f"{data['q0']:.6f},{data['q1']:.6f},{data['q2']:.6f},{data['q3']:.6f},"
                    f"{data['acc_x']:.6f},{data['acc_y']:.6f},{data['acc_z']:.6f},"
                    f"{data['gyro_x']:.3f},{data['gyro_y']:.3f},{data['gyro_z']:.3f},"
                    f"{data['norm_mag_x']:.6f},{data['norm_mag_y']:.6f},{data['norm_mag_z']:.6f},"
                    f"{data['sensor_temp']:.2f}\n"
                )
                
                self.file.write(line)
                self.data_count += 1
                count += 1
                
            except queue.Empty:
                break
        
        if count > 0:
            self.file.flush()
        
        return count
    
    def reset_start_time(self, sync_time: float | None = None):
        """重置起始时间（用于与相机同步）

        Args:
            sync_time: 与相机共享的 perf_counter 同步时间点；若为空则用当前时间。
        """
        # 清空队列中之前的数据
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        # 重置起始时间
        self.start_time = float(sync_time) if sync_time is not None else time.perf_counter()
        self.data_count = 0
        print(f"IMU 起始时间已重置: {self.start_time:.6f}")
    
    def stop(self):
        """停止录制"""
        self.is_recording = False
        
        # 停止IMU管理器
        if self.manager:
            self.manager.stop()
        
        # 写入剩余数据
        self.flush_data()
        
        # 关闭文件
        if self.file:
            self.file.close()
    
    def get_latest_data(self):
        """获取最新的IMU数据用于显示"""
        if self.manager:
            return self.manager.get_data()
        return None


class RecorderGUI:
    """录制系统GUI主窗口 - 多进程架构版本"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("多相机+IMU数据采集系统 (多进程)")
        self.root.geometry("1200x800")
        
        # 加载配置
        self.config = self.load_config()
        self.cam_to_video, self.video_to_cam = self._build_cam_maps_from_config()
        
        # 录制器
        self.camera_manager = None  # 多进程相机管理器
        self.camera_ids = []  # 成功启动的相机ID列表
        self.imu_recorder = None
        self.is_recording = False
        
        # 录制参数
        self.output_dir = Path.home() / "recordings"
        self.current_session_dir = None
        self.recording_start_time = None
        
        # 定时器
        self.update_timer = None
        self.imu_flush_timer = None
        self.preview_interval_ms = 120  # 约 8fps 预览，比原 3fps 更流畅

        # 页面模式: record / playback
        self.current_mode = None

        # 回放状态
        self.recordings_dir = Path.home() / "recordings"
        self.playback_recordings = []
        self.selected_playback_recording = None
        self.player = None
        self.imu_viz = None
        self.imu_data = []
        self.playback_thread = None
        self.playback_stop_flag = False
        self.playback_paused = False
        self.playback_speed = 1.0
        self.is_playback_running = False
        
        # 应用深色主题
        self._setup_theme()

        # 创建UI
        self.create_ui()
        
        # 初始化相机列表
        self.scan_cameras()
    
    def _setup_theme(self):
        """深色主题配色与 ttk 样式"""
        self.C = {
            'bg':         '#1a1a2e',
            'sidebar':    '#151525',
            'card':       '#1e2140',
            'border':     '#2a2d4a',
            'accent':     '#5b8dee',
            'rec':        '#c94f4f',
            'rec_hover':  '#e06060',
            'stop':       '#3a9d5d',
            'stop_hover': '#4ab56e',
            'text':       '#dde1f0',
            'text_dim':   '#5a6080',
            'value':      '#7dd3fc',
            'good':       '#86efac',
            'warn':       '#fcd34d',
            'input_bg':   '#10101e',
            'input_fg':   '#c0c4e0',
            'header_bg':  '#0d0d1e',
            'section':    '#4f7fd4',
        }
        C = self.C
        self.root.configure(bg=C['bg'])

        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        style.configure('D.TEntry',
            fieldbackground=C['input_bg'], foreground=C['input_fg'],
            bordercolor=C['border'], lightcolor=C['border'], darkcolor=C['border'],
            insertcolor=C['text'], selectbackground=C['accent'], selectforeground='white',
            padding=4,
        )
        style.map('D.TEntry',
            fieldbackground=[('disabled', C['card'])],
            foreground=[('disabled', C['text_dim'])],
        )
        style.configure('D.TSpinbox',
            fieldbackground=C['input_bg'], foreground=C['input_fg'],
            bordercolor=C['border'], lightcolor=C['border'], darkcolor=C['border'],
            arrowcolor=C['text_dim'], insertcolor=C['text'],
            padding=4,
        )
        style.map('D.TSpinbox',
            fieldbackground=[('disabled', C['card'])],
            foreground=[('disabled', C['text_dim'])],
        )

    def load_config(self):
        """加载配置文件"""
        try:
            config_path = REPO_ROOT / "config" / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"配置文件加载失败: {e}")
        return {}

    def _build_cam_maps_from_config(self):
        """从 config 构建 cam_id->video_id 与 video_id->cam_id 映射。"""
        cam_cfg = self.config.get('camera', {})
        camera_ids = [int(x) for x in cam_cfg.get('ids', [0])]
        raw_index_map = cam_cfg.get('index_map', {})

        # 统一为 cam_id -> /dev/video_id
        cam_to_video = resolve_cam_to_video(camera_ids, raw_index_map)

        # 补齐缺失项：按顺序给未映射 video 分配下一个 cam_id
        used_videos = set(cam_to_video.values())
        next_cam_id = (max(cam_to_video.keys()) + 1) if cam_to_video else 0
        for video_id in camera_ids:
            if video_id not in used_videos:
                cam_to_video[next_cam_id] = video_id
                next_cam_id += 1

        video_to_cam = {int(v): int(c) for c, v in cam_to_video.items()}
        return cam_to_video, video_to_cam

    def _ordered_video_ids_from_config(self):
        """按 cam0..camN 顺序返回对应 /dev/video ID 列表。"""
        return [self.cam_to_video[cid] for cid in sorted(self.cam_to_video.keys())]

    def _format_cam_label(self, video_id: int) -> str:
        cam_id = self.video_to_cam.get(int(video_id), int(video_id))
        return f"Cam{cam_id} (/dev/video{video_id})"

    def _rename_recorded_videos_to_cam_ids(self):
        """将输出视频按逻辑相机编号重命名（例如 0..7.mp4）。"""
        if not self.current_session_dir:
            return

        for video_id in self.camera_ids:
            cam_id = self.video_to_cam.get(int(video_id), int(video_id))
            src = self.current_session_dir / f"{int(video_id)}.mp4"
            dst = self.current_session_dir / f"{int(cam_id)}.mp4"
            if not src.exists() or src == dst:
                continue
            try:
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
            except Exception as e:
                print(f"重命名失败 {src.name} -> {dst.name}: {e}")
    
    def create_ui(self):
        """创建用户界面 - 深色主题"""
        C = self.C

        # 顶部标题栏
        header = tk.Frame(self.root, bg=C['header_bg'], height=46)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header, text='  ⬛  HELMET RECORDER',
            bg=C['header_bg'], fg=C['accent'],
            font=('Arial', 12, 'bold'), anchor='w', padx=14,
        ).pack(side=tk.LEFT, fill=tk.Y)

        # 录制计时器（右上角）
        self.record_time_var = tk.StringVar(value='00:00:00')
        self._timer_label = tk.Label(
            header, textvariable=self.record_time_var,
            bg=C['header_bg'], fg=C['text_dim'],
            font=('Courier', 15, 'bold'), padx=16,
        )
        self._timer_label.pack(side=tk.RIGHT, fill=tk.Y)

        # 底部状态栏（必须在 body 之前 pack 才能自动占据底部）
        self.create_status_bar(self.root)

        # 主区域
        body = tk.Frame(self.root, bg=C['bg'])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.sidebar = tk.Frame(body, bg=C['sidebar'], width=286)
        self.sidebar.grid(row=0, column=0, sticky='nsew')
        self.sidebar.pack_propagate(False)
        self.sidebar.grid_propagate(False)

        self.content_container = tk.Frame(body, bg=C['bg'])
        self.content_container.grid(row=0, column=1, sticky='nsew')
        self.content_container.rowconfigure(0, weight=1)
        self.content_container.columnconfigure(0, weight=1)

        self.create_control_panel(self.sidebar)
        self.create_preview_panel(self.content_container)
        self.switch_mode('record')
    
    def create_control_panel(self, parent):
        """创建左侧控制面板 - 深色主题侧边栏"""
        C = self.C
        self._setting_widgets = []

        def sec(host, icon, text):
            f = tk.Frame(host, bg=C['sidebar'])
            f.pack(fill=tk.X, padx=14, pady=(14, 2))
            tk.Label(f, text=f'{icon}  {text}', bg=C['sidebar'], fg=C['section'],
                     font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
            tk.Frame(f, bg=C['border'], height=1).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), pady=5)

        def row(host, lbl, widget_fn):
            f = tk.Frame(host, bg=C['sidebar'])
            f.pack(fill=tk.X, padx=14, pady=2)
            tk.Label(f, text=lbl, bg=C['sidebar'], fg=C['text_dim'],
                     font=('Arial', 9), width=8, anchor='w').pack(side=tk.LEFT)
            w = widget_fn(f)
            w.pack(side=tk.LEFT, fill=tk.X, expand=True)
            return w

        nav = tk.Frame(parent, bg=C['sidebar'])
        nav.pack(fill=tk.X, padx=14, pady=(12, 6))
        self.nav_record_btn = tk.Button(
            nav, text='●  录制', command=lambda: self.switch_mode('record'),
            bg=C['accent'], fg='white', activebackground=C['section'], activeforeground='white',
            relief=tk.FLAT, cursor='hand2', font=('Arial', 10, 'bold'), pady=7,
        )
        self.nav_record_btn.pack(fill=tk.X, pady=(0, 6))
        self.nav_playback_btn = tk.Button(
            nav, text='▶  回放', command=lambda: self.switch_mode('playback'),
            bg=C['card'], fg=C['text_dim'], activebackground=C['border'], activeforeground=C['text'],
            relief=tk.FLAT, cursor='hand2', font=('Arial', 10, 'bold'), pady=7,
        )
        self.nav_playback_btn.pack(fill=tk.X)

        tk.Frame(parent, bg=C['border'], height=1).pack(fill=tk.X, padx=14, pady=(6, 4))

        self.record_controls_frame = tk.Frame(parent, bg=C['sidebar'])
        self.record_controls_frame.pack(fill=tk.BOTH, expand=True)

        # ── 相机配置 ──
        sec(self.record_controls_frame, '🎥', '相机配置')
        self.camera_ids_var = tk.StringVar(
            value=','.join(map(str, self._ordered_video_ids_from_config()))
        )
        w = row(self.record_controls_frame, 'Camera IDs', lambda f: ttk.Entry(f, textvariable=self.camera_ids_var, style='D.TEntry'))
        self._setting_widgets.append(w)

        scan_btn = tk.Button(
            self.record_controls_frame, text='↻  扫描可用相机',
            bg=C['card'], fg=C['accent'],
            activebackground=C['border'], activeforeground=C['accent'],
            relief=tk.FLAT, cursor='hand2', pady=5, font=('Arial', 9),
            command=self.scan_cameras,
        )
        scan_btn.pack(fill=tk.X, padx=14, pady=(2, 6))
        self._setting_widgets.append(scan_btn)

        self.width_var  = tk.IntVar(value=self.config.get('camera', {}).get('width', 640))
        self.height_var = tk.IntVar(value=self.config.get('camera', {}).get('height', 480))
        res_f = tk.Frame(self.record_controls_frame, bg=C['sidebar'])
        res_f.pack(fill=tk.X, padx=14, pady=2)
        tk.Label(res_f, text='分辨率', bg=C['sidebar'], fg=C['text_dim'],
                 font=('Arial', 9), width=8, anchor='w').pack(side=tk.LEFT)
        we = ttk.Entry(res_f, textvariable=self.width_var, style='D.TEntry', width=7)
        we.pack(side=tk.LEFT)
        tk.Label(res_f, text=' × ', bg=C['sidebar'], fg=C['text_dim'],
                 font=('Arial', 9)).pack(side=tk.LEFT)
        he = ttk.Entry(res_f, textvariable=self.height_var, style='D.TEntry', width=7)
        he.pack(side=tk.LEFT)
        self._setting_widgets += [we, he]

        self.fps_var = tk.IntVar(value=30)
        fps_s = row(self.record_controls_frame, '帧率 fps', lambda f: ttk.Spinbox(
            f, from_=1, to=120, textvariable=self.fps_var,
            style='D.TSpinbox', width=8,
        ))
        self._setting_widgets.append(fps_s)

        # ── IMU 配置 ──
        sec(self.record_controls_frame, '📡', 'IMU 配置')
        self.imu_port_var = tk.StringVar(
            value=self.config.get('imu', {}).get('port', '/dev/ttyUSB0')
        )
        w = row(self.record_controls_frame, '串口', lambda f: ttk.Entry(f, textvariable=self.imu_port_var, style='D.TEntry'))
        self._setting_widgets.append(w)

        self.imu_baud_var = tk.IntVar(value=self.config.get('imu', {}).get('bps', 460800))
        w = row(self.record_controls_frame, '波特率', lambda f: ttk.Entry(f, textvariable=self.imu_baud_var, style='D.TEntry'))
        self._setting_widgets.append(w)

        # ── 输出路径 ──
        sec(self.record_controls_frame, '💾', '输出路径')
        self.output_dir_var = tk.StringVar(value=str(self.output_dir))
        dir_lbl = tk.Label(
            self.record_controls_frame, textvariable=self.output_dir_var,
            bg=C['input_bg'], fg=C['text_dim'], font=('Arial', 8),
            anchor='w', padx=8, pady=5, relief=tk.FLAT,
        )
        dir_lbl.pack(fill=tk.X, padx=14, pady=2)
        browse_btn = tk.Button(
            self.record_controls_frame, text='📁  选择目录',
            bg=C['card'], fg=C['text_dim'],
            activebackground=C['border'], activeforeground=C['text'],
            relief=tk.FLAT, cursor='hand2', pady=5, font=('Arial', 9),
            command=self.select_output_dir,
        )
        browse_btn.pack(fill=tk.X, padx=14, pady=(2, 12))
        self._setting_widgets.append(browse_btn)

        # ── 录制按钮 ──
        tk.Frame(self.record_controls_frame, bg=C['border'], height=1).pack(fill=tk.X, padx=14, pady=(4, 12))
        self.record_button = tk.Button(
            self.record_controls_frame,
            text='●  开始录制',
            command=self.toggle_recording,
            bg=C['rec'], fg='white',
            activebackground=C['rec_hover'], activeforeground='white',
            font=('Arial', 13, 'bold'), relief=tk.FLAT, cursor='hand2', pady=13,
        )
        self.record_button.pack(fill=tk.X, padx=14, pady=(0, 16))

        # 回放模式控件
        self.playback_controls_frame = tk.Frame(parent, bg=C['sidebar'])

        sec(self.playback_controls_frame, '📁', '录制目录')
        self.playback_dir_var = tk.StringVar(value=str(self.recordings_dir))
        play_dir_lbl = tk.Label(
            self.playback_controls_frame, textvariable=self.playback_dir_var,
            bg=C['input_bg'], fg=C['text_dim'], font=('Arial', 8),
            anchor='w', padx=8, pady=5, relief=tk.FLAT,
        )
        play_dir_lbl.pack(fill=tk.X, padx=14, pady=2)
        play_dir_btn = tk.Button(
            self.playback_controls_frame, text='📁  选择目录',
            bg=C['card'], fg=C['text_dim'], activebackground=C['border'], activeforeground=C['text'],
            relief=tk.FLAT, cursor='hand2', pady=5, font=('Arial', 9),
            command=self._browse_playback_dir,
        )
        play_dir_btn.pack(fill=tk.X, padx=14, pady=(2, 6))

        refresh_btn = tk.Button(
            self.playback_controls_frame, text='↻  刷新录制列表',
            bg=C['card'], fg=C['accent'], activebackground=C['border'], activeforeground=C['accent'],
            relief=tk.FLAT, cursor='hand2', pady=5, font=('Arial', 9),
            command=self._scan_playback_recordings,
        )
        refresh_btn.pack(fill=tk.X, padx=14, pady=(0, 8))

        sec(self.playback_controls_frame, '🗂', '录制列表')
        list_wrap = tk.Frame(self.playback_controls_frame, bg=C['sidebar'])
        list_wrap.pack(fill=tk.BOTH, expand=True, padx=14, pady=2)
        self.playback_listbox = tk.Listbox(
            list_wrap, bg=C['input_bg'], fg=C['text'], selectbackground=C['accent'],
            selectforeground='white', relief=tk.FLAT, height=10,
        )
        self.playback_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll = tk.Scrollbar(list_wrap, command=self.playback_listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.playback_listbox.config(yscrollcommand=list_scroll.set)
        self.playback_listbox.bind('<<ListboxSelect>>', self._on_playback_select)
        self.playback_listbox.bind('<Double-1>', self._on_playback_double_click)

        self.playback_detail_var = tk.StringVar(value='请选择一个录制')
        tk.Label(
            self.playback_controls_frame, textvariable=self.playback_detail_var,
            bg=C['card'], fg=C['text_dim'], justify=tk.LEFT, anchor='nw',
            font=('Arial', 8), padx=8, pady=6,
        ).pack(fill=tk.X, padx=14, pady=(6, 8))

        ctrl_row = tk.Frame(self.playback_controls_frame, bg=C['sidebar'])
        ctrl_row.pack(fill=tk.X, padx=14, pady=(0, 6))
        self.play_btn = tk.Button(
            ctrl_row, text='▶ 播放', command=self._start_playback,
            bg=C['card'], fg=C['text_dim'], relief=tk.FLAT, cursor='hand2', state='disabled'
        )
        self.play_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.pause_btn = tk.Button(
            ctrl_row, text='⏸ 暂停', command=self._toggle_playback_pause,
            bg=C['card'], fg=C['text_dim'], relief=tk.FLAT, cursor='hand2', state='disabled'
        )
        self.pause_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.stop_btn = tk.Button(
            ctrl_row, text='■ 停止', command=self._stop_playback,
            bg=C['card'], fg=C['text_dim'], relief=tk.FLAT, cursor='hand2', state='disabled'
        )
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        speed_row = tk.Frame(self.playback_controls_frame, bg=C['sidebar'])
        speed_row.pack(fill=tk.X, padx=14, pady=(0, 8))
        tk.Label(speed_row, text='速度', bg=C['sidebar'], fg=C['text_dim'], font=('Arial', 9)).pack(side=tk.LEFT)
        self.playback_speed_var = tk.StringVar(value='1.0x')
        self.playback_speed_combo = ttk.Combobox(
            speed_row, textvariable=self.playback_speed_var,
            values=['0.25x', '0.5x', '1.0x', '1.5x', '2.0x', '4.0x'], width=7, state='readonly'
        )
        self.playback_speed_combo.pack(side=tk.LEFT, padx=8)
        self.playback_speed_combo.bind('<<ComboboxSelected>>', self._on_playback_speed_change)

        seek_row = tk.Frame(self.playback_controls_frame, bg=C['sidebar'])
        seek_row.pack(fill=tk.X, padx=14, pady=(0, 12))
        tk.Button(seek_row, text='-30s', command=lambda: self._seek_playback_relative(-30),
                  bg=C['card'], fg=C['text_dim'], relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(seek_row, text='-5s', command=lambda: self._seek_playback_relative(-5),
                  bg=C['card'], fg=C['text_dim'], relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=2)
        tk.Button(seek_row, text='+5s', command=lambda: self._seek_playback_relative(5),
                  bg=C['card'], fg=C['text_dim'], relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=2)
        tk.Button(seek_row, text='+30s', command=lambda: self._seek_playback_relative(30),
                  bg=C['card'], fg=C['text_dim'], relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=(4, 0))
    
    def create_preview_panel(self, parent):
        """创建右侧预览面板 - 深色主题"""
        C = self.C
        right = tk.Frame(parent, bg=C['bg'])
        right.grid(row=0, column=0, sticky='nsew')
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.record_right = tk.Frame(right, bg=C['bg'])
        self.record_right.place(x=0, y=0, relwidth=1, relheight=1)
        self.record_right.rowconfigure(0, weight=3)
        self.record_right.rowconfigure(1, weight=1)
        self.record_right.columnconfigure(0, weight=1)

        # ── 预览画布 ──
        canvas_wrap = tk.Frame(self.record_right, bg=C['card'], padx=2, pady=2)
        canvas_wrap.grid(row=0, column=0, sticky='nsew', padx=8, pady=(8, 4))
        self.preview_canvas = tk.Canvas(canvas_wrap, bg='#06060f', highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # ── IMU 数据面板 ──
        imu_card = tk.Frame(self.record_right, bg=C['card'])
        imu_card.grid(row=1, column=0, sticky='nsew', padx=8, pady=(4, 8))

        hdr = tk.Frame(imu_card, bg=C['card'])
        hdr.pack(fill=tk.X, padx=12, pady=(7, 0))
        tk.Label(hdr, text='📡  IMU 实时数据',
                 bg=C['card'], fg=C['text_dim'], font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        tk.Frame(imu_card, bg=C['border'], height=1).pack(fill=tk.X, padx=12, pady=(4, 0))

        # 数值网格：3列（姿态角 / 加速度 / 角速度）
        grid_f = tk.Frame(imu_card, bg=C['card'])
        grid_f.pack(fill=tk.X, padx=12, pady=4)
        for c in range(3):
            grid_f.columnconfigure(c, weight=1)

        groups = [
            ('姿态角', [('Roll',   'roll'),   ('Pitch', 'pitch'), ('Yaw',    'yaw')],    C['value']),
            ('加速度', [('Acc X',  'acc_x'),  ('Acc Y', 'acc_y'),  ('Acc Z',  'acc_z')],  C['good']),
            ('角速度', [('Gyro X', 'gyro_x'), ('Gyro Y','gyro_y'), ('Gyro Z', 'gyro_z')], C['warn']),
        ]
        self._imu_val_labels = {}
        for col, (group_title, fields, val_color) in enumerate(groups):
            gcol = tk.Frame(grid_f, bg=C['card'])
            gcol.grid(row=0, column=col, sticky='nsew', padx=4)
            tk.Label(gcol, text=group_title, bg=C['card'], fg=C['text_dim'],
                     font=('Arial', 8, 'bold'), anchor='w').pack(fill=tk.X, pady=(0, 3))
            for lbl_text, key in fields:
                rf = tk.Frame(gcol, bg=C['input_bg'])
                rf.pack(fill=tk.X, pady=1)
                tk.Label(rf, text=lbl_text, bg=C['input_bg'], fg=C['text_dim'],
                         font=('Arial', 8), anchor='w', width=7, padx=4).pack(side=tk.LEFT)
                val_lbl = tk.Label(rf, text='—', bg=C['input_bg'], fg=val_color,
                                   font=('Courier', 9, 'bold'), anchor='e', padx=4)
                val_lbl.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                self._imu_val_labels[key] = val_lbl

        # 底部：磁力计 + 温度 + 数据包
        tk.Frame(imu_card, bg=C['border'], height=1).pack(fill=tk.X, padx=12, pady=(4, 0))
        bot = tk.Frame(imu_card, bg=C['card'])
        bot.pack(fill=tk.X, padx=12, pady=(4, 7))

        for lbl_text, key in [('Mag X', 'norm_mag_x'), ('Mag Y', 'norm_mag_y'), ('Mag Z', 'norm_mag_z')]:
            cell = tk.Frame(bot, bg=C['input_bg'])
            cell.pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(cell, text=lbl_text, bg=C['input_bg'], fg=C['text_dim'],
                     font=('Arial', 8), padx=4).pack(side=tk.LEFT)
            val_lbl = tk.Label(cell, text='—', bg=C['input_bg'], fg=C['accent'],
                               font=('Courier', 9, 'bold'), padx=4)
            val_lbl.pack(side=tk.LEFT)
            self._imu_val_labels[key] = val_lbl

        temp_cell = tk.Frame(bot, bg=C['input_bg'])
        temp_cell.pack(side=tk.LEFT, padx=(8, 4))
        tk.Label(temp_cell, text='温度', bg=C['input_bg'], fg=C['text_dim'],
                 font=('Arial', 8), padx=4).pack(side=tk.LEFT)
        self._imu_temp_lbl = tk.Label(temp_cell, text='—', bg=C['input_bg'], fg=C['warn'],
                                      font=('Courier', 9, 'bold'), padx=4)
        self._imu_temp_lbl.pack(side=tk.LEFT)

        pkt_cell = tk.Frame(bot, bg=C['input_bg'])
        pkt_cell.pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(pkt_cell, text='数据包', bg=C['input_bg'], fg=C['text_dim'],
                 font=('Arial', 8), padx=4).pack(side=tk.LEFT)
        self._imu_pkt_lbl = tk.Label(pkt_cell, text='0', bg=C['input_bg'], fg=C['good'],
                                     font=('Courier', 9, 'bold'), padx=4)
        self._imu_pkt_lbl.pack(side=tk.LEFT)

        # 回放视图
        self.playback_right = tk.Frame(right, bg=C['bg'])
        self.playback_right.place(x=0, y=0, relwidth=1, relheight=1)
        self.playback_right.rowconfigure(0, weight=3)
        self.playback_right.rowconfigure(1, weight=2)
        self.playback_right.rowconfigure(2, weight=0)
        self.playback_right.columnconfigure(0, weight=1)

        playback_video_wrap = tk.Frame(self.playback_right, bg=C['card'], padx=2, pady=2)
        playback_video_wrap.grid(row=0, column=0, sticky='nsew', padx=8, pady=(8, 4))
        self.playback_video_canvas = tk.Canvas(playback_video_wrap, bg='#06060f', highlightthickness=0)
        self.playback_video_canvas.pack(fill=tk.BOTH, expand=True)

        playback_imu_wrap = tk.Frame(self.playback_right, bg=C['card'], padx=2, pady=2)
        playback_imu_wrap.grid(row=1, column=0, sticky='nsew', padx=8, pady=4)
        self.playback_imu_canvas = tk.Canvas(playback_imu_wrap, bg='#06060f', highlightthickness=0)
        self.playback_imu_canvas.pack(fill=tk.BOTH, expand=True)

        playback_ctrl = tk.Frame(self.playback_right, bg=C['card'])
        playback_ctrl.grid(row=2, column=0, sticky='ew', padx=8, pady=(4, 8))
        playback_ctrl.columnconfigure(0, weight=1)
        self.playback_progress_var = tk.DoubleVar(value=0)
        self.playback_progress_scale = ttk.Scale(
            playback_ctrl, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.playback_progress_var, command=self._on_playback_seek,
        )
        self.playback_progress_scale.grid(row=0, column=0, sticky='ew', padx=8, pady=(8, 2))
        self.playback_time_var = tk.StringVar(value='00:00 / 00:00')
        tk.Label(playback_ctrl, textvariable=self.playback_time_var,
                 bg=C['card'], fg=C['text_dim'], font=('Courier', 10, 'bold')).grid(row=1, column=0, pady=(0, 6))
    
    def create_status_bar(self, parent):
        """创建底部状态栏 - 深色主题"""
        C = self.C
        bar = tk.Frame(parent, bg=C['header_bg'], height=26)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)

        self._status_dot = tk.Label(
            bar, text='⬤', bg=C['header_bg'], fg=C['good'],
            font=('Arial', 7), padx=8,
        )
        self._status_dot.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value='就绪')
        tk.Label(
            bar, textvariable=self.status_var,
            bg=C['header_bg'], fg=C['text_dim'],
            font=('Arial', 9), anchor='w',
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def switch_mode(self, mode: str):
        """切换页面模式：record / playback"""
        if mode not in ('record', 'playback'):
            return
        if mode == self.current_mode and self.current_mode is not None:
            return

        if self.is_recording and mode != 'record':
            messagebox.showwarning("提示", "录制进行中，停止录制后再切换到回放模式")
            return

        if self.is_playback_running and mode != 'playback':
            self._stop_playback()

        C = self.C
        if mode == 'record':
            self.playback_controls_frame.pack_forget()
            self.record_controls_frame.pack(fill=tk.BOTH, expand=True)
            self.playback_right.place_forget()
            self.record_right.place(x=0, y=0, relwidth=1, relheight=1)
            self.nav_record_btn.config(bg=C['accent'], fg='white')
            self.nav_playback_btn.config(bg=C['card'], fg=C['text_dim'])
            self.status_var.set("录制模式")
        else:
            self.record_controls_frame.pack_forget()
            self.playback_controls_frame.pack(fill=tk.BOTH, expand=True)
            self.record_right.place_forget()
            self.playback_right.place(x=0, y=0, relwidth=1, relheight=1)
            self.nav_record_btn.config(bg=C['card'], fg=C['text_dim'])
            self.nav_playback_btn.config(bg=C['accent'], fg='white')
            self._scan_playback_recordings()
            self.status_var.set("回放模式")

        self.current_mode = mode

    def _browse_playback_dir(self):
        """选择回放目录"""
        directory = filedialog.askdirectory(initialdir=self.recordings_dir)
        if directory:
            self.recordings_dir = Path(directory)
            self.playback_dir_var.set(str(self.recordings_dir))
            self._scan_playback_recordings()

    def _scan_playback_recordings(self):
        """扫描回放目录"""
        try:
            self.recordings_dir = Path(self.playback_dir_var.get().strip())
            self.playback_recordings = scan_recordings(self.recordings_dir)
            self.playback_listbox.delete(0, tk.END)
            for rec in self.playback_recordings:
                duration_str = f"{int(rec.duration_sec // 60)}:{int(rec.duration_sec % 60):02d}"
                videos_str = f"{rec.video_count}" + ("+IMU" if rec.has_imu else "")
                self.playback_listbox.insert(tk.END, f"{rec.name} | {videos_str} | {duration_str}")
            self.selected_playback_recording = None
            self.play_btn.config(state='disabled')
            self.playback_detail_var.set("请选择一个录制")
            self.status_var.set(f"回放: 找到 {len(self.playback_recordings)} 个录制")
        except Exception as e:
            self.status_var.set(f"回放扫描失败: {e}")

    def _on_playback_select(self, event=None):
        """选择回放条目"""
        selection = self.playback_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        if 0 <= idx < len(self.playback_recordings):
            rec = self.playback_recordings[idx]
            self.selected_playback_recording = rec
            detail = (
                f"名称: {rec.name}\n"
                f"视频数: {rec.video_count}\n"
                f"帧数: {rec.frame_count}\n"
                f"时长: {rec.duration_sec:.1f}s\n"
                f"IMU: {'有' if rec.has_imu else '无'}"
            )
            self.playback_detail_var.set(detail)
            if not self.is_playback_running:
                self.play_btn.config(state='normal')

    def _on_playback_double_click(self, event=None):
        if self.selected_playback_recording:
            self._start_playback()

    def _on_playback_speed_change(self, event=None):
        speed_str = self.playback_speed_var.get()
        try:
            self.playback_speed = float(speed_str.replace('x', '').strip())
        except Exception:
            self.playback_speed = 1.0

    def _start_playback(self):
        """开始回放"""
        if not self.selected_playback_recording:
            return
        if self.is_playback_running:
            return

        rec = self.selected_playback_recording
        video_files = sorted(rec.path.glob('*.mp4'))
        imu_file = rec.path / 'imu_data.txt'
        if not video_files:
            messagebox.showerror("错误", "未找到视频文件")
            return

        self._stop_playback(reset_status=False)

        self.player = VideoPlayer(video_files, fps=30.0, grid_width=800)
        self.imu_viz = None
        self.imu_data = []
        if imu_file.exists():
            self.imu_data = load_imu_data(imu_file)
            if self.imu_data:
                self.imu_viz = IMUVisualizer(self.imu_data, width=800, height=360)

        self.is_playback_running = True
        self.playback_stop_flag = False
        self.playback_paused = False

        self.play_btn.config(state='disabled')
        self.pause_btn.config(state='normal', text='⏸ 暂停')
        self.stop_btn.config(state='normal')
        self.status_var.set(f"正在回放: {rec.name}")

        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def _stop_playback(self, reset_status=True):
        """停止回放"""
        self.playback_stop_flag = True
        self.playback_paused = False

        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)

        if self.player:
            self.player.release()
            self.player = None

        self.imu_viz = None
        self.imu_data = []
        self.is_playback_running = False

        self.pause_btn.config(state='disabled', text='⏸ 暂停')
        self.stop_btn.config(state='disabled')
        self.play_btn.config(state='normal' if self.selected_playback_recording else 'disabled')

        self.playback_video_canvas.delete('all')
        self.playback_imu_canvas.delete('all')
        self.playback_progress_var.set(0)
        self.playback_time_var.set('00:00 / 00:00')

        if reset_status and self.current_mode == 'playback':
            self.status_var.set('回放已停止')

    def _toggle_playback_pause(self):
        if not self.is_playback_running:
            return
        self.playback_paused = not self.playback_paused
        self.pause_btn.config(text='▶ 继续' if self.playback_paused else '⏸ 暂停')
        self.status_var.set('回放暂停' if self.playback_paused else '正在回放')

    def _seek_playback_relative(self, seconds):
        if not self.player:
            return
        frame_delta = int(seconds * self.player.fps)
        self.player.seek(self.player.current_frame + frame_delta)
        if self.imu_viz:
            self.imu_viz.reset()

    def _on_playback_seek(self, value):
        if not self.player:
            return
        frame = int(float(value) / 100.0 * max(1, self.player.total_frames - 1))
        self.player.seek(frame)
        if self.imu_viz:
            self.imu_viz.reset()

    def _playback_loop(self):
        """回放循环（后台线程）"""
        if not self.player:
            return

        fps = self.player.fps or 30.0
        frame_interval = 1.0 / max(1e-6, fps)
        last_frame_time = time.perf_counter()

        while (
            not self.playback_stop_flag
            and self.player
            and self.player.current_frame < self.player.total_frames
        ):
            if self.playback_paused:
                time.sleep(0.03)
                last_frame_time = time.perf_counter()
                continue

            now = time.perf_counter()
            target = frame_interval / max(0.01, self.playback_speed)
            if now - last_frame_time < target:
                time.sleep(0.001)
                continue

            video_frame = self.player.read_combined_frame()
            video_time = self.player.get_current_time()

            if video_frame is not None:
                self._update_playback_video_canvas(video_frame)

            if self.imu_viz:
                imu_frame = self.imu_viz.render(video_time)
                self._update_playback_imu_canvas(imu_frame)

            total_time = self.player.total_frames / max(1e-6, fps)
            progress = (self.player.current_frame / max(1, self.player.total_frames)) * 100
            self._update_playback_progress(progress, video_time, total_time)

            last_frame_time = now

        self.root.after(0, self._on_playback_end)

    def _update_playback_video_canvas(self, frame):
        def update():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            canvas_w = self.playback_video_canvas.winfo_width()
            canvas_h = self.playback_video_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                scale = min(canvas_w / img.width, canvas_h / img.height)
                img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.NEAREST)
            self._playback_video_photo = ImageTk.PhotoImage(img)
            self.playback_video_canvas.delete('all')
            self.playback_video_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._playback_video_photo)
        self.root.after(0, update)

    def _update_playback_imu_canvas(self, frame):
        def update():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            canvas_w = self.playback_imu_canvas.winfo_width()
            canvas_h = self.playback_imu_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                scale = min(canvas_w / img.width, canvas_h / img.height)
                img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.NEAREST)
            self._playback_imu_photo = ImageTk.PhotoImage(img)
            self.playback_imu_canvas.delete('all')
            self.playback_imu_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._playback_imu_photo)
        self.root.after(0, update)

    def _update_playback_progress(self, progress, current_time, total_time):
        def update():
            self.playback_progress_var.set(progress)
            cur_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
            tot_str = f"{int(total_time // 60):02d}:{int(total_time % 60):02d}"
            self.playback_time_var.set(f"{cur_str} / {tot_str}")
        self.root.after(0, update)

    def _on_playback_end(self):
        if self.playback_stop_flag:
            return
        if self.player:
            self.player.release()
            self.player = None
        self.imu_viz = None
        self.imu_data = []
        self.is_playback_running = False
        self.pause_btn.config(state='disabled', text='⏸ 暂停')
        self.stop_btn.config(state='disabled')
        self.play_btn.config(state='normal' if self.selected_playback_recording else 'disabled')
        if self.current_mode == 'playback':
            self.status_var.set('回放完成')
    
    def scan_cameras(self):
        """扫描可用相机"""
        self.status_var.set("正在扫描相机...")
        self.root.update()
        
        available = list_cameras()
        
        if available:
            ordered_cfg = self._ordered_video_ids_from_config()
            ordered = [vid for vid in ordered_cfg if vid in available]
            remaining = [vid for vid in available if vid not in ordered]
            final_ids = ordered + remaining
            self.camera_ids_var.set(','.join(map(str, final_ids)))
            self.status_var.set(f"找到 {len(available)} 个相机(已按config排序): {final_ids}")
        else:
            self.status_var.set("未找到可用相机")
            messagebox.showwarning("警告", "未找到可用的相机设备")
    
    def select_output_dir(self):
        """选择输出目录"""
        directory = filedialog.askdirectory(initialdir=self.output_dir)
        if directory:
            self.output_dir = Path(directory)
            self.output_dir_var.set(str(self.output_dir))
    
    def toggle_recording(self):
        """切换录制状态"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """开始录制 - 多进程架构"""
        try:
            # 解析相机ID
            camera_ids_str = self.camera_ids_var.get().strip()
            if not camera_ids_str:
                messagebox.showerror("错误", "请输入相机ID")
                return
            
            camera_ids = [int(x.strip()) for x in camera_ids_str.split(',')]
            
            if not camera_ids:
                messagebox.showerror("错误", "没有有效的相机ID")
                return

            # 按 config/config.yaml 中 cam 顺序重排 /dev/video 列表
            cfg_order = self._ordered_video_ids_from_config()
            in_cfg = [vid for vid in cfg_order if vid in camera_ids]
            extra = [vid for vid in camera_ids if vid not in in_cfg]
            camera_ids = in_cfg + extra
            
            # 创建会话目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session_dir = self.output_dir / f"recording_{timestamp}"
            self.current_session_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化视频录制器
            width = self.width_var.get()
            height = self.height_var.get()
            fps = self.fps_var.get()
            
            self.status_var.set("正在启动相机进程...")
            self.root.update()
            
            # 创建多进程相机管理器
            print(f"\n=== 启动多进程相机系统 ===")
            print(f"目标帧率: {fps} fps, 分辨率: {width}x{height}")
            print(f"相机设备顺序(/dev/video): {camera_ids}")
            
            self.camera_manager = MultiCameraManager(camera_ids, width, height, fps)
            
            # 启动所有相机进程
            self.camera_ids = self.camera_manager.start_all(self.current_session_dir)
            
            if not self.camera_ids:
                messagebox.showerror("错误", "没有成功初始化任何相机")
                self.cleanup_recording()
                return
            
            print(f"成功启动 {len(self.camera_ids)} 个相机进程: {self.camera_ids}")
            
            # 初始化IMU录制器
            self.status_var.set("正在初始化IMU...")
            self.root.update()
            
            imu_port = self.imu_port_var.get()
            imu_baud = self.imu_baud_var.get()
            
            self.imu_recorder = IMURecorder(imu_port, imu_baud)
            if not self.imu_recorder.start(self.current_session_dir):
                messagebox.showwarning("警告", "IMU初始化失败，将仅录制视频")
                self.imu_recorder = None
            
            # 禁用设置控件
            self.set_controls_state('disabled')
            
            # 显示倒计时并同步启动录制
            self._show_countdown_and_start()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"启动录制失败: {e}")
            self.cleanup_recording()
    
    def _show_countdown_and_start(self):
        """显示深色主题倒计时弹窗并同步启动所有相机"""
        C = self.C
        countdown_window = tk.Toplevel(self.root)
        countdown_window.title("准备录制")
        countdown_window.geometry("300x200")
        countdown_window.resizable(False, False)
        countdown_window.transient(self.root)
        countdown_window.grab_set()
        countdown_window.configure(bg=C['card'])

        # 居中显示
        countdown_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 300) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 200) // 2
        countdown_window.geometry(f"+{x}+{y}")

        tk.Label(
            countdown_window, text="相机同步中...",
            bg=C['card'], fg=C['text_dim'], font=('Arial', 12),
        ).pack(pady=10)

        countdown_var = tk.StringVar(value="3")
        tk.Label(
            countdown_window, textvariable=countdown_var,
            bg=C['card'], fg=C['accent'],
            font=('Arial', 72, 'bold'),
        ).pack(pady=4)

        status_var = tk.StringVar(value="准备同步所有相机...")
        tk.Label(
            countdown_window, textvariable=status_var,
            bg=C['card'], fg=C['text_dim'], font=('Arial', 9),
        ).pack(pady=8)

        def do_countdown(count):
            if count > 0:
                countdown_var.set(str(count))
                status_var.set("即将开始录制...")
                countdown_window.after(1000, lambda: do_countdown(count - 1))
            else:
                countdown_var.set("GO!")
                status_var.set("同步启动所有相机!")
                countdown_window.after(300, finish_countdown)

        def finish_countdown():
            countdown_window.destroy()
            self._finalize_recording_start()

        countdown_window.after(100, lambda: do_countdown(3))
    
    def _finalize_recording_start(self):
        """完成录制启动（倒计时结束后）"""
        # 更新UI
        self.is_recording = True
        self.record_button.config(
            text='■  停止录制',
            bg=self.C['stop'], activebackground=self.C['stop_hover'],
        )
        self._timer_label.config(fg=self.C['rec'])
        self._status_dot.config(fg=self.C['rec'])
        self.status_var.set(f"正在录制到: {self.current_session_dir.name}")
        
        # 开始所有相机录制（带同步时间点）
        print("发送同步开始录制信号...")
        sync_time = self.camera_manager.begin_recording(countdown_seconds=0.1)
        print(f"同步时间点: {sync_time:.6f}")

        # IMU 使用同一个 perf_counter 同步点，并补偿半帧对齐相机首帧时刻
        if self.imu_recorder:
            frame_period = 1.0 / max(1, int(self.fps_var.get()))
            imu_sync_time = sync_time + 0.5 * frame_period
            self.imu_recorder.reset_start_time(imu_sync_time)
        
        # 开始UI更新
        self.recording_start_time = time.perf_counter()
        self.schedule_update()
        
        # 启动IMU数据刷新循环
        self.schedule_imu_flush()
        
        print(f"所有系统已启动，开始多进程录制 {len(self.camera_ids)} 个相机\n")
    
    def schedule_imu_flush(self):
        """调度IMU数据刷新"""
        if not self.is_recording:
            return
        
        # 刷新IMU数据到文件
        if self.imu_recorder:
            self.imu_recorder.flush_data()
        
        # 每50ms刷新一次IMU数据
        self.imu_flush_timer = self.root.after(50, self.schedule_imu_flush)
    
    def schedule_update(self):
        """调度UI更新"""
        if not self.is_recording:
            return
        
        # 更新录制时间
        elapsed = time.perf_counter() - self.recording_start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        self.record_time_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # 更新预览
        self.update_preview()
        
        # 更新IMU显示
        if self.imu_recorder:
            self.update_imu_display()
        
        # 约 8fps 预览：更流畅，同时避免 UI 线程过载
        self.update_timer = self.root.after(self.preview_interval_ms, self.schedule_update)
    
    def update_preview(self):
        """更新相机预览 - 从多进程共享内存读取，显示所有相机网格"""
        if not self.camera_manager:
            return
        
        try:
            # 从多进程管理器获取所有预览帧
            preview_frames = self.camera_manager.get_all_previews()
            all_stats = self.camera_manager.get_all_stats()
            
            if not preview_frames:
                return
            
            num_cams = len(self.camera_ids)

            # 固定缩略图尺寸：画面更清晰，标签更易读
            if num_cams >= 8:
                thumb_size = (240, 135)  # 降低拼接分辨率，减轻 UI 压力
                font_scale = 0.48
            elif num_cams <= 2:
                thumb_size = (360, 240)
                font_scale = 0.6
            else:
                thumb_size = (280, 180)
                font_scale = 0.55

            def build_tile(cam_id):
                frame = preview_frames.get(cam_id)
                stats = all_stats.get(cam_id)

                if frame is None:
                    tile = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
                    cv2.putText(tile, self._format_cam_label(cam_id), (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (120, 220, 255), 1)
                    cv2.putText(tile, "NO SIGNAL", (8, thumb_size[1] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 220), 2)
                else:
                    tile = cv2.resize(frame, thumb_size, interpolation=cv2.INTER_LINEAR)

                    # 顶部信息条（更美观）
                    cv2.rectangle(tile, (0, 0), (thumb_size[0], 26), (18, 18, 28), -1)
                    label = self._format_cam_label(cam_id)
                    if stats:
                        label += f"  {stats['hw_fps']:.0f}fps"
                    cv2.putText(tile, label, (8, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (125, 220, 255), 1)

                # 卡片边框
                tile = cv2.copyMakeBorder(tile, 2, 2, 2, 2,
                                          borderType=cv2.BORDER_CONSTANT,
                                          value=(32, 32, 44))
                return tile

            # 构造所有 tile（即使没帧也显示占位，布局稳定）
            tiles = [build_tile(cam_id) for cam_id in self.camera_ids]

            # 8 路固定两栏 4+4：左列前4个，右列后4个（避免上下 4+4）
            if num_cams == 8:
                left_col = cv2.vconcat(tiles[:4])
                right_col = cv2.vconcat(tiles[4:8])
                combined = cv2.hconcat([left_col, right_col])
            else:
                # 其它数量回退到常规网格
                if num_cams <= 2:
                    cols = num_cams
                elif num_cams <= 4:
                    cols = 2
                elif num_cams <= 9:
                    cols = 3
                else:
                    cols = 4

                rows_needed = (len(tiles) + cols - 1) // cols
                h, w = tiles[0].shape[:2]
                while len(tiles) < rows_needed * cols:
                    tiles.append(np.zeros((h, w, 3), dtype=np.uint8))

                row_imgs = []
                for r in range(rows_needed):
                    row_frames = tiles[r * cols: (r + 1) * cols]
                    row_imgs.append(cv2.hconcat(row_frames))
                combined = cv2.vconcat(row_imgs)
            
            # 缩放适应canvas
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                ch, cw = combined.shape[:2]
                scale = min(canvas_width / cw, canvas_height / ch) * 0.95
                new_w = max(1, int(cw * scale))
                new_h = max(1, int(ch * scale))
                
                resized = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                photo = ImageTk.PhotoImage(image=img)
                
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(
                    canvas_width // 2, canvas_height // 2, image=photo
                )
                self.preview_canvas.image = photo
                    
        except Exception as e:
            print(f"预览更新错误: {e}")
    
    def update_imu_display(self):
        """更新IMU数据显示"""
        if not self.imu_recorder:
            return
        data = self.imu_recorder.get_latest_data()
        if not data:
            return

        fmts = {
            'roll':      lambda v: f'{v:+8.2f}°',
            'pitch':     lambda v: f'{v:+8.2f}°',
            'yaw':       lambda v: f'{v:+8.2f}°',
            'acc_x':     lambda v: f'{v:+7.3f}g',
            'acc_y':     lambda v: f'{v:+7.3f}g',
            'acc_z':     lambda v: f'{v:+7.3f}g',
            'gyro_x':    lambda v: f'{v:+7.2f}°/s',
            'gyro_y':    lambda v: f'{v:+7.2f}°/s',
            'gyro_z':    lambda v: f'{v:+7.2f}°/s',
            'norm_mag_x': lambda v: f'{v:+7.3f}',
            'norm_mag_y': lambda v: f'{v:+7.3f}',
            'norm_mag_z': lambda v: f'{v:+7.3f}',
        }
        for key, lbl in self._imu_val_labels.items():
            if key in data and key in fmts:
                lbl.config(text=fmts[key](data[key]))
        self._imu_temp_lbl.config(text=f"{data.get('sensor_temp', 0):.1f} °C")
        self._imu_pkt_lbl.config(text=str(self.imu_recorder.data_count))
    
    def stop_recording(self):
        """停止录制 - 多进程架构"""
        self.is_recording = False
        
        # 取消定时器
        if hasattr(self, 'imu_flush_timer') and self.imu_flush_timer:
            self.root.after_cancel(self.imu_flush_timer)
        if hasattr(self, 'update_timer') and self.update_timer:
            self.root.after_cancel(self.update_timer)
        
        self.status_var.set("正在停止相机进程...")
        self.root.update()
        
        # 获取最终统计信息
        summary_lines = []
        all_stats = {}
        target_fps = self.fps_var.get()
        
        if self.camera_manager:
            all_stats = self.camera_manager.get_all_stats()
            for cam_id in self.camera_ids:
                stats = all_stats.get(cam_id)
                if stats:
                    line = (f"{self._format_cam_label(cam_id)}: {stats['frames']}帧, "
                            f"新帧{stats['new_frames']}({stats['hw_fps']:.1f}fps), "
                            f"重复{stats['duplicated']}")
                    print(line)
                    summary_lines.append(line)
            
            # 停止所有相机进程
            print("\n停止所有相机进程...")
            self.camera_manager.stop_all()
        
        if self.imu_recorder:
            self.imu_recorder.stop()
            print(f"IMU: {self.imu_recorder.data_count} 个数据包")

        # 将文件名从 /dev/video 编号转换为逻辑 cam 编号（0..N）
        self._rename_recorded_videos_to_cam_ids()
        
        # 保存录制信息
        self.save_recording_info(all_stats)
        
        saved_dir = self.current_session_dir
        
        # 检查是否有相机帧率不足
        low_fps_cams = []
        for cam_id, stats in all_stats.items():
            if stats and stats['hw_fps'] < target_fps * 0.9:
                low_fps_cams.append(f"{self._format_cam_label(cam_id)}: {stats['hw_fps']:.1f}fps")
        
        # 清理
        self.cleanup_recording()
        
        # 更新UI
        self.record_button.config(
            text='●  开始录制',
            bg=self.C['rec'], activebackground=self.C['rec_hover'],
        )
        self._timer_label.config(fg=self.C['text_dim'])
        self._status_dot.config(fg=self.C['good'])
        self.status_var.set(f"录制完成: {saved_dir}")
        self.record_time_var.set("00:00:00")
        
        # 启用设置控件
        self.set_controls_state('normal')
        
        msg = f"录制已保存到:\n{saved_dir}\n\n"
        msg += "\n".join(summary_lines)
        
        if low_fps_cams:
            msg += f"\n\n⚠️ 以下相机硬件帧率不足:\n" + "\n".join(low_fps_cams)
            msg += f"\n建议: 降低分辨率或减少相机数量"
        
        messagebox.showinfo("完成", msg)
    
    def save_recording_info(self, all_stats=None):
        """保存录制信息"""
        if not self.current_session_dir:
            return
        
        try:
            info_file = self.current_session_dir / "recording_info.txt"
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"录制时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"架构: 多进程 (每个相机独立进程)\n")
                f.write(f"相机数量: {len(self.camera_ids)}\n")
                f.write(f"相机设备ID(/dev/video): {self.camera_ids}\n")
                f.write(f"分辨率: {self.width_var.get()}x{self.height_var.get()}\n")
                f.write(f"帧率: {self.fps_var.get()} fps\n")
                f.write(f"IMU端口: {self.imu_port_var.get()}\n")
                f.write(f"IMU波特率: {self.imu_baud_var.get()}\n")
                f.write("\n录制统计:\n")
                f.write(f"cam->video 映射: {dict(sorted(self.cam_to_video.items()))}\n")
                
                if all_stats:
                    for cam_id, stats in all_stats.items():
                        if stats:
                            f.write(f"  {self._format_cam_label(cam_id)}: "
                                   f"{stats['frames']} 帧, "
                                   f"新帧 {stats['new_frames']} (硬件{stats['hw_fps']:.1f}fps), "
                                   f"重复帧 {stats['duplicated']}, "
                                   f"黑帧 {stats['dropped']}\n")
                
                if self.imu_recorder:
                    f.write(f"  IMU数据: {self.imu_recorder.data_count} 个数据包\n")
        
        except Exception as e:
            print(f"保存录制信息失败: {e}")
    
    def cleanup_recording(self):
        """清理录制资源"""
        if self.camera_manager:
            self.camera_manager.stop_all()
            self.camera_manager = None
        self.camera_ids = []
        self.imu_recorder = None
        self.current_session_dir = None
    
    def set_controls_state(self, state):
        """设置控件启用/禁用状态"""
        tk_state = 'disabled' if state == 'disabled' else 'normal'
        for w in getattr(self, '_setting_widgets', []):
            try:
                w.config(state=tk_state)
            except Exception:
                pass
    
    def on_closing(self):
        """窗口关闭事件"""
        if self.is_playback_running:
            self._stop_playback(reset_status=False)

        if self.is_recording:
            if messagebox.askokcancel("退出", "正在录制中，确定要退出吗？"):
                self.stop_recording()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """主函数"""
    # 检查依赖
    try:
        _ = Image
        _ = ImageTk
    except Exception:
        print("错误: 需要安装 Pillow 库")
        print("请运行: uv add pillow")
        return
    
    print("=" * 50)
    print("多相机+IMU数据采集系统 (多进程架构)")
    print("每个相机在独立进程中运行，绑过 Python GIL")
    print("=" * 50)
    
    # 创建GUI
    root = tk.Tk()
    app = RecorderGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    # macOS/Windows 需要在 main guard 内设置 spawn 方式
    # 这必须在创建任何 Process 之前调用
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # 已经设置过了
    
    # Windows 打包需要 freeze_support
    mp.freeze_support()
    
    main()
