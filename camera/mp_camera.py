#!/usr/bin/env python3
"""
多进程相机管理模块 - 每个相机运行在独立进程中，完全绑过Python GIL
"""

import cv2
import numpy as np
import time
import os
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
from ctypes import c_uint64, c_double, c_bool


def _open_camera_capture(camera_id: int, width: int, height: int, fps: int):
    """
    打开相机并按正确顺序设置格式。

    关键：V4L2 必须在打开后「先设 FOURCC」，再设分辨率/FPS。
    若先设分辨率，驱动会以默认格式(YUYV 15fps)初始化，后续 set() 无效。
    """
    backend = cv2.CAP_V4L2 if (os.name == 'posix' and hasattr(cv2, 'CAP_V4L2')) else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        # V4L2 失败时回退默认后端
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        return cap

    # 1. 先设 FOURCC（MJPG 压缩，USB 带宽从 ~150MB/s 降至 ~5MB/s）
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # 2. 再设分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # 3. 最后设 FPS（驱动需要知道分辨率才能协商帧率）
    cap.set(cv2.CAP_PROP_FPS, fps)
    # 4. 双缓冲：2 个 DMA 缓冲区
    #    BUFFERSIZE=1 时只有 1 个缓冲区，app 读帧时相机无处写入，
    #    造成每帧都要等 app 归还缓冲区，实际帧率砍半(30→15fps)。
    #    BUFFERSIZE=2 实现双缓冲：相机填一个，app 同时读另一个。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    return cap


def _log_camera_config(camera_id: int, cap):
    """打印相机实际协商到的参数，便于排查帧率异常。"""
    actual_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = ''.join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    backend   = cap.getBackendName()
    print(
        f"[Cam{camera_id}] 后端={backend}  格式={fourcc_str}  "
        f"分辨率={actual_w}x{actual_h}  FPS={actual_fps:.1f}"
    )


def camera_worker_process(
    camera_id: int,
    width: int,
    height: int,
    fps: int,
    rotate_180: bool,
    output_path: str,
    start_event,
    stop_event,
    ready_event,
    sync_start_time,  # mp.Value('d') for synchronized start time
    shm_name: str,
    stats_array,  # mp.Array for sharing statistics
    error_msg,    # mp.Array for error message
):
    """
    相机工作进程 - 在独立进程中完成采集和写入
    
    每个相机一个进程，采集→写入完全在进程内完成，无需跨进程传输视频数据。
    只有预览帧通过共享内存传递回主进程。
    
    同步机制：
    - ready_event: 进程初始化完成
    - start_event: 开始录制信号
    - sync_start_time: 统一的开始时间点（精确同步第一帧）
    
    stats_array 布局:
        [0] = frame_count (写入磁盘的帧数)
        [1] = new_frames (硬件返回的新帧数)
        [2] = duplicated_frames (复制帧数)
        [3] = dropped_frames (黑帧数)
        [4] = start_time (录制开始时间戳)
        [5] = is_running (进程是否在录制中)
        [6] = preview_seq (预览帧序列号，主进程用于检测新帧)
    """
    
    try:
        # 打开相机（内部按顺序设置 FOURCC → 分辨率 → FPS）
        cap = _open_camera_capture(camera_id, width, height, fps)
        if not cap.isOpened():
            error_msg.value = f"无法打开相机 {camera_id}".encode('utf-8')
            ready_event.set()
            return
        
        # 打印实际协商参数（若帧率仍显示 15，说明相机不支持 MJPG 高帧率）
        _log_camera_config(camera_id, cap)
        
        # 读取第一帧验证相机工作
        ret = False
        for _ in range(30):
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.05)
        
        if not ret:
            error_msg.value = f"相机 {camera_id} 无法读取帧".encode('utf-8')
            cap.release()
            ready_event.set()
            return
        
        # 验证实际分辨率（set 不一定生效）
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != width or actual_h != height:
            print(f"[Cam{camera_id}] 警告: 请求 {width}x{height}, 实际 {actual_w}x{actual_h}")
            width, height = actual_w, actual_h
        
        # 连接共享内存（用于预览帧传输）
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            preview_buffer = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)
        except Exception as e:
            error_msg.value = f"共享内存连接失败: {e}".encode('utf-8')
            cap.release()
            ready_event.set()
            return
        
        # 创建视频写入器
        output_file = Path(output_path) / f"{camera_id}.mp4"
        # Linux 多相机场景下 mp4v 通常编码更轻，先尝试 mp4v 再回退 avc1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height), True)
        
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            error_msg.value = f"无法创建视频文件: {output_file}".encode('utf-8')
            cap.release()
            shm.close()
            ready_event.set()
            return
        
        # 初始化完成，通知主进程
        error_msg.value = b''  # 清空错误信息表示成功
        ready_event.set()
        
        # 等待开始信号
        start_event.wait()
        
        # 预热：清空相机缓冲区，确保读取的是最新帧
        for _ in range(5):
            cap.grab()
        
        # 等待统一的开始时间点（精确同步所有相机的第一帧）
        target_start = sync_start_time.value
        if target_start > 0:
            wait_time = target_start - time.perf_counter()
            if wait_time > 0:
                time.sleep(wait_time)
        
        # 初始化计数器
        frame_count = 0
        new_frames = 0
        duplicated = 0
        preview_seq = 0

        # 起始时间放到“首帧成功采集时”再写入，避免统计起点提前
        actual_start = 0.0
        stats_array[4] = 0.0
        stats_array[5] = 1  # is_running = True
        
        # ================================================================
        # 核心录制循环：由 cap.grab() 自然阻塞驱动帧率
        #
        # 【关键设计】cap.grab() 是阻塞式调用，它会等到相机硬件送出下一帧
        # 才返回，天然决定了采集帧率。在其上再加 sleep 会造成双重等待，
        # 导致实际帧率砍半（30fps → 15fps）。因此循环中不加任何 sleep。
        # ================================================================
        while not stop_event.is_set():
            # grab() 阻塞等待硬件下一帧——这是帧率的唯一驱动者
            ret = cap.grab()
            if not ret:
                # 相机掉线或出错，短暂等待后重试
                time.sleep(0.005)
                continue
            
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            
            # 确保分辨率匹配
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            # 安装方向倒置的相机：旋转 180° 后再写盘/预览（180° 旋转不改变分辨率）
            if rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            new_frames += 1
            frame_count += 1

            if actual_start == 0.0:
                actual_start = time.perf_counter()
                stats_array[4] = actual_start
            
            # 写入视频
            writer.write(frame)
            
            # 更新预览帧（每3帧一次，减少共享内存开销）
            if frame_count % 3 == 0:
                np.copyto(preview_buffer, frame)
                preview_seq += 1
                stats_array[6] = preview_seq
            
            # 低频更新统计
            if frame_count % 10 == 0:
                stats_array[0] = frame_count
                stats_array[1] = new_frames
                stats_array[2] = 0  # 不再填充重复帧
                stats_array[3] = 0
        
        # 清理资源：写入最终统计
        stats_array[0] = frame_count
        stats_array[1] = new_frames
        stats_array[2] = 0
        stats_array[3] = 0
        stats_array[5] = 0  # is_running = False
        writer.release()
        cap.release()
        shm.close()
        
    except Exception as e:
        import traceback
        error_msg.value = f"进程异常: {e}\n{traceback.format_exc()}".encode('utf-8')[:1024]
        if 'cap' in locals():
            cap.release()
        if 'writer' in locals():
            writer.release()
        if 'shm' in locals():
            shm.close()


class MultiProcessCamera:
    """
    多进程相机管理器 - 每个相机在独立进程中运行
    
    主要特点：
    1. 完全绑过 Python GIL，真正并行
    2. 采集和写入在同一进程，无跨进程大数据传输
    3. 只通过共享内存传递预览帧 (~1MB)
    4. 使用进程同步原语协调开始/停止
    """
    
    def __init__(self, camera_id: int, width: int = 640, height: int = 480,
                 fps: int = 30, rotate_180: bool = False):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate_180 = rotate_180

        self.process = None
        self.start_event = None
        self.stop_event = None
        self.ready_event = None
        self.sync_start_time = None  # 同步开始时间
        self.stats_array = None
        self.error_msg = None
        self.shm = None
        self.preview_buffer = None
        self._last_preview_seq = 0
        
    def start(self, output_path: Path) -> bool:
        """启动相机进程"""
        try:
            # 创建共享内存用于预览帧
            frame_size = self.width * self.height * 3
            self.shm = shared_memory.SharedMemory(create=True, size=frame_size)
            self.preview_buffer = np.ndarray(
                (self.height, self.width, 3), 
                dtype=np.uint8, 
                buffer=self.shm.buf
            )
            self.preview_buffer.fill(0)
            
            # 创建同步事件
            self.start_event = mp.Event()
            self.stop_event = mp.Event()
            self.ready_event = mp.Event()
            self.sync_start_time = mp.Value('d', 0.0)  # 同步开始时间
            
            # 创建共享统计数组
            # [frame_count, new_frames, duplicated, dropped, start_time, is_running, preview_seq]
            self.stats_array = mp.Array('d', 7)
            
            # 创建错误消息缓冲区
            self.error_msg = mp.Array('c', 1024)
            
            # 启动工作进程
            self.process = mp.Process(
                target=camera_worker_process,
                args=(
                    self.camera_id,
                    self.width,
                    self.height,
                    self.fps,
                    self.rotate_180,
                    str(output_path),
                    self.start_event,
                    self.stop_event,
                    self.ready_event,
                    self.sync_start_time,
                    self.shm.name,
                    self.stats_array,
                    self.error_msg,
                ),
                daemon=True
            )
            self.process.start()
            
            # 等待进程初始化完成
            if not self.ready_event.wait(timeout=10.0):
                self._cleanup()
                print(f"相机 {self.camera_id}: 初始化超时")
                return False
            
            # 检查是否有错误
            error = self.error_msg.value.decode('utf-8').strip('\x00')
            if error:
                self._cleanup()
                print(f"相机 {self.camera_id}: {error}")
                return False
            
            return True
            
        except Exception as e:
            print(f"相机 {self.camera_id} 启动失败: {e}")
            self._cleanup()
            return False
    
    def begin_recording(self, sync_time: float = 0.0):
        """开始录制（发送信号给工作进程）
        
        Args:
            sync_time: 同步开始时间点，所有相机会等待到这个时间再开始
        """
        if self.sync_start_time:
            self.sync_start_time.value = sync_time
        if self.start_event:
            self.start_event.set()
    
    def stop(self):
        """停止录制并等待进程结束"""
        if self.stop_event:
            self.stop_event.set()
        
        if self.process and self.process.is_alive():
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                print(f"相机 {self.camera_id}: 强制终止进程")
                self.process.terminate()
                self.process.join(timeout=2.0)
        
        self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        try:
            if self.shm:
                self.shm.close()
                self.shm.unlink()
                self.shm = None
        except Exception:
            pass
        self.preview_buffer = None
    
    def get_preview_frame(self):
        """获取预览帧（从共享内存读取）"""
        if self.preview_buffer is None:
            return False, None
        
        try:
            current_seq = int(self.stats_array[6])
            if current_seq > self._last_preview_seq:
                self._last_preview_seq = current_seq
                # 复制帧数据（避免共享内存访问冲突）
                frame = self.preview_buffer.copy()
                return True, frame
            elif current_seq > 0:
                # 返回最近的帧（即使没有更新）
                return True, self.preview_buffer.copy()
        except Exception:
            pass
        
        return False, None
    
    def get_stats(self):
        """获取录制统计"""
        if self.stats_array is None:
            return None
        
        try:
            frame_count = int(self.stats_array[0])
            new_frames = int(self.stats_array[1])
            start_time = float(self.stats_array[4])
            is_running = bool(self.stats_array[5])
            
            if start_time > 0:
                duration = time.perf_counter() - start_time
                hw_fps = new_frames / duration if duration > 0 else 0
            else:
                duration = 0
                hw_fps = 0
            
            return {
                'frames': frame_count,
                'new_frames': new_frames,
                'duplicated': 0,
                'dropped': 0,
                'duration': duration,
                'fps': hw_fps,
                'hw_fps': hw_fps,
                'is_running': is_running,
            }
        except Exception:
            return None
    
    @property
    def is_recording(self):
        """检查是否在录制中"""
        if self.stats_array:
            return bool(self.stats_array[5])
        return False


class MultiCameraManager:
    """
    多相机管理器 - 协调多个 MultiProcessCamera
    
    统一启动/停止所有相机，保证同步开始录制
    """
    
    def __init__(self, camera_ids: list, width: int = 640, height: int = 480,
                 fps: int = 30, rotate_ids=None):
        # rotate_ids: 需要旋转 180° 的 /dev/video 物理设备号集合
        rotate_ids = set(rotate_ids or [])
        self.cameras = {}
        for cam_id in camera_ids:
            self.cameras[cam_id] = MultiProcessCamera(
                cam_id, width, height, fps, rotate_180=(cam_id in rotate_ids)
            )
        self.fps = fps
        self.is_recording = False
    
    def start_all(self, output_path: Path) -> list:
        """
        并行启动所有相机进程，同时等待各相机就绪信号。
        返回成功启动的相机 ID 列表
        """
        import threading

        results: dict[int, bool] = {}
        results_lock = threading.Lock()

        def _start_one(cam_id: int, camera: 'MultiProcessCamera'):
            ok = camera.start(output_path)
            with results_lock:
                results[cam_id] = ok
            print(f"相机 {cam_id}: {'进程启动成功' if ok else '进程启动失败'}")

        threads = [
            threading.Thread(target=_start_one, args=(cam_id, camera), daemon=True)
            for cam_id, camera in self.cameras.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)  # 并行等待，总耗时约等于最慢单台相机的时间

        # 按原始相机列表顺序返回成功列表
        return [cam_id for cam_id in self.cameras if results.get(cam_id, False)]
    
    def begin_recording(self, countdown_seconds: float = 0.0):
        """同时开始所有相机的录制
        
        Args:
            countdown_seconds: 倒计时秒数，用于预热和同步
        """
        self.is_recording = True
        
        # 计算同步开始时间点（给足时间让所有进程准备好）
        sync_time = time.perf_counter() + max(0.1, countdown_seconds)
        
        # 同时发送开始信号（携带同步时间点）
        for camera in self.cameras.values():
            camera.begin_recording(sync_time)
        
        return sync_time
    
    def stop_all(self):
        """停止所有相机"""
        self.is_recording = False
        for camera in self.cameras.values():
            camera.stop()
    
    def get_all_previews(self):
        """获取所有相机的预览帧"""
        frames = {}
        for cam_id, camera in self.cameras.items():
            ret, frame = camera.get_preview_frame()
            if ret:
                frames[cam_id] = frame
        return frames
    
    def get_all_stats(self):
        """获取所有相机的统计信息"""
        stats = {}
        for cam_id, camera in self.cameras.items():
            s = camera.get_stats()
            if s:
                stats[cam_id] = s
        return stats
