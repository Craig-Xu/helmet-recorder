#!/usr/bin/env python3
"""
多进程相机管理模块 - 每个相机运行在独立进程中，完全绑过Python GIL
"""

import cv2
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
from ctypes import c_uint64, c_double, c_bool


def camera_worker_process(
    camera_id: int,
    width: int,
    height: int,
    fps: int,
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
        # 打开相机
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            error_msg.value = f"无法打开相机 {camera_id}".encode('utf-8')
            ready_event.set()
            return
        
        # 配置相机 - MJPG格式减少USB带宽
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 读取第一帧验证相机工作
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.05)
        
        if not ret:
            error_msg.value = f"相机 {camera_id} 无法读取帧".encode('utf-8')
            cap.release()
            ready_event.set()
            return
        
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
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height), True)
        
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                if wait_time > 0.002:
                    time.sleep(wait_time - 0.001)
                # 忙等待最后一点时间
                while time.perf_counter() < target_start:
                    pass
        
        # 同步点：所有相机应该同时到达这里
        # 初始化计数器
        frame_count = 0
        new_frames = 0
        duplicated = 0
        dropped = 0
        preview_seq = 0
        last_good_frame = None
        
        # 立即抓取并处理第一帧（这是同步的关键帧）
        ret = cap.grab()
        if ret:
            ret, first_frame = cap.retrieve()
            if ret and first_frame is not None:
                if first_frame.shape[1] != width or first_frame.shape[0] != height:
                    first_frame = cv2.resize(first_frame, (width, height))
                writer.write(first_frame)
                np.copyto(preview_buffer, first_frame)
                last_good_frame = first_frame
                frame_count = 1
                new_frames = 1
                preview_seq = 1
                stats_array[6] = preview_seq
        
        # 开始录制主循环
        frame_interval = 1.0 / fps
        actual_start = time.perf_counter()
        stats_array[4] = actual_start  # start_time
        stats_array[5] = 1  # is_running = True
        
        next_frame_time = actual_start + frame_interval  # 下一帧的目标时间
        
        while not stop_event.is_set():
            current_time = time.perf_counter()
            
            # 精确帧率控制
            if current_time < next_frame_time:
                sleep_time = next_frame_time - current_time
                if sleep_time > 0.001:
                    time.sleep(sleep_time - 0.001)
                # 忙等待最后1ms
                while time.perf_counter() < next_frame_time:
                    pass
            
            # 读取帧（非阻塞式grab + retrieve可以更好控制）
            ret = cap.grab()
            if ret:
                ret, frame = cap.retrieve()
            
            if ret and frame is not None:
                # 确保分辨率匹配
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                last_good_frame = frame
                new_frames += 1
                
                # 写入视频
                writer.write(frame)
                frame_count += 1
                
                # 更新预览帧到共享内存（低频率更新，减少开销）
                if frame_count % 3 == 0:  # 每3帧更新一次预览
                    np.copyto(preview_buffer, frame)
                    preview_seq += 1
                    stats_array[6] = preview_seq
                    
            elif last_good_frame is not None:
                # 相机没给新帧，复用上一帧
                writer.write(last_good_frame)
                frame_count += 1
                duplicated += 1
            else:
                # 黑帧
                black = np.zeros((height, width, 3), dtype=np.uint8)
                writer.write(black)
                frame_count += 1
                dropped += 1
            
            # 更新统计到共享内存
            stats_array[0] = frame_count
            stats_array[1] = new_frames
            stats_array[2] = duplicated
            stats_array[3] = dropped
            
            next_frame_time += frame_interval
            
            # 防止累积延迟（如果严重落后，重置时间）
            if time.perf_counter() - next_frame_time > 0.5:
                next_frame_time = time.perf_counter()
        
        # 清理资源
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
    
    def __init__(self, camera_id: int, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        
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
            duplicated = int(self.stats_array[2])
            dropped = int(self.stats_array[3])
            start_time = float(self.stats_array[4])
            is_running = bool(self.stats_array[5])
            
            if start_time > 0:
                duration = time.perf_counter() - start_time
                fps = frame_count / duration if duration > 0 else 0
                hw_fps = new_frames / duration if duration > 0 else 0
            else:
                duration = 0
                fps = 0
                hw_fps = 0
            
            return {
                'frames': frame_count,
                'new_frames': new_frames,
                'duplicated': duplicated,
                'dropped': dropped,
                'duration': duration,
                'fps': fps,
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
    
    def __init__(self, camera_ids: list, width: int = 640, height: int = 480, fps: int = 30):
        self.cameras = {}
        for cam_id in camera_ids:
            self.cameras[cam_id] = MultiProcessCamera(cam_id, width, height, fps)
        self.fps = fps
        self.is_recording = False
    
    def start_all(self, output_path: Path) -> list:
        """
        启动所有相机进程
        返回成功启动的相机ID列表
        """
        successful = []
        for cam_id, camera in self.cameras.items():
            if camera.start(output_path):
                successful.append(cam_id)
                print(f"相机 {cam_id}: 进程启动成功")
            else:
                print(f"相机 {cam_id}: 进程启动失败")
        
        return successful
    
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
