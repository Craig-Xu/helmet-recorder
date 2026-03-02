import cv2
import numpy as np
import threading
import time
import argparse
import yaml
import os
import re
from pathlib import Path

class CameraStream:
    """
    多线程摄像头流读取类，用于减少顺序读取导致的卡顿
    """
    def __init__(self, cam_id, width=640, height=480):
        self.cap = cv2.VideoCapture(cam_id)
        # 使用MJPG格式大幅降低USB带宽占用（压缩传输 vs 原始YUV）
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲减少延迟
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 请求相机以30fps输出
        
        self.width = width
        self.height = height
        self.ret = False
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.stopped = False
        self.ready = False  # 标记是否已读取到第一帧
        self.frame_seq = 0  # 帧序列号，每次新帧递增
        
        if self.cap.isOpened():
            self.thread = threading.Thread(target=self.update, args=(), daemon=True)
            self.thread.start()
            # 等待第一帧读取成功（最多等待1秒）
            for _ in range(100):
                if self.ready:
                    break
                time.sleep(0.01)

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            ret, frame = self.cap.read()
            if ret:
                # 直接使用读取的帧，不resize（相机已设置为目标分辨率）
                # 如果相机分辨率不匹配，可能需要resize
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    self.frame = cv2.resize(frame, (self.width, self.height))
                else:
                    self.frame = frame
                self.ret = True
                self.frame_seq += 1
                if not self.ready:
                    self.ready = True
            # cap.read()本身是阻塞的，按相机帧率返回，不需要额外sleep

    def read(self):
        return self.ret, self.frame
    
    def read_with_seq(self):
        """返回帧和序列号，用于检测是否有新帧"""
        return self.ret, self.frame, self.frame_seq

    def stop(self):
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()

def _parse_video_index(device_path: Path):
    """从 /dev/videoX 路径中解析索引 X。"""
    match = re.fullmatch(r"video(\d+)", device_path.name)
    if not match:
        return None
    return int(match.group(1))


def _try_open_camera(source, backends):
    """尝试使用多个后端打开相机，返回 (is_opened, backend_name)。"""
    for backend in backends:
        cap = None
        try:
            if backend is None:
                cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source, backend)

            if cap.isOpened():
                backend_name = cap.getBackendName()
                ret, _ = cap.read()
                if ret:
                    return True, backend_name
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()

    return False, None


def list_cameras(max_to_test=10):
    """
    列出所有可用的摄像头 ID 并排序

    Linux: 优先扫描 /dev/video* 并使用 V4L2 后端，避免默认后端导致漏检。
    其他平台: 回退到索引扫描。
    """
    available_cameras = set()
    print("正在扫描摄像头...")

    preferred_backends = []
    if hasattr(cv2, 'CAP_V4L2'):
        preferred_backends.append(cv2.CAP_V4L2)
    preferred_backends.append(None)  # 默认后端回退

    if os.name == 'posix' and Path('/dev').exists():
        video_devices = sorted(
            Path('/dev').glob('video*'),
            key=lambda p: (_parse_video_index(p) is None, _parse_video_index(p) or 10**9)
        )

        for dev in video_devices:
            idx = _parse_video_index(dev)
            if idx is None:
                continue

            ok, backend_name = _try_open_camera(str(dev), preferred_backends)
            if ok:
                available_cameras.add(idx)
                print(f"找到摄像头 ID: {idx} ({backend_name}, {dev})")

        # Linux 下如果 /dev 扫描没有结果，回退到索引扫描
        if not available_cameras:
            for i in range(max_to_test):
                ok, backend_name = _try_open_camera(i, preferred_backends)
                if ok:
                    available_cameras.add(i)
                    print(f"找到摄像头 ID: {i} ({backend_name})")
    else:
        for i in range(max_to_test):
            ok, backend_name = _try_open_camera(i, preferred_backends)
            if ok:
                available_cameras.add(i)
                print(f"找到摄像头 ID: {i} ({backend_name})")

    return sorted(available_cameras)

def load_config(config_path=None):
    """
    从配置文件加载相机设置
    """
    if config_path is None:
        # 默认配置文件路径（脚本所在目录的上级目录）
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('camera', {})
    except Exception as e:
        print(f"警告: 读取配置文件失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='多相机采集系统')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径 (默认: ../config/config.yaml)')
    parser.add_argument('--ids', type=str, default=None,
                       help='相机ID列表，用逗号或空格分隔 (例如: "0,1,2,3")')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式选择相机 (忽略配置文件和--ids参数)')
    parser.add_argument('--width', type=int, default=None,
                       help='相机分辨率宽度')
    parser.add_argument('--height', type=int, default=None,
                       help='相机分辨率高度')
    
    args = parser.parse_args()
    
    # 读取配置文件
    camera_config = load_config(args.config)
    
    # 确定相机分辨率
    width = args.width or (camera_config.get('width') if camera_config else 640) or 640
    height = args.height or (camera_config.get('height') if camera_config else 480) or 480
    max_cols = (camera_config.get('display', {}).get('max_cols') if camera_config else 4) or 4
    
    # 1. 罗列并排序摄像头
    available_ids = list_cameras()
    
    if not available_ids:
        print("未发现任何可用的摄像头。")
        return

    print(f"\n当前可用的摄像头 ID 列表: {available_ids}")
    
    selected_ids = []
    
    # 确定使用哪种方式选择相机
    if args.interactive:
        # 交互式输入
        user_input = input("请输入你想打开的摄像头 ID (多个请用空格或逗号分隔，例如 0,1): ")
        try:
            raw_ids = user_input.replace(',', ' ').split()
            selected_ids = [int(i) for i in raw_ids if int(i) in available_ids]
        except ValueError:
            print("输入格式错误，请输入数字 ID。")
            return
    elif args.ids:
        # 命令行参数指定
        try:
            raw_ids = args.ids.replace(',', ' ').split()
            selected_ids = [int(i) for i in raw_ids if int(i) in available_ids]
            print(f"使用命令行参数指定的相机: {selected_ids}")
        except ValueError:
            print("命令行参数格式错误，请输入数字 ID。")
            return
    elif camera_config and 'ids' in camera_config:
        # 从配置文件读取
        config_ids = camera_config['ids']
        selected_ids = [i for i in config_ids if i in available_ids]
        print(f"使用配置文件中的相机: {selected_ids}")
        
        # 检查配置中的相机是否都可用
        unavailable = [i for i in config_ids if i not in available_ids]
        if unavailable:
            print(f"警告: 配置文件中的相机 {unavailable} 不可用")
    else:
        # 默认：交互式输入
        user_input = input("请输入你想打开的摄像头 ID (多个请用空格或逗号分隔，例如 0,1): ")
        # 默认：交互式输入
        user_input = input("请输入你想打开的摄像头 ID (多个请用空格或逗号分隔，例如 0,1): ")
        try:
            raw_ids = user_input.replace(',', ' ').split()
            selected_ids = [int(i) for i in raw_ids if int(i) in available_ids]
        except ValueError:
            print("输入格式错误，请输入数字 ID。")
            return
    
    # 解析用户输入
    # selected_ids = []
    # try:
    #     # 兼容逗号和空格分隔
    #     raw_ids = user_input.replace(',', ' ').split()
    #     selected_ids = [int(i) for i in raw_ids if int(i) in available_ids]
    # except ValueError:
    #     print("输入格式错误，请输入数字 ID。")
    #     return

    if not selected_ids:
        print("未选择任何有效的摄像头。")
        return

    print(f"正在配置摄像头: {selected_ids} ...")

    # 2. 使用多线程包装器打开选定的摄像头
    streams = []
    for cam_id in selected_ids:
        stream = CameraStream(cam_id, width, height)
        if stream.cap.isOpened():
            streams.append(stream)
        else:
            print(f"提示: 无法打开摄像头索引 {cam_id}")

    if not streams:
        print("错误: 没有成功打开任何摄像头。")
        return

    print("\n运行中... 点击弹出窗口后按 'q' 键退出。")

    try:
        while True:
            frames = []
            for stream in streams:
                ret, frame = stream.read()
                # 即使读取 ret 为 False (如掉线)，stream.frame 也会返回最后的占位图
                frames.append(frame)

            # 3. 拼接所有摄像头画面 (每行最多 max_cols 个)
            rows = []
            for i in range(0, len(frames), max_cols):
                chunk = frames[i : i + max_cols]
                # 如果最后一行不足 max_cols 个，用黑色图像填充对齐
                while len(chunk) < max_cols:
                    chunk.append(np.zeros((height, width, 3), dtype=np.uint8))
                
                # 横向拼接这一行
                row_frame = cv2.hconcat(chunk)
                rows.append(row_frame)

            # 纵向拼接所有行
            combined_frame = cv2.vconcat(rows)

            # 4. 实时显示
            cv2.imshow('Multi-Camera Preview (480p - Optimized)', combined_frame)

            # 检测退出按键 (1ms 足够检测，不再需要 30ms 补偿)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 5. 资源释放
        print("正在关闭摄像头线程...")
        for stream in streams:
            stream.stop()
        cv2.destroyAllWindows()
        print("程序已退出。")

if __name__ == "__main__":
    main()
