#!/usr/bin/env python3
"""
ROS2 多相机图像发布节点
每个相机以独立线程采集，发布到 /helmet/cam{id}/image_raw

用法:
    # source ROS2 环境后运行
    python3 scripts/ros2_camera_publisher.py

    # 或覆盖相机ID / 帧率:
    ros2 run <pkg> ros2_camera_publisher.py \
        --ros-args -p camera_ids:=[0,2,4] -p fps:=30

话题:
    /helmet/cam{id}/image_raw   (sensor_msgs/msg/Image, encoding=bgr8)

参数 (ROS2 params):
    camera_ids  (list[int])  默认来自 config/config.yaml
    width       (int)        默认 640
    height      (int)        默认 480
    fps         (int)        默认 30
    config_path (str)        config/config.yaml 绝对路径，默认自动查找
"""

import sys
import os
import threading
import time
from pathlib import Path

# ── 把项目根目录加入路径，以便读取配置文件 ──
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import cv2
import numpy as np
import yaml

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header
    from builtin_interfaces.msg import Time as RosTime
except ImportError:
    print("ERROR: rclpy not found. Please source your ROS2 environment first.")
    print("  source /opt/ros/<distro>/setup.bash")
    sys.exit(1)

# BEST_EFFORT + depth=1：永远发最新帧，旧帧直接丢弃，最低延迟
_QOS_REALTIME = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def _resolve_cam_to_video(camera_ids: list[int], raw_index_map: dict) -> dict[int, int]:
    """统一 index_map 为 cam_id -> /dev/video_id，兼容新旧两种配置方向。"""
    ids = [int(x) for x in camera_ids]
    id_set = set(ids)
    raw = {int(k): int(v) for k, v in (raw_index_map or {}).items()}
    if not raw:
        return {cid: cid for cid in ids}

    keys = set(raw.keys())
    vals = set(raw.values())

    # 新格式：cam -> video（value 在 camera.ids 里）
    if vals.issubset(id_set):
        return {cam: vid for cam, vid in raw.items() if vid in id_set}

    # 旧格式：video -> cam（key 在 camera.ids 里）
    if keys.issubset(id_set):
        return {cam: vid for vid, cam in raw.items() if vid in id_set}

    # 回退：尽量按新格式解释
    return {cam: vid for cam, vid in raw.items()}


# ─────────────────────────────────────────────
# 工具函数：V4L2 正确属性设置顺序
# ─────────────────────────────────────────────

def _open_camera(camera_id: int, width: int, height: int, fps: int):
    """
    按正确顺序打开相机并设置 V4L2 参数。
    顺序必须是: FOURCC → 分辨率 → FPS → BUFFERSIZE
    """
    backend = (cv2.CAP_V4L2
               if os.name == 'posix' and hasattr(cv2, 'CAP_V4L2')
               else cv2.CAP_ANY)
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   2)      # 双缓冲，防止 15fps 陷阱

    # 预热：冲刷初始缓冲帧
    for _ in range(4):
        cap.grab()

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = ''.join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    print(f"[Cam{camera_id}] {fourcc_str} {actual_w}x{actual_h} @ {actual_fps:.0f}fps  "
          f"backend={cap.getBackendName()}")
    return cap


def _make_image_msg(width: int, height: int, frame_id: str) -> Image:
    """
    预分配一个 Image 消息，后续每帧只更新 stamp 和 data，
    避免每次创建 Python 对象的开销。
    """
    msg = Image()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.height   = height
    msg.width    = width
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step     = width * 3
    # 预分配 bytearray，后续直接原地写入，避免每帧 tobytes() 分配新内存
    msg.data = bytearray(height * width * 3)
    return msg


# ─────────────────────────────────────────────
# ROS2 节点
# ─────────────────────────────────────────────

class CameraPublisherNode(Node):
    """多相机图像发布节点（无 GUI，无录制）"""

    def __init__(self):
        super().__init__('helmet_camera_publisher')

        # ── 读取配置文件 ──
        config = self._load_config()
        cam_cfg = config.get('camera', {})

        # ── 声明 ROS2 参数（可在命令行覆盖）──
        self.declare_parameter('camera_ids', cam_cfg.get('ids', [0]))
        self.declare_parameter('width',      cam_cfg.get('width',  640))
        self.declare_parameter('height',     cam_cfg.get('height', 480))
        self.declare_parameter('fps',        30)

        camera_ids = [int(x) for x in self.get_parameter('camera_ids').value]
        width      = int(self.get_parameter('width').value)
        height     = int(self.get_parameter('height').value)
        fps        = int(self.get_parameter('fps').value)
        cam_to_video = _resolve_cam_to_video(camera_ids, cam_cfg.get('index_map', {}))

        self.get_logger().info(
            f"Cam->Video map={cam_to_video}  {width}x{height} @ {fps}fps"
        )

        # ── 为每个相机创建发布者和采集线程 ──
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        for topic_cam_id in sorted(cam_to_video.keys()):
            cam_id = int(cam_to_video[topic_cam_id])
            topic = f'/helmet/cam{topic_cam_id}/image_raw'
            pub = self.create_publisher(Image, topic, qos_profile=_QOS_REALTIME)
            self.get_logger().info(
                f"Publishing /dev/video{cam_id} as Cam{topic_cam_id} → {topic}"
            )

            t = threading.Thread(
                target=self._camera_loop,
                args=(cam_id, topic_cam_id, width, height, fps, pub),
                daemon=True,
                name=f'cam{cam_id}',
            )
            self._threads.append(t)
            t.start()

    # ── 私有方法 ──

    def _load_config(self) -> dict:
        config_path = _HERE / 'config' / 'config.yaml'
        if not config_path.exists():
            config_path = _HERE / 'config.yaml'
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                self.get_logger().warn(f"配置文件读取失败 ({config_path}): {e}")
        return {}

    def _camera_loop(
        self,
        camera_id: int,
        topic_cam_id: int,
        width: int,
        height: int,
        fps: int,
        pub,
    ):
        """单相机采集 + 发布循环（运行在独立线程）"""
        cap = _open_camera(camera_id, width, height, fps)
        if cap is None:
            self.get_logger().error(f"[Cam{camera_id}] 无法打开相机，线程退出")
            return

        # ── 预分配消息，每帧只更新 stamp + data，避免重复分配 Python 对象 ──
        msg = _make_image_msg(width, height, f'cam{topic_cam_id}')
        # numpy view 直接指向 msg.data 的内存，写入 view == 写入消息
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)

        frames_published = 0
        t_report = time.monotonic()

        while not self._stop_event.is_set():
            # ── 无订阅者时只 grab() 维持相机心跳，跳过 MJPEG 解码和发布 ──
            if pub.get_subscription_count() == 0:
                cap.grab()
                time.sleep(0.01)
                continue

            # grab() 阻塞等待硬件帧，由硬件帧率驱动，无需 sleep
            if not cap.grab():
                self.get_logger().warn(f"[Cam{camera_id}] grab() 失败，重试...")
                time.sleep(0.1)
                continue

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue

            # ── 零拷贝写入预分配缓冲区 ──
            # 确保 frame 是 C 连续内存，再用 np.copyto 原地写入，避免 tobytes() 额外分配
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            np.copyto(buf, frame)

            # 仅更新时间戳，其余字段保持预分配状态
            msg.header.stamp = self.get_clock().now().to_msg()
            pub.publish(msg)
            frames_published += 1

            # 每 5 秒打印一次实际帧率
            now = time.monotonic()
            dt = now - t_report
            if dt >= 5.0:
                self.get_logger().info(
                    f"[Cam{camera_id}] {frames_published / dt:.1f} fps  "
                    f"({frames_published} frames / {dt:.1f}s)"
                )
                frames_published = 0
                t_report = now

        cap.release()
        self.get_logger().info(f"[Cam{camera_id}] 线程已退出")

    def destroy_node(self):
        self.get_logger().info("节点关闭，停止所有相机线程...")
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=3.0)
        super().destroy_node()


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

def main():
    rclpy.init()
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
