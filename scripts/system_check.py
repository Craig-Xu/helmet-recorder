#!/usr/bin/env python3
"""
系统状态检查脚本
验证所有依赖和模块是否正确安装
"""

import sys
from pathlib import Path

print("=" * 60)
print("ROS2 Helmet Recorder 系统状态检查")
print("=" * 60)
print()

# 检查Python版本
print(f"✓ Python 版本: {sys.version.split()[0]}")

# 检查必要的包
required_packages = {
    'rclpy': 'ROS2 Python客户端',
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'yaml': 'PyYAML',
    'serial': 'PySerial (IMU通信)',
    'matplotlib': 'Matplotlib (可选,用于FPS监控)',
}

print("\n📦 依赖检查:")
print("-" * 60)

missing = []
for package, desc in required_packages.items():
    try:
        __import__(package)
        print(f"✅ {package:15s} - {desc}")
    except ImportError:
        print(f"❌ {package:15s} - {desc} [未安装]")
        missing.append(package)

# 检查ROS2模块
print("\n🔧 ROS2 模块检查:")
print("-" * 60)

try:
    from helmet_recorder_ros2.imu.imu_manager import IMUManager
    print("✅ IMU 模块导入")
except Exception as e:
    print(f"❌ IMU 模块导入失败: {e}")
    missing.append('imu_module')

try:
    from helmet_recorder_ros2.common import load_yaml
    print("✅ Common 模块导入")
except Exception as e:
    print(f"❌ Common 模块导入失败: {e}")

# 检查串口设备
print("\n🔌 串口设备检查:")
print("-" * 60)

import glob
tty_devices = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
if tty_devices:
    for dev in tty_devices:
        try:
            import os
            stat = os.stat(dev)
            readable = os.access(dev, os.R_OK)
            writable = os.access(dev, os.W_OK)
            status = "✅" if (readable and writable) else "⚠️ (权限不足)"
            print(f"{status} {dev}")
        except:
            print(f"⚠️  {dev} (无法访问)")
else:
    print("⚠️  未找到串口设备 (/dev/ttyACM* 或 /dev/ttyUSB*)")

# 检查相机设备
print("\n📷 相机设备检查:")
print("-" * 60)

video_devices = glob.glob('/dev/video*')
if video_devices:
    # 只显示偶数设备（通常是主设备）
    main_devices = [d for d in video_devices if int(d.split('video')[-1]) % 2 == 0]
    for dev in main_devices[:8]:  # 最多显示8个
        print(f"✅ {dev}")
    if len(main_devices) > 8:
        print(f"   ... 还有 {len(main_devices) - 8} 个设备")
else:
    print("⚠️  未找到相机设备")

# 总结
print("\n" + "=" * 60)
print("📊 总结:")
print("=" * 60)

if not missing:
    print("✅ 所有依赖已安装")
else:
    print(f"❌ 缺少 {len(missing)} 个依赖:")
    for pkg in missing:
        if pkg == 'serial':
            print(f"   - {pkg:15s} → pip install pyserial")
        elif pkg == 'matplotlib':
            print(f"   - {pkg:15s} → pip install matplotlib")
        else:
            print(f"   - {pkg:15s}")

if not tty_devices:
    print("\n⚠️  提醒:")
    print("   - 未检测到 IMU 串口设备")
    print("   - 如果已连接，请检查驱动和权限")
    print("   - 运行: sudo usermod -a -G dialout $USER")

if 'serial' not in missing and tty_devices:
    print("\n✅ 可以启动 IMU 发布节点")
    print("   ros2 run helmet_recorder_ros2 imu_publisher")

if len(video_devices) >= 8:
    print("\n✅ 可以启动完整系统")
    print("   ros2 launch helmet_recorder_ros2 camera_imu_full.launch.py")
elif len(video_devices) > 0:
    print(f"\n⚠️  只检测到 {len(video_devices)//2} 个相机")
    print("   系统需要 8 个相机")

print("\n" + "=" * 60)
