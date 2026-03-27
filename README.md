# Helmet Recorder Tools

多相机 + IMU 数据采集头盔工具集。支持独立 GUI 录制（无需 ROS）和 ROS1 Noetic 驱动两种使用模式。

---

## 目录

1. [硬件与依赖](#1-硬件与依赖)
2. [安装](#2-安装)
3. [配置文件](#3-配置文件)
4. [ArUco 相机校准（必须）](#4-aruco-相机校准必须)
5. [独立录制 GUI（无需 ROS）](#5-独立录制-gui无需-ros)
6. [ROS1 驱动模式](#6-ros1-驱动模式)
   - [编译](#61-编译)
   - [Launch 参数说明](#62-launch-参数说明)
   - [常用启动命令](#63-常用启动命令)
7. [实用工具脚本](#7-实用工具脚本)

---

## 1. 硬件与依赖

| 组件 | 说明 |
|------|------|
| 相机 | 8 路 USB 相机，支持 MJPG 格式（USB 带宽允许 8×640×480@30fps） |
| IMU | Yesense 系列 IMU，串口连接 |
| 系统 | Ubuntu 20.04，Python ≥ 3.8 |
| ROS1 | Noetic（仅 ROS1 模式需要） |

---

## 2. 安装

### 2.1 安装 Python 依赖（uv）

```bash
# 安装 uv（如未安装）
curl -Lsf https://astral.sh/uv/install.sh | sh

# 在项目根目录创建虚拟环境并安装依赖
cd /path/to/helmet_recorder_tools
uv sync
```

---

## 3. 配置文件

所有系统配置位于 `config/config.yaml`：

```yaml
imu:
  port: /dev/ttyACM0      # IMU 串口设备
  bps: 460800             # 波特率

camera:
  ids: [0, 2, 4, 6, 8, 10, 12, 14]   # /dev/video 设备 ID 列表
  index_map:              # 逻辑相机编号 → /dev/video ID 映射
    0: 8                  # cam0 对应 /dev/video8
    1: 14
    2: 2
    3: 10
    4: 0
    5: 6
    6: 4
    7: 12
  width: 640
  height: 480
```

**查找 IMU 串口**（拔插检测）：

```bash
uv run scripts/imu_find_port.py
# 检测到端口后会询问是否自动写入 config.yaml
# 或使用 --auto-write 跳过确认
uv run scripts/imu_find_port.py --auto-write
```

---

## 4. ArUco 相机校准（必须）

> ⚠️ **重要：每次重新连接相机或系统重启后，USB 设备枚举顺序可能发生变化，导致 `/dev/video*` 编号与物理相机位置错位。必须在使用录制或 ROS1 发布前运行一次 ArUco 校准，以确保相机编号顺序正确。**

校准使用 **ArUco marker_0**（ID=0，字典 DICT_6X6_1000）作为公共参考点，脚本会自动识别各相机看到 marker 的位置关系，重新排列 `index_map` 并写入 `config/config.yaml`。

![ArUco 校准示意图](docs/images/ArUco%20Calibration.png)

打印 ArUco marker_0（推荐边长 **10 cm**，纸张平整粘在硬板上），放置在所有相机均可见的位置。

#### 运行校准脚本

```bash
uv run scripts/cam_calib_standalone.py
```

标定窗口会显示所有相机的实时画面和 ArUco 检测状态：

```
┌──────────┬──────────┬──────────┬──────────┐
│  Cam0    │  Cam1    │  Cam2    │  Cam3    │
│ [marker] │ [marker] │          │ [marker] │
│ ✅ 45smp │ ✅ 52smp │ ❌ 未检测 │ ✅ 48smp │
├──────────┼──────────┼──────────┼──────────┤
│  Cam4    │  Cam5    │  Cam6    │  Cam7    │
│ [marker] │          │ [marker] │ [marker] │
│ ✅ 50smp │ ❌ 未检测 │ ✅ 39smp │ ✅ 61smp │
└──────────┴──────────┴──────────┴──────────┘
```

| 按键 | 功能 |
|------|------|
| `y` | 开始收集样本（约 5 秒），完成后自动计算并写入 `config/config.yaml` |
| `q` | 退出 |

标定成功后，`config.yaml` 中会写入 `camera.extrinsics` 和 `camera.index_map`。

---

## 5. 独立录制 GUI（无需 ROS）

> ⚠️ **前置条件：请先完成 [ArUco 相机校准](#4-aruco-相机校准必须)，确保 `config.yaml` 中 `index_map` 正确。**

直接使用 `uv` 运行，**无需安装 ROS**：

```bash
uv run scripts/recorder_gui.py
```

### 界面说明

**左侧控制面板**：

| 控件 | 说明 |
|------|------|
| Camera IDs | /dev/video 设备 ID 列表，可手动编辑或点击「扫描」自动填充 |
| ↻ 扫描可用相机 | 自动扫描 `/dev/video*` 并按 config 顺序排列 |
| 分辨率 / FPS | 录制参数（默认 640×480 @ 30fps） |
| 输出目录 | 录像保存路径（默认 `~/recordings`） |
| IMU Port / Baud | 串口设备和波特率 |

**录制流程**：

1. 点击「扫描可用相机」确认 8 路相机全部识别
2. 确认 IMU Port 正确（可通过 `imu_find_port.py` 查找）
3. 点击「开始录制」→ 倒计时 3 秒后同步启动所有相机和 IMU
4. 录制完成后点击「停止录制」

**输出文件结构**：

```
~/recordings/recording_20260307_143022/
├── 0.mp4        # cam0 视频
├── 1.mp4        # cam1 视频
├── ...
├── 7.mp4        # cam7 视频
└── imu_data.txt # IMU 时序数据
```

---

## 6. ROS1 驱动模式

> ⚠️ **前置条件：请先完成 [ArUco 相机校准](#4-aruco-相机校准必须)，确保 `config.yaml` 中 `index_map` 正确。**

### 6.1 安装与编译

#### 方法一：软链接（推荐）

适合本地开发，修改 Python 源码无需重新编译：

```bash
# 1. 创建软链接到 catkin 工作空间（必须使用绝对路径）
mkdir -p ~/catkin_ws/src
ln -s /path/to/helmet_recorder_tools/helmet_recorder_ros ~/catkin_ws/src/

# 2. 编译
cd ~/catkin_ws
catkin_make

# 3. 每次新终端都需要 source
source devel/setup.bash
```

**注意事项：**
- **必须使用绝对路径**创建软链接，相对路径会导致 catkin 编译失败
- 软链接方式下，`config/config.yaml` 会自动通过 `rospkg` 定位到项目根目录（`helmet_recorder_tools/config/`）

#### 方法二：直接复制（适合部署）

```bash
# 1. 复制整个包到 catkin 工作空间
cp -r /path/to/helmet_recorder_tools/helmet_recorder_ros ~/catkin_ws/src/

# 2. 同时需要复制 config 目录（或在 launch 时指定绝对路径）
cp -r /path/to/helmet_recorder_tools/config ~/catkin_ws/src/helmet_recorder_ros/

# 3. 编译
cd ~/catkin_ws && catkin_make && source devel/setup.bash
```

#### 验证安装

```bash
# 检查包是否被识别
rospack find helmet_recorder_ros

# 检查 Python 模块是否可导入
python3 -c "import helmet_recorder_ros; print('OK')"

# 检查节点是否可执行
rosrun helmet_recorder_ros camera_publisher.py --help
```

### 6.2 Launch 参数说明

Launch 文件：`helmet_recorder_ros/launch/capture_system.launch`

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `config_path` | `''` | 任意路径 | 配置文件路径，留空时自动查找当前目录下的 `config/config.yaml` |
| `enable_imu` | `true` | `true` / `false` | 是否启动 IMU 发布节点，`false` 时仅发布相机话题 |
| `enable_viewer` | `true` | `true` / `false` | 是否启动可视化窗口，`false` 为无头模式（纯话题发布） |
| `high_res` | `true` | `true` / `false` | 可视化分辨率：`true`=640×480（高清），`false`=160×120（优先帧率） |

**节点自动选择逻辑**：

```
enable_imu=true  + enable_viewer=true  → camera_imu_vis（相机+IMU联合可视化）
enable_imu=false + enable_viewer=true  → multi_cam_vis（仅相机网格）
任意              + enable_viewer=false → 无可视化（仅发布话题）
```

**退出行为**：关闭可视化窗口（按 `q`）会自动触发所有节点退出（`required="true"`）。

### 6.3 常用启动命令

**完整模式（相机 + IMU + 可视化）**：

```bash
roslaunch helmet_recorder_ros capture_system.launch
```

**高分辨率可视化**（默认即为高清）：

```bash
roslaunch helmet_recorder_ros capture_system.launch high_res:=true
```

**低分辨率模式**（优先帧率，适合性能受限场景）：

```bash
roslaunch helmet_recorder_ros capture_system.launch high_res:=false
```

**仅相机，无 IMU**：

```bash
roslaunch helmet_recorder_ros capture_system.launch enable_imu:=false
```

**无头模式**（仅发布话题，不显示窗口）：

```bash
roslaunch helmet_recorder_ros capture_system.launch enable_viewer:=false
```

**指定配置文件路径**：

```bash
roslaunch helmet_recorder_ros capture_system.launch \
    config_path:=/path/to/config/config.yaml
```

**可视化窗口按键**：

| 按键 | 功能 |
|------|------|
| `q` | 退出（并关闭所有节点） |
| `f` | 切换全屏 |

**话题列表**：

```bash
# 相机话题
/helmet/cam0/image_raw  ...  /helmet/cam7/image_raw

# IMU 话题
/helmet/imu/data
```

---

## 7. 实用工具脚本

| 脚本 | 用途 |
|------|------|
| `uv run scripts/imu_find_port.py` | 拔插检测 IMU 串口，可自动写入 config |
| `uv run scripts/cam_extrinsics_visualizer.py` | 3D 可视化相机外参布局 |
| `uv run scripts/system_check.py` | 系统环境检查（相机、IMU、依赖） |
