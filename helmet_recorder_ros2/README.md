# helmet_recorder_ros2

ROS2 驱动包（ament_python），用于多相机发布、ArUco 标定与外参保存。

## 构建

```bash
cd <workspace_root>
colcon build --packages-select helmet_recorder_ros2
source install/setup.bash
```

## 运行

```bash
ros2 run helmet_recorder_ros2 camera_publisher
ros2 run helmet_recorder_ros2 aruco_calib
ros2 run helmet_recorder_ros2 save_cam_extrinsics
ros2 run helmet_recorder_ros2 image_calib_save_extrinsics
```

`image_calib_save_extrinsics` 交互说明：

- 按 `y`：开始采样并在约 5 秒后写入 `config/config.yaml`
- 按 `q`：退出

关闭交互窗口模式（自动采样保存）：

```bash
ros2 run helmet_recorder_ros2 image_calib_save_extrinsics --ros-args -p interactive:=false
```

可视化：

```bash
ros2 run helmet_recorder_ros2 multi_cam_vis
ros2 run helmet_recorder_ros2 multi_cam_aruco_vis
```

## 参数

所有节点支持参数：
- `config_path`：配置文件路径（默认读取当前工作目录 `config/config.yaml`）

例如：

```bash
ros2 run helmet_recorder_ros2 camera_publisher --ros-args -p config_path:=/path/to/config/config.yaml
```
