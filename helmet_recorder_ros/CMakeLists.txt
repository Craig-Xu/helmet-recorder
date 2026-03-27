cmake_minimum_required(VERSION 3.0.2)
project(helmet_recorder_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  src/helmet_recorder_ros/camera_publisher.py
  src/helmet_recorder_ros/imu_publisher.py
  src/helmet_recorder_ros/camera_imu_vis.py
  src/helmet_recorder_ros/multi_cam_vis.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
