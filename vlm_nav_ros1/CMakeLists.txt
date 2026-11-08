cmake_minimum_required(VERSION 3.0.2)
project(vlm_nav_ros1)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS nodes/vlm_nav_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
