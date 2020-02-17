cmake_minimum_required(VERSION 2.8.3)
set (CMAKE_CXX_STANDARD 14)
project(cam_lidar_calib)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  message_filters
  cv_bridge
  tf
  tf_conversions
)

find_package( PCL REQUIRED)
find_package( OpenCV REQUIRED)
find_package( Ceres REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
  DEPENDS PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)


add_executable(${PROJECT_NAME}_node src/cam_lidar_calib_node.cpp)

add_executable(cam_lidar_proj_node src/cam_lidar_proj_node.cpp)


target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
   ${CERES_LIBRARIES}
)

target_link_libraries(cam_lidar_proj_node
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        )
