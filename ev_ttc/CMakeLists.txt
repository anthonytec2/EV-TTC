cmake_minimum_required(VERSION 3.5)

project(ev_ttc)
add_compile_options(-fPIC  -fno-omit-frame-pointer  -march=native  -funsafe-math-optimizations -O3 -ffast-math)


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(CUDA REQUIRED)

set(ROS2_DEPENDENCIES
  "rclcpp"
  "rclcpp_components"
  "event_camera_msgs"
  "event_camera_codecs"
  "sensor_msgs"
  "image_transport"
)

foreach(pkg ${ROS2_DEPENDENCIES})
  find_package(${pkg} REQUIRED)
endforeach()

ament_auto_find_build_dependencies(REQUIRED ${ROS2_DEPENDENCIES})


ament_auto_add_library(exp src/exp.cpp)
target_link_libraries(exp  nvinfer nvinfer_plugin ${CUDA_LIBRARIES}) # Package libnvinfer-dev, nvidia-tensorrt
target_include_directories(exp PUBLIC  ${CUDA_INCLUDE_DIRS})
rclcpp_components_register_nodes(exp "ev_ttc::Exp")

ament_auto_add_executable(exp_node src/exp_node.cpp)
target_link_libraries(exp_node exp ${CUDA_LIBRARIES})


install(TARGETS
  exp_node
  DESTINATION lib/${PROJECT_NAME}/)


install(TARGETS
  exp
  DESTINATION lib/
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
  FILES_MATCHING PATTERN "*.py")

ament_package()