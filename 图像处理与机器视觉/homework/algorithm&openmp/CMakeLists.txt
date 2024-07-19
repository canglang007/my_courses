cmake_minimum_required(VERSION 3.10)
project(stereo_camera)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 添加头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加源文件
add_executable(${PROJECT_NAME}
  main.cpp
  val.cpp
  val.h
  sgbm.cpp
  sgbm.h
  d_type.h
  sad_ssd_ncc.cpp
  sad_ssd_ncc.h
  )

# 链接OpenCV库
target_link_libraries(stereo_camera ${OpenCV_LIBS})
