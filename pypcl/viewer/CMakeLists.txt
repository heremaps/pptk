# # Uncomment the following lines to run as standalone cmake script
# cmake_minimum_required(VERSION 2.8.11)
# project(viewer)

# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake" ${CMAKE_MODULE_PATH})
# find_package(Eigen REQUIRED)
# find_package(OpenGL REQUIRED)
# find_package(Qt5 CONFIG REQUIRED COMPONENTS Widgets Network OpenGL Core)
# include(UsefulMacros)

add_executable(viewer
  main.cpp
  octree.h
  box3.h
  FloorGrid.h
  SelectionBox.h
  Background.h
  LookAt.h
  PointCloud.h
  Viewer.h
  QtCamera.h
  Camera.h
  CommFuncs.h
  OpenGLFuncs.h
  Text.h
  CameraDolly.h
  Splines.h
  PointAttributes.h)

set_target_properties(viewer PROPERTIES AUTOMOC TRUE)
target_link_libraries(viewer Qt5::Widgets Qt5::Network Qt5::OpenGL ${OPENGL_gl_LIBRARY})
target_include_directories(viewer PRIVATE ${Eigen_INCLUDE_DIR} )
copy_target(viewer)
copy_target_dependencies(viewer)
copy_file(viewer.py)
copy_file(__init__.py)
copy_file(qt.conf)

