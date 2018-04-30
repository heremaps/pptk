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
  floor_grid.h
  selection_box.h
  background.h
  look_at.h
  point_cloud.h
  viewer.h
  qt_camera.h
  camera.h
  comm_funcs.h
  opengl_funcs.h
  text.h
  camera_dolly.h
  splines.h
  point_attributes.h)

set_target_properties(viewer PROPERTIES AUTOMOC TRUE)
target_link_libraries(viewer Qt5::Widgets Qt5::Network Qt5::OpenGL ${OPENGL_gl_LIBRARY})
target_include_directories(viewer PRIVATE ${Eigen_INCLUDE_DIR} )
copy_target(viewer)
copy_target_dependencies(viewer)
copy_file(viewer.py)
copy_file(__init__.py)
copy_file(qt.conf)

