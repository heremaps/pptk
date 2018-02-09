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
copy_file(viewer.py)
copy_file(__init__.py)
copy_file(qt.conf)

# deploy runtime dll's to libs folder
get_target_property(_qmake_exe Qt5::qmake IMPORTED_LOCATION)
get_filename_component(_qt_bin ${_qmake_exe} DIRECTORY)
if (WIN32)
  add_custom_command(TARGET viewer POST_BUILD 
    COMMAND ${_qt_bin}/windeployqt.exe $<TARGET_FILE:viewer>
      --libdir ${CMAKE_CURRENT_BINARY_DIR}/../libs
      --no-compiler-runtime
      --no-translations
      --no-quick-import
      --no-webkit2
      --no-angle
      --no-system-d3d-compiler)
elseif (APPLE)
  add_custom_command(TARGET viewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -P
      ${PROJECT_SOURCE_DIR}/cmake/CopyAppleDependencies.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:viewer> ${PYPCL_LIBS_DIR})
elseif (UNIX)
  # TODO
endif()
unset(_qmake_exe)
unset(_qt_bin)
