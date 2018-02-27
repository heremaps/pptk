# # Uncomment the following lines to run as standalone cmake script
# cmake_minimum_required(VERSION 3.0)
# project(vfuncs)

# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake" ${CMAKE_MODULE_PATH})
# find_package(PythonLibs 2.7 REQUIRED)
# find_package(Numpy REQUIRED)
# find_package(Eigen REQUIRED)
# include(UsefulMacros)

add_library(vfuncs SHARED vfuncs.cpp)
set_target_python_module_name(vfuncs)
if(WIN32)
    target_link_libraries(vfuncs ${PYTHON_LIBRARY})
elseif(APPLE)
  set_target_properties(vfuncs PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup")
elseif(UNIX)
  set_target_properties(vfuncs PROPERTIES
    LINK_FLAGS "-static-libstdc++")
endif()
target_include_directories(vfuncs PRIVATE
  ../include  # for python_util.h
  ${PYTHON_INCLUDE_DIR}
  ${Eigen_INCLUDE_DIR}
  ${Numpy_INCLUDE_DIR})
copy_target(vfuncs)
copy_target_dependencies(vfuncs)
copy_file(__init__.py)

