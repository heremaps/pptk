# # Uncomment the following lines to run as standalone cmake script
# cmake_minimum_required(VERSION 3.0)
# project(estimate_normals)

# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake" ${CMAKE_CURRENT_SOURCE_DIR})
# find_package(PythonLibs 2.7 REQUIRED)
# find_package(Numpy REQUIRED)
# find_package(TBB REQUIRED)
# find_package(Eigen REQUIRED)
# include(UsefulMacros)
# include(FindOpenMP)

add_library(estimate_normals SHARED estimate_normals.cpp)
set(_link_flags ${OpenMP_CXX_FLAGS})
if(WIN32)
  target_link_libraries(estimate_normals ${PYTHON_LIBRARY})
elseif(APPLE)
  set(_link_flags "${_link_flags} -undefined dynamic_lookup")
elseif(UNIX)
  set(_link_flags "${_link_flags} -static-libstdc++")
endif()
set_target_properties(estimate_normals PROPERTIES
  COMPILE_FLAGS ${OpenMP_CXX_FLAGS}
  LINK_FLAGS ${_link_flags})
set_target_python_module_name(estimate_normals)
target_compile_definitions(estimate_normals PRIVATE -DUSE_TBB)
target_link_libraries(estimate_normals
  ${TBB_tbb_LIBRARY}
  ${TBB_tbbmalloc_LIBRARY})
target_include_directories(estimate_normals PRIVATE
  ../../include     # for python_util.h
  ../../kdtree/src  # for kdtree.h
  ${Eigen_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIR}
  ${TBB_INCLUDE_DIR}
  ${Numpy_INCLUDE_DIR})
copy_target(estimate_normals)
copy_target_dependencies(estimate_normals)
copy_file(__init__.py)

