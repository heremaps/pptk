# # Uncomment the following lines to run as standalone cmake script
# cmake_minimum_required(VERSION 3.0)
# project(kdtree)

# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake" ${CMAKE_CURRENT_SOURCE_DIR})
# find_package(PythonLibs 2.7 REQUIRED)
# find_package(Numpy REQUIRED)
# find_package(TBB REQUIRED)
# include(UsefulMacros)

add_library(kdtree SHARED kdtree_wrapper.cpp)
set_target_python_module_name(kdtree)
target_include_directories(kdtree PRIVATE
  ../include  # for python_util.h
  src         # for k-d tree source code
  ${TBB_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIR}
  ${Numpy_INCLUDE_DIR} )
target_link_libraries(kdtree
  ${TBB_tbb_LIBRARY}
  ${TBB_tbbmalloc_LIBRARY}
  ${PYTHON_LIBRARY} )
copy_target(kdtree)
copy_file(__init__.py)

if (WIN32)
  # TODO
elseif (APPLE)
  add_custom_command(TARGET kdtree POST_BUILD
    COMMAND ${CMAKE_COMMAND} -P
      ${PROJECT_SOURCE_DIR}/cmake/CopyAppleDependencies.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:vfuncs> ${PYPCL_LIBS_DIR})
elseif (UNIX)
  # TODO
endif()
